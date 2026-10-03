"""Phase-boundary takeover and multi-GPU scheduling of the unchanged frozen V1.2 code.

This is an orchestration artifact, not a new training source version. Its hash is
recorded independently; children import the prior run's verified source snapshot.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.v12.animal_experiment import AnimalExperiment, animal_gate, nominated_gate
from boardbench.v12.budget import Ledger, kill_tree
from boardbench.v12.experiment import rows
from boardbench.v12.launch import idle_gpu


def process(pid):
    """Linux identity + state, guarding every later signal against PID reuse."""
    try:
        root = Path('/proc') / str(pid)
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': stat[0], 'ppid': int(stat[1]), 'start': stat[19],
                'argv': (root / 'cmdline').read_bytes().decode().strip('\0').split('\0')}
    except FileNotFoundError:
        return None


def signal_checked(identity, sig):
    actual = process(identity['pid'])
    if not actual or actual['start'] != identity['start']:
        raise RuntimeError('process identity changed; refusing signal')
    os.kill(identity['pid'], sig)


def verify_phase_output(argv):
    if '--out' not in argv:
        raise ValueError('phase has no independently verifiable output')
    out = Path(argv[argv.index('--out') + 1])
    if 'boardbench.v12.training' in argv:
        report = read_json(out / 'summary.json')
        meta = read_json(out / 'final.resume.json')
        if report['status'] != 'completed' or digest(out / 'final.resume.pt') != meta['resume_sha256'] or digest(out / 'final.resume.weights.pt') != meta['weights_sha256']:
            raise ValueError('incomplete training checkpoint at handoff')
    elif 'boardbench.v12.evaluation' in argv:
        report = read_json(out / 'summary.json')
        if report['attempts'] != len(read_json(out / 'config.json')['seeds']):
            raise ValueError('incomplete evaluation at handoff')
    elif 'boardbench.v12.animal_diagnostics' in argv:
        read_json(out / 'report.json')
    else:
        raise ValueError('handoff only supports training/development diagnosis phases')
    return str(out)


def handoff(prior, deadline):
    """Stop scheduling new work, never interrupt the active child or edit its files."""
    allocation = read_json(prior / 'allocation.json')
    status = read_json(prior / 'status.json')
    if allocation['host'] != os.uname().nodename or (prior / 'frozen_selection.json').exists():
        raise ValueError('handoff must be on the original host before final selection/test')
    watchdog, parent = process(allocation['watchdog_pid']), process(status['pid'])
    if not watchdog or 'boardbench.v12.launch' not in watchdog['argv'] or not parent or 'boardbench.v12.animal_experiment' not in parent['argv']:
        raise ValueError('expected live original watchdog/controller not found')
    if parent['ppid'] != watchdog['pid'] or str(prior) not in parent['argv']:
        raise ValueError('original process ownership mismatch')
    signal_checked(parent, signal.SIGSTOP)
    committed = False
    try:
        for _ in range(100):
            if process(parent['pid'])['state'] in ('T', 't'):
                break
            time.sleep(.01)
        else:
            raise RuntimeError('controller did not reach stopped state')
        children = [process(int(pid)) for pid in Path(f'/proc/{parent["pid"]}/task/{parent["pid"]}/children').read_text().split()]
        children = [p for p in children if p]
        for child in children:
            if not any(module in child['argv'] for module in ('boardbench.v12.training', 'boardbench.v12.evaluation', 'boardbench.v12.animal_diagnostics')):
                raise ValueError('unsupported active child; leave original run in control')
        while any((p := process(child['pid'])) and p['start'] == child['start'] and p['state'] != 'Z' for child in children):
            if time.time() >= deadline - 1800:
                raise TimeoutError('insufficient original window for a safe parallel handoff')
            time.sleep(.25)
        completed = [verify_phase_output(child['argv']) for child in children]
        write_json(prior / 'parallel_handoff.json', {'reason': 'user requested independent multi-GPU jobs',
                   'completed_child_outputs': completed, 'parent_identity': parent,
                   'active_child_was_interrupted': False, 'requested_unix': time.time()})
        # Old watchdog writes its final accounting and releases the project lock.
        signal_checked(watchdog, signal.SIGTERM)
        committed = True
        until = time.monotonic() + 15
        while not (prior / 'watchdog_result.json').exists():
            if time.monotonic() > until:
                raise TimeoutError('old watchdog did not finish accounting')
            time.sleep(.1)
    finally:
        if not committed:
            signal_checked(parent, signal.SIGCONT)


def gpu_seconds(records, now=None):
    now = time.monotonic() if now is None else now
    return sum(max(0., record.get('ended', now) - record['started']) for record in records)


def prepare_prior(prior, deadline):
    if (prior / 'frozen_selection.json').exists():
        raise ValueError('prior already entered frozen final evaluation; do not restart selection')
    if (prior / 'watchdog_result.json').exists():
        allocation = read_json(prior / 'allocation.json')
        current = process(allocation['watchdog_pid'])
        if current and 'boardbench.v12.launch' in current['argv']:
            raise ValueError('prior watchdog still exiting; cannot take ownership yet')
        return 'reuse_stopped_run'  # No signal and no failed partial output adopted.
    handoff(prior, deadline)
    return 'completed_child_handoff'


class Accounting:
    def __init__(self, root):
        self.root, self.records, self.lock = Path(root), [], threading.Lock()

    def start(self, name, gpu):
        with self.lock:
            row = {'name': name, 'gpu': gpu, 'started': time.monotonic(), 'started_unix': time.time()}
            self.records.append(row)
            write_json(self.root / 'gpu_intervals.json', self.records)
            return row

    def end(self, row, code):
        with self.lock:
            row.update(ended=time.monotonic(), returncode=code)
            write_json(self.root / 'gpu_intervals.json', self.records)

    def snapshot(self):
        with self.lock:
            return deepcopy(self.records)


class ParallelLedger(Ledger):
    def __init__(self, root, seconds, accounting, prior_gpu, **kwargs):
        self.accounting, self.prior_gpu = accounting, prior_gpu
        super().__init__(root, seconds, **kwargs)

    def update(self, phase, **extra):
        records = self.accounting.snapshot()
        charged = gpu_seconds(records)
        super().update(phase, **{**extra, 'allocated_gpu_seconds': charged,
            'cumulative_charged_gpu_seconds': self.prior_gpu + charged,
            'active_gpu_jobs': [r for r in records if 'ended' not in r],
            'gpu_accounting': 'sum of each assigned job reservation, including Python/CUDA startup; no GPU reserved during CPU-only evaluation'})


class ParallelExperiment(AnimalExperiment):
    def __init__(self, plan, root, repo, prior, gpus, prior_gpu):
        super().__init__(plan, root, repo, 'cuda')
        self.prior = Path(prior)
        self.gpus = gpus
        self.accounting = Accounting(self.root)
        self.ledger = ParallelLedger(self.root, plan['budget_seconds'], self.accounting, prior_gpu,
                                    status_path=self.repo / 'outputs/v12/latest_status.json')
        self.reused = []
        self.teacher = self.prior / 'teachers/dataset.json'

    def configuration(self, route, seed):
        config = self.config('bc' if route == 'bc' else 'ppo', seed)
        config['target_kl'] = .02
        if route == 'bc':
            config['teacher_sha256'] = digest(self.teacher)
        if route == 'bc_ppo':
            config['learning_rate'] = self.plan['bc_ppo_learning_rate']
        return config

    def result(self, route, seed, name, config, base):
        path = base / 'training' / name
        cfg = base / 'configs' / (name + '.json')
        summary = read_json(path / 'summary.json')
        meta = read_json(path / 'final.resume.json')
        weights = path / 'final.resume.weights.pt'
        if read_json(cfg) != config or summary['config'] != config or summary['source_id'] != source_id():
            raise ValueError('completed job configuration/source mismatch')
        if digest(weights) != meta['weights_sha256'] or digest(path / 'final.resume.pt') != meta['resume_sha256']:
            raise ValueError('completed job checkpoint hash mismatch')
        if summary['status'] != 'completed':
            raise ValueError('completed job status mismatch')
        spec = {'id': name, 'route': route, 'method': 'imitation' if route == 'bc' else 'rl',
                'label': f'{route.upper()} · {seed}', 'training_seed': seed,
                'solver': {'id': 'compact', 'params': {**config['network'], 'device': 'cpu'}},
                'weights': str(weights), 'weights_sha256': digest(weights),
                'training_config_sha256': digest(cfg), 'training_summary': summary}
        return spec, path / 'final.resume.pt'

    def train_group(self, jobs):
        """Independent jobs only; every job completes before CPU timed evaluation."""
        results, pending = {}, []
        for job in jobs:
            route, seed, name, seconds, resume, initialize = job
            config = self.configuration(route, seed)
            if (self.prior / 'training' / name / 'summary.json').exists():
                results[name] = self.result(route, seed, name, config, self.prior)
                self.reused.append({'kind': 'training', 'name': name, 'root': str(self.prior)})
            else:
                pending.append(job)
        write_json(self.root / 'reused_results.json', self.reused)
        if not pending:
            return results
        available = queue.Queue()
        for gpu in self.gpus:
            available.put(idle_gpu(gpu))  # Check each stage; never touch an occupied GPU.
        self.ledger.update('parallel_training', queued=[j[2] for j in pending])
        cancel = threading.Event()

        def run(job):
            route, seed, name, seconds, resume, initialize = job
            if cancel.is_set():
                raise RuntimeError('parallel group cancelled')
            gpu = available.get()
            record = self.accounting.start(name, gpu)
            child, code = None, None
            try:
                config = self.configuration(route, seed)
                cfg = self.root / 'configs' / (name + '.json')
                out = self.root / 'training' / name
                write_json(cfg, config)
                command = [sys.executable, '-m', 'boardbench.v12.training', '--config', str(cfg),
                           '--out', str(out), '--seconds', str(seconds), '--device', 'cuda']
                if resume:
                    command += ['--resume', str(resume)]
                if initialize:
                    command += ['--initialize-bc', initialize['weights'], '--initialize-sha256', initialize['weights_sha256']]
                if route == 'bc':
                    command += ['--teacher', str(self.teacher)]
                log = self.root / 'logs' / (name + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
                deadline = min(time.monotonic() + seconds + 90, self.ledger.deadline - self.reserve)
                with log.open('x') as stream:
                    child = subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu},
                                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    while child.poll() is None:
                        if cancel.is_set() or time.monotonic() >= deadline:
                            kill_tree(child.pid)
                            raise TimeoutError(f'parallel job stopped: {name}')
                        time.sleep(.1)
                    code = child.returncode
                if code:
                    raise RuntimeError(f'{name} failed; inspect {log}')
                return name, self.result(route, seed, name, config, self.root)
            except BaseException:
                cancel.set()
                raise
            finally:
                if child is not None and child.poll() is None:
                    kill_tree(child.pid); child.wait(timeout=2)
                self.accounting.end(record, child.returncode if child else code)
                available.put(gpu)

        with ThreadPoolExecutor(max_workers=len(self.gpus)) as pool:
            futures = [pool.submit(run, job) for job in pending]
            try:
                while not all(future.done() for future in futures):
                    self.ledger.update('parallel_training', queued=[j[2] for j, f in zip(pending, futures) if not f.done()])
                    time.sleep(2)
                for future in futures:
                    name, value = future.result()
                    results[name] = value
            except BaseException:
                cancel.set()
                raise
        self.ledger.update('parallel_training_finished')
        return results

    def evaluate(self, spec, stage, which, seconds, task='harmonies', final=False):
        if any('ended' not in r for r in self.accounting.snapshot()):
            raise RuntimeError('timed evaluation may not compete with training')
        name = f'{stage}_{task}_{spec["id"]}'
        old = self.prior / 'evaluations' / name
        if stage == 'dev' and (old / 'summary.json').exists():
            cfg = read_json(old / 'config.json')
            if cfg['seeds'] != which or cfg['seconds'] != seconds or cfg['policy']['solver'] != spec['solver'] or cfg['policy'].get('weights_sha256') != spec.get('weights_sha256'):
                raise ValueError('reused development evaluation mismatch')
            self.reused.append({'kind': 'evaluation', 'name': name, 'root': str(old),
                                'episodes_sha256': digest(old / 'episodes.jsonl')})
            write_json(self.root / 'reused_results.json', self.reused)
            return {'summary': read_json(old / 'summary.json'), 'rows': rows(old), 'path': str(old)}
        return super().evaluate(spec, stage, which, seconds, task, final)

    def execute(self):
        budget = read_json(self.prior / 'online_budget.json')['primary_seconds']
        write_json(self.root / 'online_budget.json', {'primary_seconds': budget, 'frozen_from': str(self.prior)})
        diagnosis = read_json(self.prior / 'diagnosis/report.json')
        if diagnosis['levels']['placement'].get('bc', {}).get('status') != 'fit_passed':
            raise ValueError('original animal diagnosis did not pass')
        write_json(self.root / 'diagnosis_reuse.json', {'root': str(self.prior),
                   'report_sha256': digest(self.prior / 'diagnosis/report.json'), 'report': diagnosis})
        legacy_rl = max((p for p in self.legacy['harmonies'] if p['method'] == 'rl'), key=lambda p: p.get('validation_mean') or -1)
        baselines = [p for p in self.legacy['harmonies'] if p['method'] in ('heuristic', 'search')] + [legacy_rl]
        baselines += [{'id': 'time_aware_search', 'method': 'search', 'label': '限时搜索',
                       'solver': {'id': 'timed_search', 'params': {'max_simulation_steps': 384}}}]
        incumbent_rows = {p['id']: rows(self.previous / 'evaluations' / f'validation_harmonies_{p["id"]}') for p in baselines}
        seeds = self.plan['training_seeds']
        jobs = [('bc', seed, f'bc_{seed}_frozen', self.plan['bc_seconds'], None, None) for seed in seeds]
        trained = self.train_group(jobs)
        frozen_bc = {seed: trained[f'bc_{seed}_frozen'][0] for seed in seeds}
        for seed, spec in frozen_bc.items():
            report = self.evaluate(spec, 'dev', self.dev, budget)
            self.curve.append({'route': 'bc', 'seed': seed, 'spec': spec, 'summary': report['summary']})
            control = deepcopy(spec)
            weights = Path(spec['weights']).parent / 'initial.resume.weights.pt'
            control.update(id=f'untrained_{seed}', method='untrained_control', label=f'未训练 · {seed}',
                           weights=str(weights), weights_sha256=digest(weights), training_summary=None)
            report = self.evaluate(control, 'dev', self.dev, budget)
            self.curve.append({'route': 'untrained', 'seed': seed, 'spec': control, 'summary': report['summary']})
        candidates, resumes, best = {'bc': list(frozen_bc.values())}, {}, {}
        previous = 0
        for stage, cumulative in enumerate(self.plan['rl_cumulative_seconds']):
            delta = cumulative - previous
            routes = ('ppo', 'bc_ppo') if stage == 0 or diagnosis['levels']['placement']['ppo']['status'] == 'fit_passed' else ('bc_ppo',)
            needed = math.ceil(len(routes)*3/len(self.gpus))*(delta + 90) + len(routes)*3*60
            gpu_needed = len(routes)*3*(delta + 90)
            remaining_gpu = 14400 - self.ledger.prior_gpu - gpu_seconds(self.accounting.snapshot())
            if self.ledger.remaining() < self.reserve + needed or remaining_gpu < gpu_needed:
                self.curve.append({'stage': stage, 'stop': 'parallel wall/GPU budget reserve'})
                break
            jobs = [(route, seed, f'{route}_{seed}_stage{stage}', delta, resumes.get((route, seed)),
                     frozen_bc[seed] if route == 'bc_ppo' and (route, seed) not in resumes else None)
                    for route in routes for seed in seeds]
            trained = self.train_group(jobs)
            for route, seed, name, *_ in jobs:
                spec, checkpoint = trained[name]
                resumes[route, seed] = checkpoint
                report = self.evaluate(spec, 'dev', self.dev, budget)
                if (route, seed) not in best or report['summary']['score_mean'] > best[route, seed]['summary']['score_mean']:
                    best[route, seed] = {'spec': spec, 'summary': report['summary']}
                self.curve.append({'route': route, 'seed': seed, 'stage': stage, 'spec': spec,
                                   'cumulative_ppo_seconds': cumulative, 'summary': report['summary']})
                write_json(self.root / 'learning_curve.json', self.curve)
            for route in routes:
                candidates[route] = [best[route, seed]['spec'] for seed in seeds]
            previous = cumulative
        write_json(self.root / 'learning_curve.json', self.curve)
        validation, grouped, gates, nominated = {}, {}, {}, {}
        for route, specs in candidates.items():
            grouped[route] = []
            for spec in specs:
                report = self.evaluate(spec, 'validation', self.validation, budget, final=True)
                spec['summary'] = report['summary']; validation[spec['id']] = report
                grouped[route].append(report['rows'])
            gates[route] = animal_gate(grouped[route], incumbent_rows, incumbent_rows[legacy_rl['id']],
                                      grouped.get('bc') if route == 'bc_ppo' else None)
            selected = max(specs, key=lambda p: p['summary']['score_mean'])
            bc_rows = validation[frozen_bc[selected['training_seed']]['id']]['rows'] if route == 'bc_ppo' else None
            single = nominated_gate(validation[selected['id']]['rows'], incumbent_rows, incumbent_rows[legacy_rl['id']], bc_rows)
            gates[route]['nominated_artifact_gate'] = single
            gates[route]['passed'] = gates[route]['passed'] and single['passed']
            nominated[route] = selected
        winners = [route for route in candidates if gates[route]['passed']]
        winner = max(winners, key=lambda route: nominated[route]['summary']['score_mean']) if winners else None
        selected = [p for group in candidates.values() for p in group] + [legacy_rl, baselines[-1]]
        write_json(self.root / 'frozen_selection.json', {'candidates': selected, 'gates': gates,
                   'promoted_family': winner, 'test_used_for_selection': False, 'test_seeds': self.test,
                   'primary_seconds': budget, 'source_id': source_id()})
        tests = {}
        for spec in selected:
            tests[spec['id']] = self.evaluate(spec, 'test', self.test, budget, final=True)['summary']
            write_json(self.root / 'test_results.json', tests)
        valid = all(not report['failures'] for report in tests.values())
        release = None
        if winner and valid:
            for task in ('micro_tiles', 'take_it_easy'):
                spec = max(self.legacy[task], key=lambda p: p.get('validation_mean') or -1)
                valid = not self.evaluate(spec, 'compatibility', self.test[:8], budget, task=task, final=True)['summary']['failures'] and valid
            if valid:
                policies = deepcopy(self.legacy); policies['harmonies'] += candidates[winner]
                release = self.publish(policies, {**self.defaults, 'harmonies': nominated[winner]['id']},
                                       'promoted_release', 'promotion', gates[winner])
        self.finish(status='completed', promoted_family=winner if release else None, policy_unchanged=release is None,
                    active_release=release, gates=gates, test_execution_valid=valid,
                    allocated_gpu_seconds=gpu_seconds(self.accounting.snapshot()),
                    prior_charged_gpu_seconds=self.ledger.prior_gpu, reused_results=self.reused)


def launch(args):
    repo, prior, root = (Path(value).resolve() for value in (args.repo, args.prior, args.root))
    if root.exists() or not root.is_relative_to(repo / 'outputs/v12') or not prior.is_relative_to(repo / 'outputs/v12'):
        raise ValueError('new project run directory and existing project prior run required')
    if not math.isfinite(args.extra_gpu_charge) or args.extra_gpu_charge < 0:
        raise ValueError('invalid extra diagnostic GPU charge')
    allocation = read_json(prior / 'allocation.json')
    if source_id(prior / 'source') != allocation['source_id']:
        raise ValueError('frozen source changed')
    if os.getloadavg()[0] > len(os.sched_getaffinity(0)) * .5:
        raise RuntimeError('host CPU load too high for parallel jobs')
    os.sched_setaffinity(0, allocation['cpu_affinity'])
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        raise ValueError('distinct requested GPU indices required')
    for gpu in args.gpus:
        if gpu != allocation['gpu_index']:
            idle_gpu(gpu)
    deadline = allocation['deadline_unix']
    if deadline - time.time() < 2400:
        raise ValueError('too little original window to change scheduling')
    root.mkdir(parents=True)
    shutil.copytree(prior / 'source', root / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, root / 'parallel_v12.py')
    takeover = prepare_prior(prior, deadline)
    started = time.monotonic()
    prior_result = read_json(prior / 'watchdog_result.json')
    prior_gpu = prior_result['cumulative_charged_seconds'] + args.extra_gpu_charge
    lock = (root.parent / 'experiment.lock').open('a')
    for attempt in range(100):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
        except BlockingIOError:
            if attempt == 99:
                raise
            time.sleep(.1)
    gpus = {index: idle_gpu(index) for index in args.gpus}
    plan = read_json(prior / 'plan.json'); plan['budget_seconds'] = deadline - time.time()
    write_json(root / 'plan.json', plan)
    write_json(root / 'allocation.json', {'started_unix': time.time(), 'deadline_unix': deadline,
               'cpu_affinity': sorted(os.sched_getaffinity(0)), 'gpu_uuids': gpus, 'prior_charged_gpu_seconds': prior_gpu,
               'gpu_seconds_cap': 14400, 'source_id': allocation['source_id'], 'prior_root': str(prior),
               'takeover': takeover, 'extra_diagnostic_gpu_seconds': args.extra_gpu_charge,
               'orchestrator_sha256': digest(root / 'parallel_v12.py'), 'watchdog_pid': os.getpid(),
               'accounting': 'GPU job reservations summed across devices; shared8CPUcores throughout; unchanged original deadline'})
    env = {**os.environ, 'PYTHONPATH': str(root / 'source/src'), 'CUDA_VISIBLE_DEVICES': '',
           'BOARDBENCH_STARTED_MONOTONIC': str(started), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1'}
    command = [sys.executable, str(root / 'parallel_v12.py'), 'controller', '--repo', str(repo),
               '--root', str(root), '--prior', str(prior), '--gpus', *map(str, args.gpus)]
    child = subprocess.Popen(command, env=env, start_new_session=True)
    interrupted = [False]
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    reason = None
    try:
        while child.poll() is None:
            path = root / 'gpu_intervals.json'
            records = read_json(path) if path.exists() else []
            if interrupted[0] or time.time() >= deadline - 2 or prior_gpu + gpu_seconds(records) >= 14398:
                reason = 'interrupted' if interrupted[0] else 'wall_or_aggregate_GPU_budget_exhausted'
                kill_tree(child.pid); break
            time.sleep(.25)
        child.wait(timeout=2)
    finally:
        if child.poll() is None:
            kill_tree(child.pid)
        records = read_json(root / 'gpu_intervals.json') if (root / 'gpu_intervals.json').exists() else []
        closed = time.monotonic()
        for record in records:
            if 'ended' not in record:
                record.update(ended=closed, returncode=child.poll(), watchdog_closed=True)
        write_json(root / 'gpu_intervals.json', records)
        used = gpu_seconds(records)
        result = {'returncode': child.poll(), 'stop_reason': reason, 'elapsed_seconds': time.monotonic()-started,
                  'allocated_gpu_seconds': used, 'cumulative_charged_gpu_seconds': prior_gpu+used,
                  'allocated_cpu_core_seconds': 8*(time.monotonic()-started), 'original_deadline_unix': deadline}
        write_json(root / 'watchdog_result.json', result)
        if child.returncode or reason:
            status = read_json(root / 'status.json') if (root / 'status.json').exists() else {}
            status.update(phase='stopped', **result, partial_results_retained=True)
            write_json(root / 'status.json', status); write_json(repo / 'outputs/v12/latest_status.json', status)
    if child.returncode:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'controller'))
    parser.add_argument('--repo', required=True); parser.add_argument('--root', required=True)
    parser.add_argument('--prior', required=True); parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--extra-gpu-charge', type=float, default=0,
                        help='additional GPU seconds from intervening diagnostics, including startup')
    args = parser.parse_args()
    if args.mode == 'launch':
        launch(args)
    else:
        root = Path(args.root); allocation = read_json(root / 'allocation.json')
        if source_id() != allocation['source_id'] or digest(__file__) != allocation['orchestrator_sha256']:
            raise ValueError('controller source/orchestration identity mismatch')
        experiment = ParallelExperiment(read_json(root / 'plan.json'), root, args.repo, args.prior,
                                        args.gpus, allocation['prior_charged_gpu_seconds'])
        try:
            experiment.execute()
        except BaseException as exc:
            experiment.ledger.update('stopped', error=f'{type(exc).__name__}: {exc}', partial_results_retained=True)
            raise


if __name__ == '__main__':
    main()
