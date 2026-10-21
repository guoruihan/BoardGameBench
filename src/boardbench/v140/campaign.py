"""Four independent bounded single-GPU trials; frozen source, node lock, receipts."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from boardbench.artifacts.store import read_json, write_json, source_files, source_id
from boardbench.v12.budget import kill_tree
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.preflight import reserve
from .preflight import trial_config


def report(root, jobs, started, deadline, phase):
    rows = []
    now = time.time()
    for job in jobs:
        directory = root/'trials'/job['name']
        curve = directory/'curve.jsonl'
        latest = {}
        if curve.exists():
            import json
            lines = curve.read_text().splitlines()
            if lines:
                latest = json.loads(lines[-1])
        done = read_json(directory/'done.json') if (directory/'done.json').exists() else None
        rows.append(dict(trial=job['name'], pid=job.get('pid'), gpu=job['gpu'], cpus=job['cpus'],
                         actions=latest.get('actions', 0), batches=latest.get('batches', 0),
                         training_score=latest.get('score_mean'), development=latest.get('development'),
                         checkpoint=(str(directory/'latest.pt') if (directory/'latest.pt').exists() else None),
                         status='completed' if done else 'failed' if (directory/'failed.json').exists() else 'running',
                         confirmation=done.get('confirmation') if done else None))
    write_json(root/'status.json', dict(phase=phase, updated_unix=now, started_unix=started,
                                       deadline_unix=deadline, elapsed_seconds=now-started, jobs=rows))


def execute(repo, root):
    launch = read_json(root/'launch.json'); config = read_json(root/'config.json')
    if os.uname().nodename != config['host'] or source_id() != launch['source_id']:
        raise ValueError('authorized host and frozen source required')
    lock = reserve(repo, config)
    cpus = launch['cpus']; os.sched_setaffinity(0, {cpus[-1]})
    started = time.time(); deadline = min(started+config['wall_seconds'], config.get('deadline_unix', float('inf')))
    env = {**os.environ, 'PYTHONPATH': str(root/'source/src'), 'PYTHONNOUSERSITE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'NUMEXPR_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    active, receipts, jobs = [], [], []
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))

    def start(command, cores, gpu, log_path, label):
        stream = log_path.open('a')
        child = subprocess.Popen(['taskset', '-c', ','.join(map(str, cores)), sys.executable, '-u', *command],
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**env, 'CUDA_VISIBLE_DEVICES': gpu})
        receipt = dict(stage=label, pid=child.pid, gpu=gpu, cpus=cores, started_unix=time.time())
        receipts.append(receipt); active.append((child, stream, receipt))
        write_json(root/'cost.json', dict(attempts=receipts))
        return child

    def wait(until):
        while any(p.poll() is None for p, _, _ in active):
            for p, _, r in active:
                if p.poll() is not None and 'elapsed_seconds' not in r:
                    r.update(returncode=p.returncode, elapsed_seconds=time.time()-r['started_unix'])
            report(root, jobs, started, deadline, 'training' if jobs else 'cuda_preflight')
            write_json(root/'cost.json', dict(attempts=receipts))
            if stopping[0] or time.time() >= until:
                for p, _, _ in active:
                    if p.poll() is None: kill_tree(p.pid)
                break
            time.sleep(5)
        failed = False
        for p, stream, r in active:
            p.wait(timeout=20); stream.close()
            if 'elapsed_seconds' not in r: r.update(returncode=p.returncode, elapsed_seconds=time.time()-r['started_unix'])
            failed |= p.returncode != 0
        active.clear()
        return failed

    try:
        for i, gpu in enumerate(config['gpu_uuids']):
            start(['-m', 'boardbench.v140.preflight', '--repo', str(root/'source'),
                   '--config', str(root/'resolved_config.json'), '--out', str(root/f'preflight_gpu{i}')],
                  cpus[4*i:4*i+4], gpu, root/'logs'/f'preflight_gpu{i}.log', 'preflight')
        if wait(min(deadline, time.time()+900)):
            raise RuntimeError('actual CUDA preflight failed or timed out; no formal training launched')
        for i in range(4):
            gate = read_json(root/f'preflight_gpu{i}/result.json')
            if not gate['passed'] or gate['source_id'] != launch['source_id']:
                raise ValueError('CUDA gate source mismatch')
        if config.get('continuation_run'):
            previous = Path(config['continuation_run'])
            # Benchmark on idle authorized hardware, at the actual training
            # batch/workers, before any replacement learner is started.
            start(['-m', 'boardbench.v141.verify', '--config', str(previous/'trials/score_a/config.json'),
                   '--checkpoint', str(previous/'trials/score_a/latest.pt'), '--out', str(root/'speed_gate')],
                  cpus[:14], config['gpu_uuids'][0], root/'logs/speed_gate.log', 'paired_speed_gate')
            if wait(min(deadline, time.time()+900)):
                raise RuntimeError('paired speed/parity gate failed; no replacement training started')
            gate = read_json(root/'speed_gate/result.json')
            if not gate['passed'] or gate['source_id'] != launch['source_id']:
                raise ValueError('speed gate source mismatch')
        for i, (objective, replica) in enumerate((('score', 0), ('score', 1), ('margin', 0), ('margin', 1))):
            name = objective+'_'+'ab'[replica]
            module = 'boardbench.v140.training'
            if config.get('continuation_run'):
                from boardbench.v141.training import continuation_config
                c = continuation_config(Path(config['continuation_run'])/'trials'/name, launch['source_id'])
                module = 'boardbench.v141.training'
                if c['deadline_unix'] > deadline or c['deadline_unix'] <= time.time():
                    raise ValueError('original trial deadline invalid')
            else:
                c = trial_config(root/'source', read_json(root/'resolved_config.json'), objective, replica)
                c['max_seconds'] = max(0., deadline-time.time()-30)
                c['deadline_unix'] = deadline-15
            path = root/'configs'/(name+'.json'); write_json(path, c)
            cores = cpus[14*i:14*i+14]; gpu = config['gpu_uuids'][i]
            p = start(['-m', module, '--config', str(path), '--out', str(root/'trials'/name)],
                      cores, gpu, root/'logs'/(name+'.log'), name)
            jobs.append(dict(name=name, gpu=gpu, cpus=cores, pid=p.pid))
        write_json(root/'jobs.json', jobs)
        failed = wait(deadline)
        report(root, jobs, started, deadline, 'interrupted' if stopping[0] else 'failed' if failed else 'completed')
    except BaseException as exc:
        write_json(root/'failed.json', dict(error=repr(exc), time=time.time()))
        raise
    finally:
        for child, stream, r in active:
            if child.poll() is None: kill_tree(child.pid)
            child.wait(timeout=20); stream.close()
            r.update(returncode=child.returncode, elapsed_seconds=time.time()-r['started_unix'])
        elapsed = time.time()-started
        write_json(root/'cost.json', dict(attempts=receipts, wall_seconds=elapsed,
                   allocated_cpu_core_seconds=sum(len(r['cpus'])*r['elapsed_seconds'] for r in receipts)+elapsed,
                   physical_gpu_seconds=sum(r['elapsed_seconds'] for r in receipts),
                   includes='preflight, failed attempts, training, evaluation, checkpoint and cleanup',
                   previous_campaign=config.get('continuation_run'),
                   deadline_unix=deadline))
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True); p.add_argument('--out', required=True)
    p.add_argument('--config'); p.add_argument('--execute', action='store_true')
    p.add_argument('--continue-from', help='gracefully stopped V140 run; preserves its absolute deadline')
    args = p.parse_args(); repo, root = Path(args.repo).resolve(), Path(args.out).resolve()
    if args.execute: return execute(repo, root)
    config = read_json(args.config)
    if args.continue_from:
        previous = Path(args.continue_from).resolve()
        status = read_json(previous/'status.json'); old = read_json(previous/'config.json')
        if status['phase'] not in ('completed', 'interrupted') or not (previous/'cost.json').exists():
            raise ValueError('parent controller must have exited before replacement')
        for key in ('host', 'gpu_uuids', 'workers', 'batch_matches', 'training', 'parent_sha256',
                    'development_environments', 'confirmation_environments'):
            if config[key] != old[key]: raise ValueError('continuation changes campaign configuration: '+key)
        deadline = status['deadline_unix']
        if deadline-time.time() < config['training']['finish_reserve_seconds']+600:
            raise ValueError('insufficient original budget for verified continuation')
        config.update(continuation_run=str(previous), deadline_unix=deadline, wall_seconds=deadline-time.time())
    if os.uname().nodename != config['host']: raise ValueError('authorized a226 host only')
    physical = physical_cpus()
    if len(physical)//2 < 57 or len(config['gpu_uuids']) != 4 or config['workers'] != 13:
        raise ValueError('four single-GPU jobs, 57 cores total, <= half physical cores required')
    root.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'configs', 'trials'): (root/name).mkdir()
    for relative in source_files(repo):
        dest = root/'source'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/relative, dest)
    write_json(root/'config.json', config)
    resolved = {**config, 'parent': str(repo/config['parent']), 'native_manifest': str(repo/config['native_manifest'])}
    write_json(root/'resolved_config.json', resolved)
    write_json(root/'launch.json', dict(source_id=source_id(repo), cpus=physical[:57],
                                       source_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()))
    os.sched_setaffinity(0, {physical[56]})
    result = subprocess.run([sys.executable, '-u', '-m', 'boardbench.v140.campaign',
                             '--repo', str(repo), '--out', str(root), '--execute'],
                            env={**os.environ, 'PYTHONPATH': str(root/'source/src')})
    if result.returncode: raise SystemExit(result.returncode)


if __name__ == '__main__': main()
