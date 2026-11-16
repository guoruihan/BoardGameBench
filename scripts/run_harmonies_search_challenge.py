#!/usr/bin/env python3
"""One preregistered eight-game challenge of four frozen budget-study submissions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from boardbench.families.evaluate import archive_source
from boardbench.families.joint_pool import verify_pool
from boardbench.families.pilot_run import _write_json, tree_hash, validate_submission
from boardbench.families.pilot_supervisor import (
    physical_cpu_ids, minimal_runtime, SandboxSupervisor, SandboxSpec, Limits,
)
from report_harmonies_pilot import summarize


def frozen_source(experiment):
    experiment = Path(experiment)
    state = json.loads((experiment/'state.json').read_text())
    frozen = json.loads((experiment/'checkpoints/frozen.json').read_text())
    sha = frozen['sha256']
    checkpoint = experiment/'checkpoints'/sha
    if state.get('status') != 'complete' or Path(state['frozen_submission']).resolve() != checkpoint.resolve():
        raise ValueError('challenge requires the final frozen submission from completed development')
    if tree_hash(checkpoint) != sha or validate_submission(checkpoint)['entry_mode'] != 'program':
        raise ValueError('challenge expects unchanged program submissions')
    return dict(source_experiment=str(experiment.resolve()), sha256=sha)


def report(root, plan):
    rows = []
    for job in plan['jobs']:
        result_path = root/job['label']/'experiment/results.jsonl'
        completion = root/job['label']/'worker_completion.json'
        if not completion.exists() or not result_path.exists():
            rows.append(dict(label=job['label'], status='pending'))
            continue
        records = [json.loads(line) for line in result_path.read_text().splitlines()]
        basic, _ = summarize(result_path.parent)
        for item, record in zip(basic, records):
            outcome = (record['result'].get('gt', {}).get('outcomes') or [{}])[-1]
            score = outcome.get('score_breakdown')
            item.update(label=job['label'], opponent_search=outcome.get('opponent_search'),
                        own_animals=score[item['seat']].get('animals') if score else None,
                        opponent_animals=score[1-item['seat']].get('animals') if score else None,
                        score_breakdown=score)
            rows.append(item)
    _write_json(root/'summary.json', dict(rows=rows, interpretation=
        'One fixed paired calibration seed per frozen submission; exploratory, not a stable ranking.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-ids', help='24 idle logical CPU IDs, one per physical core; comma separated')
    args = parser.parse_args()
    study = args.study.resolve(); root = args.output.resolve()
    if not (study/'finished.json').exists():
        raise ValueError('finish the original study before changing evaluator code or adding opponents')
    previous = json.loads((study/'preregistered_plan.json').read_text())
    template = json.loads((study/'private/eval-b100-r0.json').read_text())
    if template['deadline'] <= time.time()+1200:
        raise ValueError('insufficient original wall budget for the challenge')
    physical_cpu_ids(48)  # Allocate no more than half the physical host cores.
    cores = tuple(sorted(map(int,args.cpu_ids.split(',')))) if args.cpu_ids else physical_cpu_ids(24)
    topology = lambda c: Path(f'/sys/devices/system/cpu/cpu{c}/topology')
    if len(cores)!=24 or len(set(cores))!=24 or not set(cores)<=os.sched_getaffinity(0):
        raise ValueError('exactly 24 available CPU IDs are required')
    if len({((topology(c)/'physical_package_id').read_text(),(topology(c)/'core_id').read_text()) for c in cores})!=24:
        raise ValueError('CPU IDs must refer to distinct physical cores')
    runtime = minimal_runtime(include_codex=True, include_compiler=True)
    with tempfile.TemporaryDirectory(prefix='search-preflight-') as directory:
        probe = Path(directory); (probe/'work').mkdir(); (probe/'sdk').mkdir()
        checked = SandboxSupervisor().run(SandboxSpec(
            argv=['/usr/bin/python3','-I','-c','print("sandbox_ready")'],
            work_dir=probe/'work',sdk_dir=probe/'sdk',ledger_path=probe/'ledger.json',
            cpu_ids=cores[:1],limits=Limits(5,15,256*1024**2),runtime_mounts=runtime))
        if checked.exit_code != 0 or not checked.cleanup_verified:
            raise RuntimeError('sandbox runtime preflight failed before any GT game: '+checked.reason)
    pool = verify_pool(template['pool'])
    opponent = pool['games']['harmonies']['members']['search_900']
    if opponent['spec']['kind'] != 'terminal_search' or opponent['spec']['seconds'] != 900:
        raise ValueError('expected the frozen 900-second search opponent')
    jobs = []
    for level in ('b025', 'b050', 'b100', 'b200'):
        source = Path(previous['continuation_of']) if level == 'b025' else study
        saved = frozen_source(source/f'dev-{level}-r0/experiment')
        for seat in (0, 1):
            index = len(jobs)
            jobs.append(dict(label=f'{level}-r0-p{seat}', level=level, seat=seat,
                             cpu_ids=list(cores[index*3:index*3+3]), **saved))
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    private = root/'private'; private.mkdir(mode=0o700)
    plan = dict(protocol='frozen_search_challenge_v1', started=time.time(),
        deadline=template['deadline'], source_study=str(study), opponent='search_900',
        opponent_policy_id=opponent['policy_id'], pool_sha256=pool['manifest_sha256'],
        seed_commitment=hashlib.sha256(template['seed_key'].encode()).hexdigest(),
        matches=8, jobs=jobs, deployment='L20_C120_G0', opponent_seconds=900,
        scheduling='8 one-match workers, each with 2 agent cores and 1 separate search core',
        host=os.uname().nodename, sandbox_preflight=dict(exit_code=checked.exit_code,
            cleanup_verified=checked.cleanup_verified,cpu_seconds=checked.cpu_seconds),
        selection='last frozen submission, no redevelopment or hidden score selection',
        accounting='search CPU/decision seconds separate from agent compute; failed games not retried')
    _write_json(root/'preregistered_plan.json', plan)
    _write_json(root/'source_identity.json', archive_source(root))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    script = Path(__file__).with_name('run_harmonies_resource_study.py')
    processes = []
    try:
        for job in jobs:
            spec = dict(template, root=str(root/job['label']), label=job['label'],
                protocol=plan['protocol'], kind='evaluation', evaluation_pairs=1,
                calibration_seats=[job['seat']], test_concurrency=1, opponent='search_900',
                opponent_cpu=job['cpu_ids'][2], source_experiment=job['source_experiment'], sha256=job['sha256'],
                api_clock=str(private/'api_clock'))
            path = private/(job['label']+'.json'); _write_json(path, spec); path.chmod(0o600)
            log = (root/(job['label']+'.log')).open('w')
            proc = subprocess.Popen(['/usr/bin/taskset', '-c', ','.join(map(str, job['cpu_ids'])),
                sys.executable, str(script), '--worker', str(path)], stdout=log, stderr=subprocess.STDOUT, env=env)
            processes.append((job['label'], proc, log))
            print(json.dumps(dict(event='started', label=job['label'], pid=proc.pid, time=time.time())), flush=True)
        while any(proc.poll() is None for _, proc, _ in processes):
            report(root, plan)
            if time.time() >= plan['deadline']:
                for _, proc, _ in processes:
                    if proc.poll() is None: proc.terminate()
                break
            time.sleep(10)
        exits = {}
        for label, proc, _ in processes:
            exits[label] = proc.wait(timeout=60)
        report(root, plan)
        _write_json(root/'finished.json', dict(ended=time.time(), exit_codes=exits,
                    status='ended', deadline_reached=time.time() >= plan['deadline']))
    finally:
        for _, proc, log in processes:
            if proc.poll() is None: proc.terminate()
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            log.close()


if __name__ == '__main__':
    main()
