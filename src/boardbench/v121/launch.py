"""Continuous, resource-isolated factorial scheduler; no implicit final-test selection.

Each child has a finite phase watchdog and writes a full continuation checkpoint.
The user-authorized campaign itself has no GPU-hour limit. STOP_REQUESTED.json
requests a graceful boundary stop; failures stop the campaign with evidence.
"""
import argparse
from collections import deque
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from boardbench.artifacts.store import digest, read_json, source_files, source_id, write_json
from boardbench.v12.budget import kill_tree
from boardbench.v12.launch import idle_gpu, physical_cores
from .registry import IMPLEMENTED, THROUGHPUT, NUMERIC, prepare

MILESTONES = (4096, 400000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000)


def next_milestone(current):
    return next((x for x in MILESTONES if x > current), current * 2)


def reservation_seconds(records, now=None):
    now = time.monotonic() if now is None else now
    return sum(max(0., row.get('ended', now) - row['started']) for row in records)


def ready_for_next_target(pending, active, eval_pending, eval_active, allocation):
    if pending or active:
        return False
    backlog = len(eval_pending) + int(eval_active is not None)
    if allocation.get('pipeline_evaluations', False):
        return backlog < allocation.get('evaluation_high_watermark', 48)
    return backlog == 0


def validate_child(out):
    out = Path(out)
    summary = read_json(out / 'summary.json')
    meta = read_json(out / 'final.resume.json')
    if digest(out / 'final.resume.pt') != meta['resume_sha256'] or digest(out / 'final.resume.weights.pt') != meta['weights_sha256']:
        raise ValueError('child checkpoint hash mismatch')
    if summary['source_id'] != source_id() or meta['source_id'] != source_id():
        raise ValueError('child source mismatch')
    if summary['real_atomic_actions'] != meta['real_atomic_actions']:
        raise ValueError('child accounting mismatch')
    return summary


def controller(root):
    root = Path(root).resolve()
    allocation = read_json(root / 'allocation.json')
    if source_id() != allocation['source_id']:
        raise ValueError('controller frozen source mismatch')
    gpus = allocation['gpus']
    cpu_groups = allocation['training_cpu_groups']
    eval_cores = allocation['evaluation_cpu_cores']
    phase_seconds = allocation['phase_seconds']
    state_path = root / 'controller_state.json'
    if state_path.exists():
        state = read_json(state_path)
        # A previous failure remains a failure; explicit new launch/resume is required.
        raise ValueError('controller state exists; automatic failure retry is disabled')
    target, chunk = MILESTONES[0], 0
    latest, curve, records, eval_records = {}, [], [], []
    recovery = None
    if allocation.get('recovery_sha256'):
        if digest(root / 'recovery.json') != allocation['recovery_sha256']:
            raise ValueError('numeric recovery manifest hash mismatch')
        recovery = read_json(root / 'recovery.json')
        latest = deepcopy(recovery['entries'])
        target = recovery['parent_target_actions']
        if min(x['actions'] for x in latest.values()) >= target:
            target = next_milestone(max(x['actions'] for x in latest.values()))
    active, eval_active = {}, None
    pending = deque()
    eval_pending = deque()
    for baseline in sorted((root / 'baselines').glob('*.policy.json')):
        spec=read_json(baseline)['policy']
        eval_pending.append({'experiment':spec['id'],'seed':spec.get('training_seed'),
            'name':spec['id'],'out':str(root/'baselines'),'target':0,'actions':0,
            'split':'development','policy_file':str(baseline),'control':'frozen_baseline'})
    started = time.monotonic()
    stop_path = root / 'STOP_REQUESTED.json'
    phase = 'initializing'
    stop_sent = None
    last_error = None

    def request_stop(*_):
        write_json(stop_path, {'requested_unix': time.time(), 'reason': 'signal'})
    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}

    def queue_target():
        for seed in allocation.get('training_seeds', (911, 912, 913)):
            for experiment in allocation.get('experiments', IMPLEMENTED):
                pending.append({'experiment': experiment, 'seed': seed, 'target': target})

    def status(error=None):
        result = {'research_version': '1.2.1', 'phase': phase, 'updated_unix': time.time(),
                  'pid': os.getpid(), 'source_id': allocation['source_id'], 'current_target_actions': target,
                  'gpu_hours_cap': None, 'gpu_seconds': reservation_seconds(records),
                  'cpu_core_seconds': 8 * (reservation_seconds(records) + reservation_seconds(eval_records)),
                  'elapsed_seconds': time.monotonic()-started,
                  'recovery': str(root / 'recovery.json') if recovery else None,
                  'prior_gpu_seconds': recovery['prior_gpu_seconds'] if recovery else 0.,
                  'cumulative_gpu_seconds': reservation_seconds(records) + (recovery['prior_gpu_seconds'] if recovery else 0.),
                  'cumulative_cpu_core_seconds': 8 * (reservation_seconds(records) + reservation_seconds(eval_records)) +
                    (recovery['prior_cpu_core_seconds'] if recovery else 0.),
                  'active': [{k:v for k,v in job.items() if k not in ('process','log')} for job in active.values()],
                  'pending_jobs': list(pending), 'evaluation_pending': len(eval_pending),
                  'evaluation_active': eval_active['name'] if eval_active else None,
                  'pipeline_evaluations': allocation.get('pipeline_evaluations', False),
                  'evaluation_backpressure': bool(allocation.get('pipeline_evaluations', False) and
                    len(eval_pending) + int(eval_active is not None) >= allocation.get('evaluation_high_watermark', 48)),
                  'latest_checkpoints': latest, 'curve_points': len(curve),
                  'experiment_status': {x: ('running' if any(k.startswith(x+'_') for k in latest) or
                    any(j['experiment']==x for j in active.values()) else 'planned') for x in allocation.get('experiments', IMPLEMENTED)},
                  'final_test_executed': False, 'deployment_changed': False, 'error': error or last_error}
        write_json(root / 'status.json', result)
        write_json(root.parent / 'latest_status.json', result)
        write_json(state_path, {'latest': latest, 'target': target, 'chunk': chunk,
                               'pending': list(pending), 'source_id': allocation['source_id']})
        write_json(root / 'gpu_intervals.json', records)
        write_json(root / 'cpu_evaluation_intervals.json', eval_records)

    def spawn(command, cores, gpu, name):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1',
               'PYTHONPATH': str(root / 'source/src'), 'PYTHONUNBUFFERED': '1'}
        log = (root / 'logs' / (name + '.log')).open('x')
        command = ['taskset', '-c', ','.join(map(str, cores)), sys.executable, *map(str, command)]
        try:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        return child, log

    queue_target()
    try:
        tick = 0.
        while True:
            stopping = stop_path.exists()
            if stopping and stop_sent is None:
                stop_sent = time.monotonic()
                for job in active.values():
                    if job['process'].poll() is None:
                        job['process'].send_signal(signal.SIGTERM)
                phase = 'stopping_at_batch_boundary'
            for slot, job in list(active.items()):
                child = job['process']
                if child.poll() is None:
                    if time.monotonic() - job['record']['started'] > phase_seconds + 180 or (stop_sent and time.monotonic()-stop_sent > 120):
                        kill_tree(child.pid)
                        raise TimeoutError(f'training phase watchdog: {job["name"]}')
                    continue
                job['log'].close()
                job['record'].update(ended=time.monotonic(), returncode=child.returncode)
                del active[slot]
                if child.returncode:
                    raise RuntimeError(f'{job["name"]} failed; inspect logs; no retry or gate relaxation')
                summary = validate_child(job['out'])
                key = f'{job["experiment"]}_{job["seed"]}'
                latest[key] = {'resume': str(Path(job['out']) / 'final.resume.pt'),
                               'actions': summary['real_atomic_actions'], 'summary': str(Path(job['out']) / 'summary.json')}
                split = 'development' if summary['status'] == 'target_reached' and target >= 400000 else 'development_health'
                eval_pending.append({**{k:job[k] for k in ('experiment','seed','name','out','target')},
                                     'split': split, 'actions': summary['real_atomic_actions'],'control':'trained'})
                if summary['initial_actions']==0:
                    eval_pending.append({**{k:job[k] for k in ('experiment','seed','name','out')},
                        'name':'untrained_'+job['name'],'target':0,'split':'development_health','actions':0,
                        'weight_file':'initial.resume.weights.pt','control':'untrained'})
                if summary['status'] != 'target_reached' and not stopping:
                    pending.append({k:job[k] for k in ('experiment','seed','target')})
                print({'event': 'phase_finished', 'run': job['name'], 'status': summary['status'],
                       'actions': summary['real_atomic_actions']}, flush=True)
            if eval_active:
                child = eval_active['process']
                if child.poll() is None and time.monotonic()-eval_active['record']['started'] > 1800:
                    kill_tree(child.pid)
                    raise TimeoutError('evaluation watchdog')
                if child.poll() is not None:
                    eval_active['log'].close()
                    eval_active['record'].update(ended=time.monotonic(), returncode=child.returncode)
                    if child.returncode:
                        raise RuntimeError(f'evaluation failed: {eval_active["name"]}')
                    summary = read_json(Path(eval_active['evaluation_out']) / 'research_summary.json')
                    curve.append({k:eval_active[k] for k in ('experiment','seed','target','actions','split','name')}
                                 | {'summary': summary, 'evaluation': eval_active['evaluation_out'],
                                    'control':eval_active.get('control','trained')})
                    write_json(root / 'learning_curve.json', curve)
                    print({'event': 'evaluation_finished', 'name': eval_active['name'],
                           'split': eval_active['split'], 'score': summary['score_mean']}, flush=True)
                    eval_active = None
            if stopping:
                if not active and not eval_active:
                    phase = 'stopped_by_request'; status(); break
            else:
                phase = 'training_and_development'
                for slot, gpu in enumerate(gpus):
                    if slot in active or not pending:
                        continue
                    # Recheck before every reservation; do not evict unrelated users.
                    try:
                        actual = idle_gpu(gpu['index'])
                    except RuntimeError:
                        continue
                    if actual != gpu['uuid']:
                        raise ValueError('GPU identity changed')
                    job = pending.popleft(); chunk += 1
                    name = f'{job["experiment"]}_{job["seed"]}_a{job["target"]}_chunk{chunk:05d}'
                    out = root / 'training' / name
                    cfg = root / 'plan/configs' / f'{job["experiment"]}_{job["seed"]}.json'
                    command = ['-m','boardbench.v121.training','--config',cfg,'--out',out,
                               '--target-actions',job['target'],'--seconds',phase_seconds,'--device','cuda']
                    previous = latest.get(f'{job["experiment"]}_{job["seed"]}')
                    if previous:
                        if previous.get('numeric_import'):
                            command += ['--import-numeric',previous['resume'], '--parent-source',previous['parent_source'],
                                        '--parent-sha256',previous['parent_sha256']]
                        else:
                            command += ['--resume',previous['resume']]
                    record = {'name':name,'gpu':gpu['uuid'],'cpu_cores':cpu_groups[slot],
                              'started':time.monotonic(),'started_unix':time.time()}
                    records.append(record)
                    child, log = spawn(command,cpu_groups[slot],gpu['uuid'],name)
                    active[slot] = {**job,'name':name,'out':str(out),'process':child,'log':log,'record':record,'pid':child.pid}
                if not eval_active and eval_pending:
                    job=eval_pending.popleft()
                    name=job['split']+'_'+job['name']
                    out=root/'evaluations'/name
                    cfg=root/'plan/configs'/f'{job["experiment"]}_{job["seed"]}.json'
                    command=['-m','boardbench.v121.evaluation','--splits',root/'plan/splits.json',
                             '--split',job['split'],'--out',out,'--workers','4']
                    if job.get('policy_file'):
                        command += ['--policy',job['policy_file']]
                    else:
                        command += ['--config',cfg,'--weights',Path(job['out'])/job.get('weight_file','final.resume.weights.pt')]
                    record={'name':name,'cpu_cores':eval_cores,'started':time.monotonic(),'started_unix':time.time()}
                    eval_records.append(record)
                    child,log=spawn(command,eval_cores,'',name)
                    eval_active={**job,'name':name,'evaluation_out':str(out),'record':record,'process':child,'log':log}
                if ready_for_next_target(pending, active, eval_pending, eval_active, allocation):
                    target=next_milestone(target);queue_target()
                    print({'event':'next_checkpoint_target','actions':target},flush=True)
            if time.monotonic()-tick>5:
                status();tick=time.monotonic()
            time.sleep(.5)
    except BaseException as exc:
        last_error=f'{type(exc).__name__}: {exc}'
        phase='failed';status(last_error)
        raise
    finally:
        for job in list(active.values())+([eval_active] if eval_active else []):
            child=job['process']
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        until=time.monotonic()+60
        jobs=list(active.values())+([eval_active] if eval_active else [])
        while any(j['process'].poll() is None for j in jobs) and time.monotonic()<until:
            time.sleep(.25)
        for job in jobs:
            if job['process'].poll() is None:
                kill_tree(job['process'].pid)
            job['process'].wait(timeout=5)
            job['record'].update(ended=time.monotonic(),returncode=job['process'].returncode)
            job['log'].close()
        active.clear();eval_active=None
        for record in records + eval_records:
            if 'ended' not in record:
                record.update(ended=time.monotonic(), returncode=None, failed_before_registration=True)
        status()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def launch(repo, root, indices, phase_seconds=900., batched=False, collector_workers=0, numeric_recovery=None):
    preparation_started=time.monotonic()
    repo,root=Path(repo).resolve(),Path(root).resolve()
    if root.exists() or not root.is_relative_to(repo/'outputs/v121'):
        raise ValueError('new project V1.2.1 root required')
    if not indices or len(indices)!=len(set(indices)) or phase_seconds<=0:
        raise ValueError('distinct devices and positive phase watchdog required')
    if type(collector_workers) is not int or not 0 <= collector_workers <= 7 or (collector_workers and not batched):
        raise ValueError('worker count requires batched mode and at most seven workers')
    if numeric_recovery and not batched:
        raise ValueError('numeric recovery requires matched batched experiments')
    if 1 in indices:
        raise ValueError('GPU1 remains quarantined pending explicit device diagnosis')
    if os.getloadavg()[0]>len(os.sched_getaffinity(0))*.5:
        raise RuntimeError('CPU host load too high')
    gpus=[{'index':i,'uuid':idle_gpu(i)} for i in indices]
    cores=physical_cores(8*(len(gpus)+1))
    root.parent.mkdir(parents=True,exist_ok=True)
    lock=(root.parent/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root.mkdir();(root/'logs').mkdir()
    for relative in source_files(repo):
        dest=root/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/relative,dest)
    experiments = tuple(NUMERIC) if numeric_recovery else tuple(THROUGHPUT) if batched else IMPLEMENTED
    prepare(root/'plan', experiments, collector_workers)
    recovery = None
    if numeric_recovery:
        from .recovery import prepare_recovery
        recovery = prepare_recovery(numeric_recovery, root, experiments, (911,912,913))
    manifest=read_json(root/'plan/splits.json')
    # Explicit inference-only legacy import; old training artifacts remain untouched.
    legacy=repo/'outputs/v11/harmonies/policies/rl_trained_008192__ffd5e452377e19f0'
    legacy_meta=read_json(legacy/'manifest.json')
    old_weights=legacy/legacy_meta['artifact_path']
    if digest(old_weights)!=legacy_meta['artifact_sha256']:
        raise ValueError('INC artifact hash mismatch')
    (root/'baselines').mkdir()
    shutil.copy2(old_weights,root/'baselines/INC.weights.pt')
    baseline_specs=[{'id':'INC','training_seed':811,'solver':{'id':'ppo','params':
                     {'task_id':'harmonies','hidden':128,'device':'cpu','network':'flat_mlp'}},
                     'weights':str(root/'baselines/INC.weights.pt'),'weights_sha256':digest(old_weights)},
                    {'id':'Psearch_reference','solver':{'id':'timed_search','params':{'max_simulation_steps':384}}}]
    for spec in baseline_specs:
        write_json(root/'baselines'/f'{spec["id"]}.policy.json',{'policy':spec,'split_sha256':manifest['sha256'],
            'runtime_source_id':source_id(root/'source'),'historical_training_source':legacy_meta['training_code_version'] if spec['id']=='INC' else None,
            'mode':'explicit frozen inference reference; not a new training repetition'})
    allocation={'research_version':'1.2.1','source_id':source_id(root/'source'),'started_unix':time.time(),
                'host':os.uname().nodename,'gpus':gpus,'gpu_hours_cap':None,'user_authorization':'continuous until stop; supersedes4GPUh',
                'training_cpu_groups':[cores[i*8:(i+1)*8] for i in range(len(gpus))],
                'evaluation_cpu_cores':cores[-8:],'phase_seconds':phase_seconds,
                'training_seeds':[911,912,913],'experiments':list(experiments),
                'collector_mode':'batched_complete_games_v1' if batched else 'serial',
                'collector_workers':collector_workers,
                'recovery_sha256':digest(root/'recovery.json') if recovery else None,
                'pipeline_evaluations':bool(batched), 'evaluation_high_watermark':48,
                'watchdog_pid':os.getpid(),'accounting':'sum each GPU child reservation incl startup; CPU evaluation separately charged'}
    allocation['launcher_preparation_seconds']=time.monotonic()-preparation_started
    write_json(root/'allocation.json',allocation)
    env={**os.environ,'PYTHONPATH':str(root/'source/src'),'CUDA_VISIBLE_DEVICES':'',
         'PYTHONNOUSERSITE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    child=subprocess.Popen([sys.executable,'-m','boardbench.v121.launch','controller','--root',str(root)],env=env,start_new_session=True)
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    try:
        while child.poll() is None:
            if stop[0]:
                write_json(root/'STOP_REQUESTED.json',{'requested_unix':time.time(),'reason':'watchdog signal'})
                stop[0]=False
            path=root/'status.json'
            if path.exists() and time.time()-read_json(path)['updated_unix']>300:
                raise TimeoutError('controller heartbeat stale; keep existing checkpoints')
            time.sleep(2)
    finally:
        if child.poll() is None:
            kill_tree(child.pid)
        child.wait(timeout=10)
        write_json(root/'watchdog_result.json',{'returncode':child.returncode,'ended_unix':time.time(),
                                              'gpu_hours_cap':None,'source_id':allocation['source_id']})
    if child.returncode:
        raise SystemExit(child.returncode)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('launch','controller','stop'))
    parser.add_argument('--repo',default='.');parser.add_argument('--root',required=True)
    parser.add_argument('--gpus',nargs='+',type=int,default=[2,3,4,5])
    parser.add_argument('--phase-seconds',type=float,default=900.)
    parser.add_argument('--batched',action='store_true',help='fresh matched E00_T16/E01_T16/E10_T16/E11_T16 reruns')
    parser.add_argument('--collector-workers',type=int,default=0)
    parser.add_argument('--numeric-recovery',help='explicit new-root FP64 import of a stopped T16 campaign')
    args=parser.parse_args()
    if args.mode=='controller':controller(args.root)
    elif args.mode=='stop':
        root=Path(args.root);read_json(root/'allocation.json')
        write_json(root/'STOP_REQUESTED.json',{'requested_unix':time.time(),'reason':'explicit CLI stop'})
    else:launch(args.repo,args.root,args.gpus,args.phase_seconds,args.batched,args.collector_workers,args.numeric_recovery)
