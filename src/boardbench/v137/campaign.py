"""Bounded own-state labels -> eight paired trials -> refresh -> fixed audit."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, source_files, source_id, write_json
from boardbench.v12.budget import kill_tree
from boardbench.v121.registry import identity, validate_manifest
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.launch import validate_allocation, physical_gpu_seconds
from boardbench.v130.pipeline import checkpoint_spec, paired
from boardbench.v133.campaign import wait_resources
from boardbench.v134 import campaign as orchestration
from boardbench.v135.campaign import immutable
from .data import interval
from .training import METHODS


def new_manifest(old, old_sha, count=512):
    validate_manifest(old)
    forbidden = set().union(*(set(v) for v in old['sets'].values()))
    fresh, rng = [], random.Random(13720261022)
    while len(fresh) < count*3:
        seed = rng.randrange(70000000, 80000000)
        if seed not in forbidden:
            fresh.append(seed); forbidden.add(seed)
    value = {'format_version': 1, 'generator_seed': 13720261022, 'sets': deepcopy(old['sets']),
             'parent_split_sha256': old_sha, 'test_state': 'unchanged and sealed'}
    value['sets'].update(development=fresh[:count*2], development_health=fresh[:count*2][:32], confirmation=fresh[count*2:])
    value['sha256'] = identity(value); validate_manifest(value)
    return value


def prepare(repo, root, config):
    parents = {s: checkpoint_spec(repo/config['parent_directory']/f'skills_{s}/final.pt') for s in 'ab'}
    bases = {s: read_json(Path(spec['checkpoint']).parent/'config.json') for s, spec in parents.items()}
    for spec in parents.values():
        if read_json(Path(spec['checkpoint']).parent/'summary.json')['status'] != 'completed':
            raise ValueError('both V136 skill parents must be complete')
    if any(digest(b['split_manifest']) != b['split_sha256'] for b in bases.values()):
        raise ValueError('parent split changed')
    old = read_json(bases['a']['split_manifest'])
    if old['sets']['train'] != read_json(bases['b']['split_manifest'])['sets']['train']:
        raise ValueError('parent training pools disagree')
    manifest = new_manifest(old, bases['a']['split_sha256']); split = root/'splits.json'
    immutable(root/'config.json', config); immutable(root/'parents.json', parents); immutable(split, manifest)
    immutable(root/'reference.json', checkpoint_spec(repo/config['strong_reference']))
    jobs = []
    for method in METHODS[:-1]:
        for replica, s in enumerate('ab'):
            base, parent = bases[s], parents[s]
            count = read_json(Path(parent['checkpoint']).with_suffix('.json'))['ppo_actions']
            name = method+'_'+s
            trial = {**base, 'selection_version': '1.3.7', 'selection_method': method,
                     'skill_method': 'skills' if method == 'control' else 'manager',
                     'arm': 'persistent' if method == 'control' else 'manager',
                     'selection_origin_actions': count, 'manager_warmup_actions': 1000000,
                     'goal_dataset': None, 'goal_fit_updates': 512, 'goal_temperature': 5., 'joint_skill_bc': .005,
                     'seed': 13701+replica, 'replicate': replica, 'continuation': parent,
                     'split_manifest': str(split), 'split_sha256': digest(split),
                     'curve_seeds': manifest['sets']['development'][:512], 'final_seeds': manifest['sets']['development'][512:],
                     'workers': config['training']['workers'], 'max_seconds': config['stage1_seconds'],
                     'target_actions': count+config['stage1_actions'],
                     'milestones': [count+1000000, count+config['stage1_actions']],
                     'trajectory_directory': str(root/'stage1/trials'/name/'trajectories')}
            jobs.append([name, trial])
    immutable(root/'templates.json', jobs)


def labels(root, repo, index):
    """Two independent one-GPU batches, shared CPU lock and explicit deadline."""
    config, launch = read_json(root/'config.json'), read_json(root/'launch.json')
    directory = root/f'labels{index}'; directory.mkdir(exist_ok=True)
    parents = read_json(root/'parents.json') if index == 1 else {
        s: checkpoint_spec(root/f'stage1/trials/joint_{s}/final.pt') for s in 'ab'}
    for i, s in enumerate('ab'):
        if (directory/s/'dataset.json').exists():
            saved = read_json(directory/s/'dataset.json')
            settings = {**config['labels'], 'offset': 512 if index == 1 else 1024,
                        'future_seed': 1370000000 + index*1000000 + i*100000}
            if saved['parent'] != parents[s] or saved['settings'] != settings or digest(saved['path']) != saved['sha256']:
                raise ValueError('changed completed goal labels')
    pending = [s for s in 'ab' if not (directory/s/'dataset.json').exists()]
    if not pending:
        return {s: {k: read_json(directory/s/'dataset.json')[k] for k in ('path', 'sha256')} for s in 'ab'}
    tick = time.monotonic()
    lock = wait_resources(root, repo, {**config, 'gpu_uuids': config['gpu_uuids'][:2]},
                          launch['deadline_unix']-2400, f'labels{index}')
    waited = time.monotonic()-tick
    active, attempts = [], []
    old = read_json(root/'label_cost.json') if (root/'label_cost.json').exists() else {'attempts': [], 'waiting_seconds': 0.}
    for attempt in old['attempts']:
        if 'elapsed_seconds' not in attempt:
            # The idle-resource gate above excludes a surviving CUDA label worker.
            # Charge conservatively through the original deadline, never invent zero cost.
            attempt.update(elapsed_seconds=max(0., min(time.time(), launch['deadline_unix'])-attempt['started_unix']),
                           returncode=None, recovery='unknown interrupted interval, conservative bound')
    old['waiting_seconds'] += waited
    deadline = min(time.time()+config['label_seconds'], launch['deadline_unix']-1800)
    try:
        for i, s in enumerate('ab'):
            settings = {**config['labels'], 'offset': 512 if index == 1 else 1024,
                        'future_seed': 1370000000 + index*1000000 + i*100000}
            job = {'parent': parents[s], 'settings': settings}; immutable(directory/(s+'.json'), job)
            if s not in pending:
                continue
            cpus = launch['cpus'][i*7:(i+1)*7]
            log = (directory/(s+'.log')).open('a')
            child = subprocess.Popen(['taskset', '-c', ','.join(map(str, cpus)), sys.executable, '-u', '-m', 'boardbench.v137.data',
                '--job', str(directory/(s+'.json')), '--out', str(directory/s), '--device', 'cuda'],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': config['gpu_uuids'][i]}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            attempt = {'stage': index, 'replica': s, 'pid': child.pid, 'gpu': config['gpu_uuids'][i],
                       'cpus': cpus, 'started_unix': time.time()}
            attempts.append(attempt); active.append((child, log, attempt))
        old['attempts'].extend(attempts); write_json(root/'label_cost.json', old)
        while any(child.poll() is None for child, _, _ in active):
            if time.time() >= deadline:
                raise TimeoutError('goal-label phase budget exhausted; completed batches preserved')
            for child, _, attempt in active:
                if child.poll() is not None and 'elapsed_seconds' not in attempt:
                    attempt.update(elapsed_seconds=time.time()-attempt['started_unix'], returncode=child.returncode)
                    if child.returncode:
                        raise RuntimeError('goal-label worker failed')
            write_json(root/'label_cost.json', old)
            time.sleep(5)
        if any(child.returncode for child, _, _ in active):
            raise RuntimeError('goal-label worker failed')
    finally:
        for child, log, attempt in active:
            if child.poll() is None:
                kill_tree(child.pid); child.wait()
            if 'elapsed_seconds' not in attempt:
                attempt.update(elapsed_seconds=time.time()-attempt['started_unix'], returncode=child.returncode)
            log.close()
        old['controller_seconds'] = old.get('controller_seconds', 0.)+time.monotonic()-tick-waited
        old.update(gpu_hours=physical_gpu_seconds(old['attempts'], time.time())/3600,
                   cpu_core_hours=(sum(len(a['cpus'])*a['elapsed_seconds'] for a in old['attempts'])
                                   +old['waiting_seconds']+old['controller_seconds'])/3600)
        write_json(root/'label_cost.json', old); lock.close()
    return {s: {k: read_json(directory/s/'dataset.json')[k] for k in ('path', 'sha256')} for s in 'ab'}


def report(root, state):
    orchestration.report(root, state, title='V137 learned goal selection: fixed-worker and joint PPO')


def stage2(root):
    config = read_json(root/'config.json'); repo = root.parents[2]
    values = {name: read_json(root/'stage1/trials'/name/'evaluations/final.json')['mean_score']
              for name, _ in read_json(root/'stage1/jobs.json')}
    immutable(root/'promotion.json', {'selected': list(METHODS), 'scores': values,
              'rule': 'retain every mechanism through matched fixed endpoint; no tiny-gap elimination'})
    refreshed = labels(root, repo, 2)
    jobs = []
    for name, trial in read_json(root/'stage1/jobs.json'):
        parent = checkpoint_spec(root/'stage1/trials'/name/'final.pt')
        count = read_json(Path(parent['checkpoint']).with_suffix('.json'))['ppo_actions']
        for method in ([trial['selection_method'], 'refresh'] if trial['selection_method']=='joint' else [trial['selection_method']]):
            child = method+'_'+name[-1]
            jobs.append([child, {**trial, 'selection_method': method, 'continuation': parent,
                         'goal_dataset': refreshed[name[-1]] if method=='refresh' else trial['goal_dataset'],
                         'target_actions': count+config['stage2_actions'], 'max_seconds': config['stage2_seconds'],
                         'milestones': [count+config['stage2_actions']//2, count+config['stage2_actions']],
                         'trajectory_directory': str(root/'stage2/trials'/child/'trajectories')}])
    stage = root/'stage2'; immutable(stage/'jobs.json', jobs)
    immutable(stage/'config.json', {**config, 'preflight_cpu_core_seconds': 0., 'preflight_gpu_seconds': 0.})
    (stage/'logs').mkdir(exist_ok=True)
    return stage


def summarize(results, seeds):
    if any(len(r['games']) != len(seeds) or [g['environment_seed'] for g in r['games']] != seeds for r in results.values()):
        raise ValueError('paired confirmation alignment mismatch')
    parent = np.mean([[g['score'] for g in results[n]['games']] for n in ('parent', 'parent_b')], axis=0)
    control = np.mean([[g['score'] for g in results['control_'+s]['games']] for s in 'ab'], axis=0)
    reference = np.asarray([g['score'] for g in results['strong_reference']['games']])
    summary = {}
    for method in METHODS:
        score = np.mean([[g['score'] for g in results[method+'_'+s]['games']] for s in 'ab'], axis=0)
        animal = np.mean([[g['animal_score'] for g in results[method+'_'+s]['games']] for s in 'ab'])
        summary[method] = {'mean_score': float(score.mean()), 'animal_mean': float(animal),
                           'vs_fixed_skills': interval(score-parent), 'vs_continued_ppo': interval(score-control),
                           'vs_previous_strong_reference': interval(score-reference)}
    return {'n': len(seeds), 'methods': summary, 'models': {k:r['mean_score'] for k,r in results.items()},
            'scope': 'two training replicas; environment-paired descriptive intervals, not independent training runs'}


def finalize(root):
    import torch
    from .training import Trainer
    from .data import reader
    frozen = read_json(root/'finalization.json')['checkpoints']
    frozen.update(parent_b=read_json(root/'parents.json')['b'], strong_reference=read_json(root/'reference.json'))
    immutable(root/'evaluation_manifest.json', frozen)
    seeds = read_json(root/'splits.json')['sets']['confirmation']; results = {}
    for name, spec in frozen.items():
        t = reader(spec, 'cuda', 6)
        try:
            results[name] = t.evaluate(seeds)
            write_json(root/'confirmation'/(name+'.json'), results[name])
        finally:
            t.close()
    write_json(root/'confirmation.json', summarize(results, seeds))


def run(root, repo):
    def stop(*_):
        raise InterruptedError('V137 original deadline; preserve progress')
    old = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        if source_id() != read_json(root/'launch.json')['source_id']:
            raise ValueError('frozen source mismatch')
        datasets = labels(root, repo, 1)
        jobs = [[name, {**trial, 'goal_dataset': datasets[name[-1]]}] for name, trial in read_json(root/'templates.json')]
        immutable(root/'stage1/jobs.json', jobs)
        config = read_json(root/'config.json')
        immutable(root/'stage1/config.json', {**config, 'gpu_uuids': config['gpu_uuids'][:3]})
        (root/'stage1/logs').mkdir(exist_ok=True)
        orchestration.run(root, repo, stage_builder=stage2, reporter=report,
                          training_module='boardbench.v137.training', campaign_module='boardbench.v137.campaign')
        if (root/'completion.json').exists():
            result = read_json(root/'completion.json'); result['goal_label_cost'] = read_json(root/'label_cost.json')
            write_json(root/'completion.json', result)
    finally:
        for s,h in old.items(): signal.signal(s,h)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare','launch','resume','guard','run','finalize'))
    p.add_argument('--repo', required=True); p.add_argument('--root', required=True); p.add_argument('--config')
    a = p.parse_args(); root, repo = Path(a.root).resolve(), Path(a.repo).resolve()
    if not root.is_relative_to(repo/'outputs/v137'):
        raise ValueError('dedicated V137 output required')
    if a.mode == 'prepare': return prepare(repo, root, read_json(a.config))
    config = read_json(root/'config.json')
    if os.uname().nodename != config['host']: raise ValueError('a226 only')
    if a.mode in ('launch','resume'):
        if a.mode == 'launch':
            if (root/'launch.json').exists(): raise ValueError('already launched')
            gate, local = read_json(root/'preflight_cost.json'), read_json(root/'local_checks.json')
            if not gate['passed'] or not local['passed'] or gate['source_id'] != source_id(repo) or local['source_id'] != source_id(repo):
                raise ValueError('source-matched local and actual CUDA gates required')
            cpus = physical_cpus(); count = 8*(config['training']['workers']+1)+1
            if count > min(128,len(cpus)//2): raise ValueError('half-machine physical CPU cap')
            cpus = cpus[:count]; validate_allocation(cpus,config['gpu_uuids'],config['training']['workers']+1,2,8)
            config.update(preflight_cpu_core_seconds=gate['cpu_core_seconds']+local['cpu_core_seconds'],preflight_gpu_seconds=gate['gpu_seconds'])
            write_json(root/'config.json',config)
            for relative in source_files(repo):
                dest = root/'source'/relative; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(repo/relative,dest)
            write_json(root/'launch.json',{'source_id':source_id(repo),'cpus':cpus,'campaign_module':'boardbench.v137.campaign',
                       'started_unix':time.time(),'deadline_unix':time.time()+config['guard_seconds']})
        if time.time() >= read_json(root/'launch.json')['deadline_unix']-900: raise ValueError('original deadline expired')
        env = {**os.environ,'PYTHONPATH':str(root/'source/src'),'PYTHONNOUSERSITE':'1',
               **{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
        os.execve(sys.executable,[sys.executable,'-u','-m','boardbench.v137.campaign','guard','--repo',str(repo),'--root',str(root)],env)
    if a.mode=='guard': orchestration.guard(root,repo)
    elif a.mode=='run': run(root,repo)
    else: finalize(root)


if __name__ == '__main__': main()
