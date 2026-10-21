"""Skills first; execution-gated current-worker manager; frozen paired audit."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, source_files, source_id, write_json
from boardbench.v121.registry import identity, validate_manifest
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.launch import validate_allocation
from boardbench.v127.training import load_demonstrations
from boardbench.v130.pipeline import checkpoint_spec, paired
from boardbench.v134 import campaign as orchestration
from boardbench.v135.campaign import immutable
from .data import build
from .probe import execution_gate
from .training import METHODS


def new_manifest(old, old_sha, count=512):
    validate_manifest(old)
    forbidden = set().union(*(set(v) for v in old['sets'].values()))
    rng, fresh = random.Random(13620261021), []
    while len(fresh) < 3 * count:
        seed = rng.randrange(60000000, 70000000)
        if seed not in forbidden:
            fresh.append(seed); forbidden.add(seed)
    result = {'format_version': 1, 'sets': {**deepcopy(old['sets']), 'development': fresh[:2*count],
               'development_health': fresh[:2*count][:32], 'confirmation': fresh[2*count:]},
               'generator_seed': 13620261021, 'parent_split_sha256': old_sha, 'test_state': 'unchanged and sealed'}
    result['sha256'] = identity(result)
    validate_manifest(result)
    return result


def prepare(repo, root, config):
    started = time.monotonic()
    parent = checkpoint_spec(repo / config['parent_checkpoint'])
    directory = Path(parent['checkpoint']).parent
    if read_json(directory / 'summary.json')['status'] != 'completed':
        raise ValueError('predesignated sticky-a final parent must be complete')
    base = read_json(directory / 'config.json')
    if digest(base['split_manifest']) != base['split_sha256']:
        raise ValueError('parent split changed')
    old = read_json(base['split_manifest'])
    manifest = new_manifest(old, base['split_sha256'])
    split = root / 'splits.json'
    immutable(split, manifest); immutable(root / 'config.json', config)
    dataset = root / 'data/skills.npz'
    _, held = load_demonstrations(base, manifest)
    allowed = set(manifest['sets']['train']) - held - set(base['excluded_train_seeds'])
    if not dataset.exists():
        build(config['skill_corpus'], allowed, dataset)
    spec = {'path': str(dataset), 'sha256': digest(dataset), 'probes': str(dataset.with_suffix('.probes.json')),
            'probes_sha256': digest(dataset.with_suffix('.probes.json'))}
    stage, jobs = root / 'stage1', []
    count = read_json(Path(parent['checkpoint']).with_suffix('.json'))['ppo_actions']
    for method in METHODS:
        for replica in range(2):
            name = method + '_' + 'ab'[replica]
            trial = {**base, 'skill_version': '1.3.6', 'skill_method': method, 'skill_dataset': spec,
                     'skill_corpus': config['skill_corpus'], 'skill_origin_actions': count,
                     'method': 'sticky', 'arm': 'persistent', 'replicate': replica, 'seed': 13601+replica,
                     'continuation': parent, 'split_manifest': str(split), 'split_sha256': digest(split),
                     'curve_seeds': manifest['sets']['development'][:512], 'final_seeds': manifest['sets']['development'][512:],
                     'skill_warmup_steps': 256, 'skill_warmup_lr': .0003, 'skill_anchor_kl': 1.,
                     'skill_bc_weight': .02, 'skill_bc_decay_actions': config['stage1_actions'],
                     'manager_label_states': 32, 'manager_label_updates': 256,
                     'workers': config['training']['workers'], 'max_seconds': config['stage1_seconds'],
                     'target_actions': count+config['stage1_actions'],
                     'milestones': [count+1000000, count+config['stage1_actions']],
                     'trajectory_directory': str(stage / 'trials' / name / 'trajectories')}
            jobs.append([name, trial])
    immutable(stage / 'jobs.json', jobs); immutable(stage / 'config.json', config)
    (stage / 'logs').mkdir(exist_ok=True)
    write_json(root / 'prepare_cost.json', {'elapsed_seconds': time.monotonic()-started,
                                          'cpu_core_seconds': len(os.sched_getaffinity(0))*(time.monotonic()-started)})


def report(root, state):
    orchestration.report(root, state, title='V136 public goal execution skills')


def promotion(root):
    scores = {name: read_json(root / 'stage1/trials' / name / 'evaluations/final.json')['mean_score']
              for name, _ in read_json(root / 'stage1/jobs.json')}
    means = {method: float(np.mean([scores[method+'_'+s] for s in 'ab'])) for method in METHODS}
    winner = max(METHODS[1:], key=lambda m: means[m])
    gates = {s: execution_gate(read_json(root / f'stage1/trials/control_{s}/execution_probe.json'),
                               read_json(root / f'stage1/trials/skills_{s}/execution_probe.json')) for s in 'ab'}
    manager = bool(winner == 'skills' and means['skills'] >= means['control'] - 1
                   and all(g['passed'] for g in gates.values()))
    return {'means': means, 'scores': scores, 'selected': ['control', winner] + (['manager'] if manager else []),
            'execution_gates': gates, 'manager_admitted': manager,
            'rule': 'fixed development audit512 endpoint winner; manager only if both training execution gates pass and skills >= control-1',
            'interpretation': 'exploratory continuation, no confirmation selection or automatic superiority claim'}


def stage2(root):
    selected = promotion(root)
    immutable(root / 'promotion.json', selected)
    config, stage, jobs = read_json(root / 'config.json'), root / 'stage2', []
    for name, trial in read_json(root / 'stage1/jobs.json'):
        if trial['skill_method'] not in selected['selected']:
            continue
        parent = checkpoint_spec(root / 'stage1/trials' / name / 'final.pt')
        count = read_json(Path(parent['checkpoint']).with_suffix('.json'))['ppo_actions']
        variants = [trial['skill_method']]
        if trial['skill_method'] == 'skills' and selected['manager_admitted']:
            variants.append('manager')
        for method in variants:
            child = method + '_' + name[-1]
            jobs.append([child, {**trial, 'skill_method': method, 'arm': 'manager' if method == 'manager' else 'persistent',
                         'continuation': parent, 'target_actions': count+config['stage2_actions'],
                         'max_seconds': config['stage2_seconds'],
                         'milestones': [count+config['stage2_actions']//2, count+config['stage2_actions']],
                         'trajectory_directory': str(stage / 'trials' / child / 'trajectories')}])
    immutable(stage / 'jobs.json', jobs)
    immutable(stage / 'config.json', {**config, 'gpu_uuids': config['gpu_uuids'][:len(jobs)//2],
                                     'preflight_cpu_core_seconds': 0., 'preflight_gpu_seconds': 0.})
    (stage / 'logs').mkdir(exist_ok=True)
    return stage


def finalize(root):
    import torch
    from .training import Trainer
    frozen = read_json(root / 'finalization.json')
    seeds = read_json(root / 'splits.json')['sets']['confirmation']
    results, ablations = {}, {}
    for name, spec in frozen['checkpoints'].items():
        c = read_json(root / 'stage1/jobs.json')[0][1] if name == 'parent' else read_json(Path(spec['checkpoint']).parent / 'config.json')
        t = Trainer(c, 'cuda')
        try:
            if digest(spec['checkpoint']) != spec['sha256']:
                raise ValueError('frozen endpoint hash changed')
            if name == 'parent':
                t.import_sampling_continuation(spec)
            else:
                saved = torch.load(spec['checkpoint'], map_location='cuda', weights_only=False)
                if saved['version'] != 136 or saved['source_id'] != source_id() or saved['config'] != c:
                    raise ValueError('frozen source/config mismatch')
                t.model.load_state_dict(saved['model']); t.manager.load_state_dict(saved['manager'])
            results[name] = t.evaluate(seeds)
            write_json(root / 'confirmation' / (name+'.json'), results[name])
            if name != 'parent' and c['skill_method'] != 'control':
                t.model.intervention = 'zero'; t.model.advice_intervention = 'zero'
                removed = t.evaluate(c['final_seeds'])
                intact = read_json(Path(spec['checkpoint']).parent / 'evaluations/final.json')
                ablations[name] = paired(intact, removed)
                write_json(root / 'ablations' / (name+'.json'), {'normal': intact, 'no_worker_goal': removed,
                           'normal_minus_removed': ablations[name], 'warning': 'off-distribution development diagnostic'})
        finally:
            t.close()
    contrasts = {}
    for method in read_json(root / 'promotion.json')['selected']:
        values, gaps, animals = [], [], []
        for i, seed in enumerate(seeds):
            if any(r['games'][i]['environment_seed'] != seed for r in results.values()):
                raise ValueError('confirmation pairing mismatch')
            value = float(np.mean([results[method+'_'+s]['games'][i]['score'] for s in 'ab']))
            values.append(value)
            animals.append(float(np.mean([results[method+'_'+s]['games'][i]['animal_score'] for s in 'ab'])))
            gaps.append(value-np.mean([results['control_'+s]['games'][i]['score'] for s in 'ab']))
        mean, half = float(np.mean(gaps)), float(1.96*np.std(gaps, ddof=1)/np.sqrt(len(gaps)))
        contrasts[method] = {'mean_score': float(np.mean(values)), 'animal_mean': float(np.mean(animals)),
                             'minus_control': mean, 'environment_paired_descriptive95': [mean-half, mean+half]}
    write_json(root / 'confirmation.json', {'n': len(seeds), 'contrasts': contrasts, 'worker_goal_ablations': ablations,
               'models': {n: r['mean_score'] for n, r in results.items()},
               'paired_vs_parent': {n: paired(r, results['parent']) for n, r in results.items() if n != 'parent'},
               'scope': 'two training seeds, model-only, true total score; no formal test or online LLM'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'launch', 'resume', 'guard', 'run', 'finalize'))
    p.add_argument('--repo', required=True); p.add_argument('--root', required=True); p.add_argument('--config')
    a = p.parse_args(); repo, root = Path(a.repo).resolve(), Path(a.root).resolve()
    if not root.is_relative_to(repo / 'outputs/v136'):
        raise ValueError('dedicated V136 output required')
    if a.mode == 'prepare':
        return prepare(repo, root, read_json(a.config))
    config = read_json(root / 'config.json')
    if os.uname().nodename != config['host']:
        raise ValueError('authorized a226 only')
    if a.mode in ('launch', 'resume'):
        if a.mode == 'launch':
            if (root / 'launch.json').exists():
                raise ValueError('already launched')
            gate, local = read_json(root / 'preflight_cost.json'), read_json(root / 'local_checks.json')
            if not gate['passed'] or not local['passed'] or gate['source_id'] != source_id(repo) or local['source_id'] != source_id(repo):
                raise ValueError('source-matched local and actual CUDA gates required')
            cpus = physical_cpus(); count = 6*(config['training']['workers']+1)+1
            if count > min(128, len(cpus)//2):
                raise ValueError('half-machine physical CPU cap')
            cpus = cpus[:count]
            validate_allocation(cpus, config['gpu_uuids'], config['training']['workers']+1, 2, 6)
            config.update(preflight_cpu_core_seconds=gate['cpu_core_seconds']+local['cpu_core_seconds']+read_json(root/'prepare_cost.json')['cpu_core_seconds'],
                          preflight_gpu_seconds=gate['gpu_seconds'])
            write_json(root/'config.json', config)
            for relative in source_files(repo):
                dest = root/'source'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo/relative, dest)
            write_json(root/'launch.json', {'source_id': source_id(repo), 'cpus': cpus, 'campaign_module': 'boardbench.v136.campaign',
                       'started_unix': time.time(), 'deadline_unix': time.time()+config['guard_seconds']})
        launch = read_json(root/'launch.json')
        if time.time() >= launch['deadline_unix']-900:
            raise ValueError('deadline expired; no renewal')
        env = {**os.environ, 'PYTHONPATH': str(root/'source/src'), 'PYTHONNOUSERSITE': '1',
               **{k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
        os.execve(sys.executable, [sys.executable, '-u', '-m', 'boardbench.v136.campaign', 'guard', '--repo', str(repo), '--root', str(root)], env)
    if a.mode == 'guard':
        orchestration.guard(root, repo)
    elif a.mode == 'run':
        orchestration.run(root, repo, stage_builder=stage2, reporter=report,
                          training_module='boardbench.v136.training', campaign_module='boardbench.v136.campaign')
    else:
        finalize(root)


if __name__ == '__main__':
    main()
