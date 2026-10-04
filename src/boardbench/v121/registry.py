"""Immutable experiment definitions and independently generated research splits."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from boardbench.artifacts.store import write_json

IMPLEMENTED = ('E00', 'E01', 'E10', 'E11')
THROUGHPUT = {name + '_T16': name for name in IMPLEMENTED}
AVAILABLE = IMPLEMENTED + tuple(THROUGHPUT)
FIRST_ROUND = {
    'E00': (None, 'flat', 'fixed', 'V1.2 same-network re-run control'),
    'E01': ('E00', 'flat', 'shared', 'Shared raw-action scorer improves reuse'),
    'E10': ('E00', 'graph', 'fixed', 'Directed board/card relations improve representation'),
    'E11': ('E10', 'graph', 'shared', 'Interaction of structured representation and shared scorer'),
    'Cwide': ('E00', 'flat', 'fixed', '256x256 capacity control'),
    'Cpos': ('E01', 'flat', 'shared', 'Per-candidate real coordinates improve spatial sharing'),
    'Lgae0': ('E00', 'flat', 'fixed', 'GAE lambda0.95 reduces estimator variance'),
    'Afactor0': ('E00', 'flat', 'factorized', 'Conditional action decomposition improves exploration'),
}
FUTURE = {
    'R': (None, 'Freeze a reference platform using full development results'),
    'Rprogress': ('R', 'Public pattern progress features'),
    'Rconstraint': ('Rprogress', 'Shared/conflicting pattern constraints'),
    'Rcritic': ('R', 'Terrain and animal critics, unchanged total objective'),
    'Raux': ('R', 'Separately registered auxiliary supervision'),
    'Lthroughput': ('E00', 'Semantics-checked batched sampling; re-run matched controls'),
    'Rcurriculum': ('R', 'Reachable training-state curriculum with matched reset control'),
    'Rreset': ('R', 'Equal snapshot cost, no difficulty-selected reset control'),
    'Q0': ('R', 'Masked Double DQN with same encoding/head organization'),
    'Qbranch': ('Q0', 'Counterfactual replay branches'),
    'Qordinary': ('Q0', 'Matched simulation budget ordinary replay control'),
    'Pafter': ('R', 'One-step successor remaining-value scoring'),
    'B0': ('R', 'Same-architecture BC with trajectory-heldout diagnostics'),
    'Bnew': ('B0', 'Additional teacher trajectories'),
    'Bdag': ('B0', 'Matched-cost student-state teacher correction'),
    'Mturn': ('R', 'Variable-length turn macros with atomic submission'),
    'Hgoal': ('R', 'Interruptible high-level animal goals'),
    'Pmodel': ('R', 'Pure policy reference for search comparison'),
    'Psearch': ('R', 'Non-learning search control'),
    'Pfrozen': ('R', 'Frozen learned model plus search'),
    'Piter': ('Pfrozen', 'Iterative learned search'),
    'Prouting': ('Pfrozen', 'Learned online compute allocation with matched controls'),
}
FAMILIES = {
    'V01': ['E10', 'E11'], 'V02': ['Rprogress', 'Rconstraint'], 'V03': ['E01', 'E11'],
    'V04': ['Afactor0'], 'V05': ['Pafter'], 'V06': ['Mturn'], 'V07': ['Hgoal'],
    'V08': ['Lgae0', 'Rcritic', 'Raux', 'Lthroughput'], 'V09': ['Q0'], 'V10': ['Rcurriculum', 'Rreset'],
    'V11': ['B0', 'Bnew', 'Bdag'], 'V12': ['Qbranch', 'Qordinary'],
    'V13': ['Pmodel', 'Psearch', 'Pfrozen', 'Piter'], 'V14': ['Prouting'],
}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def definitions():
    result = {}
    for name in list(FIRST_ROUND) + list(FUTURE):
        first = FIRST_ROUND.get(name)
        parent, hypothesis = (first[0], first[3]) if first else FUTURE[name]
        architecture = {'state': first[1], 'head': first[2], 'hidden': 256 if name == 'Cwide' else 128,
                        'graph_width': 32, 'graph_layers': 2, 'action_width': 64} if first else {'parent_platform': 'R_pending'}
        result[name] = {
            'experiment_id': name, 'parent_id': parent, 'hypothesis': hypothesis,
            'changed_factors': [] if name == 'E00' else [hypothesis],
            'unchanged_factors': ['harmonies_solo_a_v1', 'true_total_score', 'STOP', 'timed_score_v1'],
            'information_access': 'public observation and rules only; no private supply/RNG',
            'objective': 'full-game total score delta /150; gamma1; STOP terminal',
            'training_distribution': 'fresh train split; normal starts for first four',
            'architecture': architecture, 'algorithm': 'PPO_MC' if name in IMPLEMENTED else 'planned_not_frozen',
            'budget': {'unit': 'real_atomic_actions', 'milestones': [400000, 1000000, 2000000, 5000000, 10000000],
                       'K1_target_per_seed': 2000000, 'training_seeds': [911, 912, 913]},
            'deployment_budget': {'seconds': 3., 'threads': 1, 'device': 'cpu'},
            'selection_rule': 'full development128; confirmation256; freeze before final test512',
            'status': 'planned', 'implementation_available': name in IMPLEMENTED,
        }
    for name, parent in THROUGHPUT.items():
        spec = deepcopy(result[parent])
        spec.update(experiment_id=name, parent_id=parent,
            hypothesis='Batched sixteen-game synchronous collector improves resource efficiency',
            changed_factors=['sampling execution only; fixed complete-game update cohort'],
            unchanged_factors=spec['unchanged_factors'] + ['network', 'batch16', 'MC_returns', 'PPO_hyperparameters'],
            collector={'kind': 'batched_complete_games_v1', 'auto_reset': False,
                       'row_order': 'episode_major', 'policy_updates_during_collection': False},
            implementation_available=True)
        result[name] = spec
    return result


def make_splits(seed=20261004, train_count=65536):
    if type(train_count) is not int or train_count < 1:
        raise ValueError('positive training pool size required')
    sizes = [('train', train_count), ('development', 128), ('confirmation', 256), ('test', 512)]
    pool = random.Random(seed).sample(range(20000000, 30000000), sum(n for _, n in sizes))
    sets, start = {}, 0
    for name, count in sizes:
        sets[name] = pool[start:start + count]
        start += count
    sets['development_health'] = sets['development'][:32]
    sets['historical_audit'] = list(range(10000000, 10000048))
    manifest = {'format_version': 1, 'generator_seed': seed, 'sets': sets,
                'test_state': 'unexecuted; ids visible, outcomes not generated'}
    manifest['sha256'] = identity(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    payload = {k: v for k, v in manifest.items() if k != 'sha256'}
    if identity(payload) != manifest['sha256']:
        raise ValueError('split manifest hash mismatch')
    sets = manifest['sets']
    seen = set()
    for key in ('train', 'development', 'confirmation', 'test', 'historical_audit'):
        values = sets[key]
        if not values or any(type(x) is not int for x in values) or len(set(values)) != len(values) or seen.intersection(values):
            raise ValueError('overlapping, empty or non-unique split')
        seen.update(values)
    if sets['development_health'] != sets['development'][:32]:
        raise ValueError('health split must be the fixed first32 development games')


def training_config(experiment_id, seed, manifest, collector_workers=0):
    validate_manifest(manifest)
    if experiment_id not in AVAILABLE:
        raise ValueError('experiment implementation not available')
    spec = definitions()[experiment_id]
    sets = manifest['sets']
    config = {'research_version': '1.2.1', 'experiment_id': experiment_id, 'experiment_sha256': identity(spec),
            'split_sha256': manifest['sha256'], 'training_seed': seed, 'algorithm': 'ppo',
            'splits': {'train': sets['train'], 'validation': sets['development'] + sets['confirmation'],
                       'test': sets['test'] + sets['historical_audit']},
            'network': {'experiment_id': experiment_id, **deepcopy(spec['architecture'])},
            'threads': 1, 'batch_episodes': 16, 'epochs': 4, 'minibatch_size': 256,
            'checkpoint_batches': 16, 'learning_rate': .0003, 'reward_scale': 150.,
            'entropy_coef': .01, 'target_kl': .02, 'clip_ratio': .2}
    if experiment_id in THROUGHPUT:
        if type(collector_workers) is not int or not 0 <= collector_workers <= 7:
            raise ValueError('collector workers must fit the eight-core allocation')
        config['collector'] = {**spec['collector'], 'workers': collector_workers}
    elif collector_workers:
        raise ValueError('serial experiment cannot use collector workers')
    return config


def prepare(out, experiments=IMPLEMENTED, collector_workers=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = make_splits()
    write_json(out / 'splits.json', manifest)
    families = deepcopy(FAMILIES)
    families['V08'] += list(THROUGHPUT)
    write_json(out / 'registry.json', {'version': '1.2.1', 'families': families, 'experiments': definitions()})
    for experiment in experiments:
        for seed in (911, 912, 913):
            write_json(out / 'configs' / f'{experiment}_{seed}.json', training_config(experiment, seed, manifest, collector_workers))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    prepare(parser.parse_args().out)
