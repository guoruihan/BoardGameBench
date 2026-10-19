"""Freeze public diagnostic states from completed V1.2.2 development replays."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.environments.harmonies import Harmonies, STACKS
from boardbench.v121.registry import validate_manifest

KINDS = ('choose_bundle', 'place_token', 'take_card', 'place_animal', 'end_turn')
PHASES = ('early', 'middle', 'late')


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def phase(obs):
    return 'early' if obs['turn_index'] <= 3 else 'middle' if obs['turn_index'] <= 7 else 'late'


def board_metrics(obs):
    score = obs['score_breakdown']
    return {'score': score['total'], 'animals': score['animals'],
            'terrain': score['total'] - score['animals'], 'turns': obs['turn_index'],
            'tokens': sum(len(STACKS[c['stack_id']]) for c in obs['board']),
            'animals_placed': sum(c['animal'] is not None for c in obs['board']),
            'completed_cards': len(obs['completed_cards']),
            'empty_cells': sum(c['stack_id'] == 0 for c in obs['board'])}


def validate_public(obs):
    expected = set(Harmonies().observe())
    if set(obs) != expected or obs['task_id'] != 'harmonies' or obs['terminated']:
        raise ValueError('nonterminal canonical public Harmonies observation required')
    if Harmonies.from_observation(obs, 0).observe() != obs:
        raise ValueError('public observation/derived legality or score mismatch')


def select_state(observations, target_phase, target_kind):
    candidates = [(index, obs) for index, obs in enumerate(observations) if not obs['terminated']]
    if not candidates:
        raise ValueError('empty replay')
    def key(pair):
        index, obs = pair
        return (target_kind not in {a['type'] for a in obs['legal_actions']},
                abs(PHASES.index(phase(obs)) - PHASES.index(target_phase)), object_hash([index, obs]))
    return min(candidates, key=key)


def build_bank(source_root, manifest, environments=32):
    validate_manifest(manifest)
    root = Path(source_root)
    plan, allocation = read_json(root / 'plan.json'), read_json(root / 'allocation.json')
    if plan['split'] != 'development' or plan['split_sha256'] != manifest['sha256']:
        raise ValueError('source must be development with the identical split manifest')
    if not 1 <= environments <= len(manifest['sets']['development']):
        raise ValueError('invalid environment count')
    states, sources = [], []
    for index, seed in enumerate(manifest['sets']['development'][:environments]):
        for producer_index, producer in enumerate(('heuristic', 'terminal_16x')):
            job_id = f'{producer}_e{index:03d}_r0'
            result_path = root / 'resolved' / f'{job_id}.json'
            result = read_json(result_path)
            if result['status'] != 'completed' or not result['natural_terminal'] or result['environment_seed'] != seed:
                raise ValueError(f'completed matching source required: {job_id}')
            replay_path = root / 'episodes' / job_id / 'decisions.jsonl'
            replay = [json.loads(line) for line in replay_path.read_text().splitlines()]
            observations = [row['observation'] for row in replay]
            target_phase = PHASES[(index + producer_index) % len(PHASES)]
            target_kind = KINDS[(2 * index + producer_index) % len(KINDS)]
            row_index, obs = select_state(observations, target_phase, target_kind)
            validate_public(obs)
            states.append({'id': f'e{index:03d}_{producer}', 'environment_index': index,
                           'environment_seed': seed, 'producer': producer, 'source_job': job_id,
                           'source_row': row_index, 'phase': phase(obs), 'target_phase': target_phase,
                           'target_kind': target_kind,
                           'target_kind_available': target_kind in {a['type'] for a in obs['legal_actions']},
                           'observation_sha256': object_hash(obs), 'observation': obs})
            sources.append({'job': job_id, 'replay_sha256': digest(replay_path),
                            'result_sha256': digest(result_path),
                            'terminal_metrics': board_metrics(observations[-1])})
    bank = {'protocol': 'public_state_bank_v123', 'split': 'development',
            'split_sha256': manifest['sha256'], 'source_root': str(root.resolve()),
            'source_id': allocation['source_id'], 'environments': environments,
            'selection': 'fixed environment prefix, two producers, legality-first then phase then stable hash; no score filter',
            'phase_counts': dict(Counter(s['phase'] for s in states)),
            'target_kind_counts': dict(Counter(s['target_kind'] for s in states)),
            'states': states, 'sources': sources}
    bank['sha256'] = object_hash(bank)
    return bank


def validate_bank(bank, manifest):
    validate_manifest(manifest)
    if bank['sha256'] != object_hash({k: v for k, v in bank.items() if k != 'sha256'}):
        raise ValueError('bank digest mismatch')
    if bank['split'] not in ('development', 'train_calibration') or bank['split_sha256'] != manifest['sha256']:
        raise ValueError('bank split mismatch; no confirmation/test')
    seeds = manifest['sets']['train' if bank['split'] == 'train_calibration' else 'development']
    if len({s['id'] for s in bank['states']}) != len(bank['states']):
        raise ValueError('duplicate state id')
    for state in bank['states']:
        index = state['environment_index']
        if type(index) is not int or not 0 <= index < bank['environments'] or state['environment_seed'] != seeds[index]:
            raise ValueError('state not in registered split prefix')
        if not state['id'].replace('_', '').isalnum():
            raise ValueError('unsafe state id')
        validate_public(state['observation'])
        if object_hash(state['observation']) != state['observation_sha256']:
            raise ValueError('state digest mismatch')
    if {s['environment_index'] for s in bank['states']} != set(range(bank['environments'])):
        raise ValueError('missing source environment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--splits', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--environments', type=int, default=32)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError('new bank path required')
    bank = build_bank(args.source, read_json(args.splits), args.environments)
    validate_bank(bank, read_json(args.splits))
    write_json(args.out, bank)
    print(json.dumps({k: bank[k] for k in ('sha256', 'environments', 'phase_counts', 'target_kind_counts')}))


if __name__ == '__main__':
    main()
