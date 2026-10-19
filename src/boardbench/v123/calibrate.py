"""Generate train-only public-state calibration inputs; never select method weights."""
import argparse
from pathlib import Path
import time

from boardbench.artifacts.store import read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.solvers.board import ranked_actions
from .bank import object_hash, select_state, validate_bank


def build(manifest):
    states = []
    for index, seed in enumerate(manifest['sets']['train'][:2]):
        engine = Harmonies()
        obs, _ = engine.reset(seed)
        replay = []
        while not obs['terminated']:
            replay.append(obs)
            obs = engine.step(ranked_actions(obs)[0][1]).observation
        phase = ('middle', 'late')[index]
        row, obs = select_state(replay, phase, 'place_token')
        states.append({'id': f'train_e{index}', 'environment_index': index, 'environment_seed': seed,
                       'producer': 'original', 'phase': phase, 'source_row': row,
                       'observation': obs, 'observation_sha256': object_hash(obs)})
    bank = {'protocol': 'public_state_bank_v123', 'split': 'train_calibration',
            'split_sha256': manifest['sha256'], 'environments': 2, 'states': states,
            'selection': 'first two train seeds; fixed middle/late token-action states, no outcome selection'}
    bank['sha256'] = object_hash(bank)
    validate_bank(bank, manifest)
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise ValueError('new calibration bank path required')
    write_json(path, build(read_json(args.splits)))
    write_json(path.with_suffix('.plan.json'), {
        'state_bank': str(path), 'split_manifest': args.splits, 'cpu_cores': 2,
        'queue_until_unix': time.time() + 120, 'run_seconds': 600, 'job_timeout_seconds': 280,
        'panels': [1, 2], 'width': 6, 'seed': 12320261019,
        'discovery_samples': 8, 'confirmation_samples': 64})


if __name__ == '__main__':
    main()
