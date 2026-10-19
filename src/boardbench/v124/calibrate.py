"""Small training-only full-game cost/legality gate, not development selection."""
import argparse
from pathlib import Path
import time

from boardbench.artifacts.store import write_json
from . import campaign
from boardbench.v122.campaign import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--cores', type=int, default=8)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    arms = []
    for p in ('original', 'space', 'habitat'):
        arms.append({'id': f'greedy_{p}', 'method': 'heuristic', 'multiplier': 1,
            'timeout_seconds': 60, 'params': {'mode': 'greedy', 'continuation': p}})
        arms.append({'id': f'timed_{p}_30s', 'method': 'terminal', 'multiplier': 1,
            'timeout_seconds': 90, 'params': {'mode': 'timed', 'continuation': p, 'seconds': 30}})
    arms.append({'id': 'original_fixed4', 'method': 'terminal', 'multiplier': 1,
        'timeout_seconds': 240, 'params': {'mode': 'fixed', 'continuation': 'original'}})
    plan = {'split': 'train_calibration', 'environment_offset': 0, 'panels': [1, 2],
        'split_manifest': 'outputs/v121/factorial_f64_20261004_c/plan/splits.json',
        'search_seeds': [12420261020], 'cpu_cores': args.cores,
        'deadline_unix': time.time() + 600, 'arms': arms}
    path = root.with_suffix('.plan.json')
    if root.exists() or path.exists():
        raise ValueError('fresh calibration root required')
    write_json(path, plan)
    launch(path, root, args.repo, backend=campaign)


if __name__ == '__main__':
    main()
