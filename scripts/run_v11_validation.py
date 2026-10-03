"""Validate every predeclared repetition; final test remains untouched."""
import argparse
from pathlib import Path
from boardbench.artifacts.store import read_json, source_id
from boardbench.benchmark import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True)
    p.add_argument('--source-id',required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if source_id()!=args.source_id:
        raise ValueError('runtime source drift before validation')
    plan=read_json('outputs/v11/experiment_plan/plan.json')
    root=Path('outputs/v11')/args.task
    runs=[root/f'ppo_{seed}' for seed in plan['training_seeds']]
    if plan['games'][args.task]['imitation_episodes']:
        runs += [root/f'imitation_{seed}' for seed in plan['training_seeds']]
    for run in runs:
        summary=read_json(run/'summary.json')
        if summary['status'] not in ('completed','wall_budget'):
            raise ValueError(f'run unfinished: {run}')
    validate(args.task,runs[0],args.out,runs[1:],plan['search_candidates'],True,args.workers)


if __name__=='__main__':
    main()
