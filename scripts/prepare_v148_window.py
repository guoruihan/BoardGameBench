"""Materialize a NEW explicitly authorized window; never launch or renew old runs."""
import argparse
from pathlib import Path
import time

from boardbench.artifacts.store import read_json,write_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--hours',required=True,type=float);p.add_argument('--started-unix',required=True,type=float)
    p.add_argument('--cycles',type=int,choices=(1,2,3),default=2);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];out=Path(a.out).resolve()
    duration=a.hours*3600;deadline=a.started_unix+duration
    if not 0<duration<=86400 or not a.started_unix<=time.time()<deadline-1800:
        raise ValueError('explicit current authorized window with time remaining required')
    out.mkdir(parents=True,exist_ok=False)
    student=repo/'outputs/v147/explore_20261023_a/learning_b/rounds/r2/continue/best_policy.pt'
    for role,source in [('search','configs/v147_b35_search.json'),('learning','configs/v147_a226_learning.json')]:
        c=read_json(repo/source)
        for key in ('previous_learning_run','initial_expert_manifest','initial_expert_manifest_sha256'):
            c.pop(key,None)
        c.update(started_unix=a.started_unix,deadline_unix=deadline,authorized_wall_seconds=duration,
            cycles=a.cycles,seconds=900,student=str(student),student_sha256=digest(student),
            search_root=str(out/'search'),learning_root=str(out/'learning'),
            search_seed=14820261024,expanded_environments=16,expert_environments=32,
            previous_run='outputs/v147/explore_20261023_a',
            mcts_config=dict(max_depth=16,initial_width=6,widening=1.5,exploration=.6,
                             value_fraction=.75,min_value_depth=4))
        if role=='learning':
            c.update(initial_expert_manifests=[str(repo/'outputs/v146/duo_teacher_b35_20261023_a/expert_manifest.json'),
                str(repo/'outputs/v147/explore_20261023_a/search/expert_manifest.json')],
                fit_steps=1200,fit_evaluate_interval=200,development_environments=64,
                decision_environments=128,selection_environments=256,confirmation_environments=512)
        write_json(out/(role+'_config.json'),c)
    print(out,flush=True)


if __name__=='__main__':main()
