"""Six bounded calibration games; execution/depth evidence, never strength ranking."""
import argparse
import os
from pathlib import Path
import shutil
import time

from boardbench.artifacts.store import digest,read_json,write_json,source_files,source_id
from boardbench.v122.campaign import physical_cpus
from boardbench.v148.search_campaign import Runner


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,default=20)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];out=Path(a.out).resolve();cpus=physical_cpus()
    if (os.uname().nodename!='b35.cluster.helixon.com' or not 0<a.seconds<=60
            or len(cpus)//2<7 or os.getloadavg()[0]>len(cpus)*.5):
        raise ValueError('idle b35,7 CPUs<=half, bounded calibration only')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('jobs','logs','episodes'):(out/name).mkdir()
    for name in source_files(repo):
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    c=read_json(repo/'configs/v147_b35_search.json')
    student=repo/'outputs/v147/explore_20261023_a/learning_b/rounds/r2/continue/best_policy.pt'
    c.update(repo=str(repo),student=str(student),student_sha256=digest(student),seconds=a.seconds,
             deadline_unix=time.time()+180,job_timeout_seconds=120,search_seed=148,
             previous_run='V147 completed; bounded V148 execution check')
    write_json(out/'config.json',c);write_json(out/'launch.json',dict(source_id=source_id(repo),
        config_sha256=digest(out/'config.json'),cpus=cpus[:7]))
    # Use the frozen package for the children and this exact checked source for
    # orchestration; no source edits are injected into children after launch.
    r=Runner(out)
    try:
        jobs=r.jobs(['search_habitat','search_mcts','search_mcts_wide'],[148900005],'gate')
        for job in jobs:job['split']='calibration'
        rows=r.run(jobs,'calibration_only')
        write_json(out/'summary.json',dict(passed=True,games=rows,source_id=r.launch['source_id'],
            note='One environment/two seats, short execution/depth check ONLY; no strength selection.'))
        r.phase='completed'
        print([dict(method=q['method'],seat=q['seat'],seconds=q['decision_seconds'],
                    depth=q['max_tree_depth'],simulations=q['completed_rounds']) for q in rows],flush=True)
    finally:r.close()


if __name__=='__main__':main()
