"""Restore all trained policies from moved checkpoints and a copied source venv."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from boardbench.artifacts.store import read_json, write_json, file_hashes


def trajectory(path):
    with Path(path).open() as stream:
        return [(e['action'],e['result']['observation']) for line in stream
                if (e:=json.loads(line))['kind']=='transition' and e['episode']==1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',default='outputs/v11')
    p.add_argument('--out',required=True)
    args=p.parse_args()
    root,out=Path(args.artifacts).resolve(),Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONNOUSERSITE='1',PIP_CONFIG_FILE='/dev/null',PIP_EXTRA_INDEX_URL='',
             PIP_DISABLE_PIP_VERSION_CHECK='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    env.pop('PYTHONPATH',None)
    report={}
    with TemporaryDirectory(prefix='boardbench-v11-portable-') as temp:
        temp=Path(temp)
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(temp/'venv')],check=True,env=env)
        python=temp/'venv/bin/python'
        first=next(root.glob('micro_tiles/policies/rl_trained_*/export/checkpoints/ep_000000/source'))
        shutil.copytree(first,temp/'source')
        with (out/'install.log').open('x') as log:
            subprocess.run([str(python),'-m','pip','install','--no-deps','-e',str(temp/'source'),
                            '--index-url','https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'],
                           check=True,cwd=temp,env=env,stdout=log,stderr=subprocess.STDOUT)
        for game in ('micro_tiles','take_it_easy','harmonies'):
            selection=read_json(root/game/'test/frozen_selection.json')
            results=read_json(root/game/'test/results.json')
            for candidate in selection['selected']:
                if candidate['method'] not in ('rl','imitation') or candidate.get('untrained'):
                    continue
                cid=candidate['id']
                policy=root/game/'policies'/cid
                manifest=read_json(policy/'manifest.json')
                checkpoint=policy/manifest['checkpoint_path']
                before=file_hashes(checkpoint)
                moved=temp/game/cid
                shutil.copytree(checkpoint,moved)
                target=out/game/cid
                seed=selection['splits']['test'][0]
                code='''
import pathlib, sys, torch
import boardbench
from boardbench.runner.engine import evaluate
torch.set_num_threads(2)
assert pathlib.Path(boardbench.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
r=evaluate(sys.argv[2],{'episode_seeds':[int(sys.argv[4])],'per_episode_limits':{'max_action_requests':256,'max_wall_seconds':120}},sys.argv[3])
assert r['completed_episodes']==1 and r['episode_errors']==0, r
'''
                subprocess.run([str(python),'-I','-c',code,str(temp/'source'),str(moved),str(target),str(seed)],
                               check=True,cwd=temp,env=env)
                original=root/game/'test'/results[cid]['raw_runs'][0]/'events.jsonl'
                assert trajectory(target/'events.jsonl')==trajectory(original)
                assert before==file_hashes(checkpoint)==file_hashes(moved)
                report[f'{game}/{cid}']={'status':'passed','actions_and_observations_identical':True,'parent_and_copy_unchanged':True}
                write_json(out/'report.json',{'status':'in_progress','policies':report})
                print(game,cid,'passed',flush=True)
    write_json(out/'report.json',{'status':'passed','policies':report,
                'boundary':'isolated copied source/new cwd/new process; installed Torch/NumPy reused, not blank-system dependencies'})


if __name__=='__main__':
    main()
