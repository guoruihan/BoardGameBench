"""Restore moved policies using copied checkpoint code in an isolated process.

The temporary venv reuses installed numerical dependencies, not workspace code.
No model/data download. Editable build backend may be installed from PyPI mirror.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from boardbench.artifacts.store import file_hashes, read_json, write_json


def trajectory(path):
    result = []
    with Path(path).open() as stream:
        for line in stream:
            e = json.loads(line)
            if e['kind'] == 'transition' and e['episode'] == 1:
                result.append((e['action'], e['result']['observation']))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--artifacts', default='outputs/v1')
    args = p.parse_args()
    root, out = Path(args.artifacts).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {}
    env = dict(os.environ, PYTHONNOUSERSITE='1', PIP_CONFIG_FILE='/dev/null',
               PIP_EXTRA_INDEX_URL='', PIP_DISABLE_PIP_VERSION_CHECK='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    env.pop('PYTHONPATH', None)
    with TemporaryDirectory(prefix='boardbench-v1-portable-') as temporary:
        temp = Path(temporary)
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(temp/'venv')], check=True, env=env)
        python = temp/'venv/bin/python'
        first = next(root.glob('micro_tiles/policies/rl_trained_*/export/checkpoints/ep_000000/source'))
        shutil.copytree(first, temp/'source')
        with (out/'install.log').open('x') as log:
            subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', '-e', str(temp/'source'),
                            '--index-url', 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'],
                           check=True, env=env, cwd=temp, stdout=log, stderr=subprocess.STDOUT)
        for game in ('micro_tiles', 'take_it_easy', 'harmonies'):
            policy = next((root/game/'policies').glob('rl_trained_*'))
            metadata = read_json(policy/'manifest.json')
            checkpoint = policy/metadata['checkpoint_path']
            before = file_hashes(checkpoint)
            moved = temp/game/'checkpoint'
            shutil.copytree(checkpoint, moved)
            evaluation_dir = out/game
            code = """
import json, pathlib, sys, torch, boardbench
from boardbench.runner.engine import evaluate
torch.set_num_threads(2)
assert pathlib.Path(boardbench.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
result=evaluate(sys.argv[2], {'episode_seeds':[300000], 'per_episode_limits':{'max_action_requests':256,'max_wall_seconds':120}}, sys.argv[3])
assert result['completed_episodes']==1 and result['episode_errors']==0, result
print(json.dumps({'source':boardbench.__file__,'status':result['status']}))
"""
            output = subprocess.check_output([str(python), '-I', '-c', code, str(temp/'source'), str(moved), str(evaluation_dir)],
                                             text=True, cwd=temp, env=env)
            assert trajectory(evaluation_dir/'events.jsonl') == trajectory(root/game/'test_pilot_b'/policy.name/'events.jsonl')
            assert file_hashes(checkpoint) == before == file_hashes(moved)
            report[game] = {'policy': policy.name, 'status': 'passed', 'new_process': json.loads(output),
                            'actions_and_observations_identical': True, 'parent_and_copy_unchanged': True}
        write_json(out/'report.json', {'status': 'passed', 'games': report,
                   'boundary': 'isolated copied code and moved checkpoints; installed Torch/NumPy reused by temporary venv'})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
