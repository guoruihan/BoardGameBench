import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from boardbench.artifacts.store import read_json, write_json
from boardbench.v121.registry import make_splits
from boardbench.v122.campaign import physical_cpus
from boardbench.v12.budget import kill_tree

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/expand_v124.py'
spec = importlib.util.spec_from_file_location('expand_v124', SCRIPT)
expand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expand)


def test_disjoint_partition_excludes_partially_started_environments():
    jobs = [{'id': f'e{e}_{a}', 'environment_index': e} for e in range(3) for a in range(2)]
    pending, fresh = expand.safe_partition(jobs, {'e0_0', 'e0_1', 'e1_0'})
    assert [j['id'] for j in pending] == ['e1_1', 'e2_0', 'e2_1']
    assert fresh == [2]
    with pytest.raises(ValueError):
        expand.safe_partition(jobs, {'unknown'})
    with pytest.raises(ValueError):
        expand.safe_partition(jobs + jobs[:1], set())


def test_pid_identity_cannot_signal_reused_or_unowned_process():
    identity = expand.identity(os.getpid())
    assert expand.live(identity)
    with pytest.raises(RuntimeError, match='identity'):
        expand.send({**identity, 'start': '-1'}, signal.SIGCONT)
    with pytest.raises(RuntimeError, match='identity'):
        expand.send({**identity, 'uid': -1}, signal.SIGCONT)


@pytest.mark.parametrize('stall', [False, True])
def test_real_drain_transfer_no_duplicate_and_independent_guard(tmp_path, stall):
    cpus = physical_cpus()
    if len(cpus) < 4 or os.getloadavg()[0] > len(cpus) * .5:
        pytest.skip('needs idle CPU host')
    repo = tmp_path / 'repo'
    repo.mkdir()
    original = SCRIPT.parents[1]
    shutil.copytree(original / 'src', repo / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(original / 'pyproject.toml', repo / 'pyproject.toml')
    shutil.copy2(SCRIPT, repo / 'expand.py')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '--allow-empty', '-qm', 'test'], check=True)
    write_json(repo / 'splits.json', make_splits(train_count=8))
    raw = {'split': 'train_calibration', 'split_manifest': 'splits.json', 'panels': [1, 2], 'search_seeds': [11],
        'cpu_cores': 2, 'deadline_unix': time.time() + (40 if stall else 65), 'arms': [
            {'id': p, 'method': 'terminal', 'multiplier': 1, 'timeout_seconds': 25,
             'params': {'mode': 'timed', 'continuation': p, 'seconds': 8}}
            for p in ['original', 'space', 'habitat']]}
    write_json(repo / 'plan.json', raw)
    parent, root = repo / 'outputs/v124/parent', repo / 'outputs/v124/expanded'
    env = {**os.environ, 'PYTHONPATH': str(repo / 'src'), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    old_log, new_log = (repo / 'old.log').open('w'), (repo / 'new.log').open('w')
    old = subprocess.Popen([sys.executable, '-m', 'boardbench.v124.campaign', 'launch', '--root', str(parent),
        '--repo', str(repo), '--plan', str(repo / 'plan.json')], cwd=repo, env=env, stdout=old_log, stderr=subprocess.STDOUT)
    new = None
    owned = []
    try:
        until = time.monotonic() + 12
        while not (parent / 'status.json').exists() and time.monotonic() < until:
            time.sleep(.05)
        status = read_json(parent / 'status.json')
        owned.extend(r['pid'] for r in status['active'])
        new = subprocess.Popen([sys.executable, str(repo / 'expand.py'), 'launch', '--parent', str(parent),
            '--root', str(root), '--reserve-physical', str(len(cpus) - 4)], cwd=repo, env=env,
            stdout=new_log, stderr=subprocess.STDOUT)
        until = time.monotonic() + 12
        while not (root / 'status.json').exists() and time.monotonic() < until:
            if new.poll() is not None:
                break
            time.sleep(.05)
        assert (root / 'status.json').exists(), (repo / 'new.log').read_text()
        new_status = read_json(root / 'status.json')
        owned.extend(r['pid'] for r in new_status['active'])
        handoff = read_json(root / 'handoff.json')
        assert expand.live(handoff['workers'][0])
        assert expand.same_process(handoff['controller'])['state'] == 'T'
        if stall:
            os.kill(new_status['pid'], signal.SIGSTOP)
        new.wait(timeout=75)
        old.wait(timeout=15)
        assert new.returncode == 0, (repo / 'new.log').read_text()
        assert (parent / 'source/src/boardbench/v124/search.py').read_bytes() == (repo / 'src/boardbench/v124/search.py').read_bytes()
        old_ids = {p.name for p in (parent / 'jobs').glob('*.json')}
        new_ids = {p.name for p in (root / 'jobs').glob('*.json')}
        assert not old_ids & new_ids
        assert old_ids == {i + '.json' for i in handoff['started_ids']}
        s = read_json(root / 'summary.json')
        assert read_json(parent / 'resolved' / next(iter(old_ids)))['status'] == 'completed'
        assert not read_json(root / 'watchdog_result.json')['old_controller_still_live']
        if stall:
            assert read_json(root / 'watchdog_result.json')['forced']
        else:
            assert len(old_ids | new_ids) == 6
            assert all(c['failed_or_censored'] == 0 for c in s['cells'].values())
            assert (root / 'parent_finalized.json').exists()
            assert len(s['resource_charge']['intervals']) == 2
            assert s['execution_batches']['fresh_environment_indices'] == [1]
    finally:
        for p in (new, old):
            if p is not None and p.poll() is None:
                kill_tree(p.pid)
                p.wait(timeout=3)
        old_log.close(); new_log.close()
    for pid in owned:
        ref = expand.identity(pid)
        assert ref is None or ref['state'] == 'Z'
