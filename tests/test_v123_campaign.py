import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from boardbench.artifacts.store import read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.v121.registry import make_splits
from boardbench.v122.campaign import physical_cpus
from boardbench.v123.bank import object_hash


@pytest.mark.parametrize('kind', ('old_lock', 'job_timeout', 'deadline', 'controller_stall'))
def test_frozen_source_shared_lock_and_independent_cleanup(tmp_path, kind):
    if len(physical_cpus()) < 4 or os.getloadavg()[0] > len(physical_cpus()) * .5:
        pytest.skip('requires four physical cores and safe host load')
    repo = tmp_path / 'repo'
    original = Path(__file__).resolve().parents[1]
    shutil.copytree(original / 'src', repo / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(original / 'pyproject.toml', repo / 'pyproject.toml')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '--allow-empty', '-qm', 'test'], check=True)
    manifest = make_splits(train_count=8)
    obs, _ = Harmonies().reset(manifest['sets']['train'][0])
    bank = {'split': 'train_calibration', 'split_sha256': manifest['sha256'], 'environments': 1,
            'states': [{'id': 'e0', 'environment_index': 0, 'producer': 'test', 'phase': 'early',
                        'environment_seed': manifest['sets']['train'][0], 'observation': obs,
                        'observation_sha256': object_hash(obs)}]}
    bank['sha256'] = object_hash(bank)
    write_json(repo / 'bank.json', bank)
    write_json(repo / 'splits.json', manifest)
    plan = {'state_bank': 'bank.json', 'split_manifest': 'splits.json', 'cpu_cores': 2,
            'queue_until_unix': time.time() + (3 if kind == 'old_lock' else 30),
            'run_seconds': 14, 'job_timeout_seconds': 2 if kind == 'job_timeout' else 120,
            'panels': [1], 'width': 6, 'seed': 5, 'discovery_samples': 4096, 'confirmation_samples': 2}
    write_json(repo / 'plan.json', plan)
    lock_path = repo / 'outputs/v122' / f'{os.uname().nodename}.cpu.lock'
    lock_path.parent.mkdir(parents=True)
    with lock_path.open('a') as lock:
        if kind == 'old_lock':
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = repo / 'outputs/v123/test'
        env = {**os.environ, 'PYTHONPATH': str(repo / 'src')}
        proc = subprocess.Popen([sys.executable, '-m', 'boardbench.v123.campaign', 'launch',
                                 '--repo', str(repo), '--root', str(root), '--plan', str(repo / 'plan.json')],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        owned = []
        try:
            if kind == 'controller_stall':
                ready_by = time.monotonic() + 8
                status = {}
                while time.monotonic() < ready_by:
                    if (root / 'status.json').exists():
                        status = read_json(root / 'status.json')
                        if status['phase'] == 'running' and status['active_count']:
                            break
                    time.sleep(.05)
                assert status.get('active_count'), status
                owned = [v['pid'] for v in status['active']]
                os.kill(status['pid'], signal.SIGSTOP)
            stdout, stderr = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                from boardbench.v12.budget import kill_tree
                kill_tree(proc.pid)
                proc.wait(timeout=3)
        assert proc.returncode == 0, stdout + stderr
        status = read_json(root / 'status.json')
        assert status['active_count'] == 0
        assert (root / 'source/src/boardbench/v123/diagnostic.py').exists()
        if kind == 'old_lock':
            assert status['phase'] == 'queue_expired'
            assert not list((root / 'jobs').glob('*.json'))
            assert not read_json(root / 'queue_result.json')['acquired']
            return
        allocation, outcome = read_json(root / 'allocation.json'), read_json(root / 'watchdog_result.json')
        assert len(allocation['cpu_affinity']) == 2 and outcome['gpu_seconds'] == 0
        assert outcome['elapsed_seconds'] < 20
        assert outcome['cumulative_charged_cpu_core_seconds'] >= outcome['allocated_cpu_core_seconds']
        attempts = list((root / 'resolved').glob('*.json'))
        assert len(attempts) == 1
        assert read_json(attempts[0])['status'] == ('watchdog_interrupted' if kind == 'controller_stall' else kind)
        for pid in owned:
            stat = Path(f'/proc/{pid}/stat')
            assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
