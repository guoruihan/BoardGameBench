from copy import deepcopy
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
from boardbench.v122.campaign import physical_cpus, incomplete
from boardbench.v124.campaign import prepare_plan, make_jobs
from boardbench.v124.report import publish, summarize


def plan():
    return prepare_plan({'split': 'development', 'environment_offset': 64, 'panels': [2, 4],
        'search_seeds': [11, 12, 13], 'arms': [
            {'id': p, 'method': 'terminal', 'multiplier': 1, 'timeout_seconds': 60,
             'params': {'mode': 'timed', 'continuation': p, 'seconds': 30}}
            for p in ['original', 'space', 'habitat']]}, make_splits(train_count=8))


def result(job, score=40, seconds=29):
    return {**job, 'status': 'completed', 'natural_terminal': True, 'score': score,
        'components': {'total': score, 'animals': 10}, 'simulation_steps': 100, 'decision_seconds': seconds,
        'terminal_metrics': {'terrain': score - 10, 'animals_placed': 2, 'completed_cards': 1, 'turns': 10, 'tokens': 30},
        'solver_budget_utilization': seconds / 30}


def test_plan_guards_and_offset():
    p = plan()
    manifest = make_splits(train_count=8)
    assert p['environment_seeds'] == manifest['sets']['development'][64:68]
    assert len(make_jobs(p)) == 36
    with pytest.raises(ValueError, match='no confirmation/test'):
        prepare_plan({**p, 'split': 'test'}, manifest)
    with pytest.raises(ValueError, match='exceeds'):
        prepare_plan({**p, 'environment_offset': 127}, manifest)
    bad = deepcopy(p)
    bad['arms'][0]['params']['seconds'] = float('nan')
    with pytest.raises(ValueError, match='timed'):
        prepare_plan(bad, manifest)


def test_paired_complete_panels_and_real_time_gate(tmp_path):
    p, jobs = plan(), make_jobs(plan())
    rows = [result(j, 40 + (10 if j['arm'] == 'habitat' else 0)) for j in jobs if j['environment_index'] < 2]
    s = publish(tmp_path, p, jobs, rows, {'phase': 'running'})
    assert s['groups']['timed_30s']['balanced_panel_size'] == 2
    comparison = s['paired']['habitat-vs-original']
    assert comparison['n'] == 2 and comparison['mean_gap'] == 10
    assert comparison['actual_mean_times_within_10_percent']
    assert summarize(p, jobs, rows[:-1])['groups']['timed_30s']['balanced_panel_size'] == 0
    for r in rows:
        if r['arm'] == 'habitat':
            r['decision_seconds'] = 10
    assert not summarize(p, jobs, rows)['paired']['habitat-vs-original']['actual_mean_times_within_10_percent']
    with pytest.raises(ValueError, match='duplicate'):
        summarize(p, jobs, rows + [rows[0]])
    bad = deepcopy(rows)
    bad[0]['params']['seconds'] = 1
    with pytest.raises(ValueError, match='parameters'):
        summarize(p, jobs, bad)
    fail = incomplete(jobs[0], tmp_path / 'missing', 'job_timeout')
    s = summarize(p, jobs, [fail])
    assert s['cells']['original']['failed_or_censored'] == 1
    assert s['groups']['timed_30s']['balanced_panel_size'] == 0


@pytest.mark.parametrize('kind', ['job_timeout', 'deadline', 'controller_stall'])
def test_v124_frozen_backend_real_cleanup(tmp_path, kind):
    if len(physical_cpus()) < 4 or os.getloadavg()[0] > len(physical_cpus()) * .5:
        pytest.skip('requires idle host with at least four physical CPUs')
    repo = tmp_path / 'repo'
    repo.mkdir()
    origin = Path(__file__).resolve().parents[1]
    shutil.copytree(origin / 'src', repo / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(origin / 'pyproject.toml', repo / 'pyproject.toml')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '--allow-empty', '-qm', 'test'], check=True)
    write_json(repo / 'splits.json', make_splits(train_count=8))
    raw = {'split': 'train_calibration', 'split_manifest': 'splits.json', 'panels': [1], 'search_seeds': [11],
        'cpu_cores': 2, 'deadline_unix': time.time() + 14, 'arms': [
            {'id': 'fixed', 'method': 'terminal', 'multiplier': 64,
             'params': {'mode': 'fixed', 'continuation': 'original'},
             'timeout_seconds': 2 if kind == 'job_timeout' else 120}]}
    write_json(repo / 'plan.json', raw)
    root = repo / 'outputs/v124/test'
    proc = subprocess.Popen([sys.executable, '-m', 'boardbench.v124.campaign', 'launch', '--root', str(root),
        '--plan', str(repo / 'plan.json'), '--repo', str(repo)], env={**os.environ, 'PYTHONPATH': str(repo / 'src')},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    owned = []
    try:
        if kind == 'controller_stall':
            until = time.monotonic() + 7
            while not (root / 'status.json').exists() and time.monotonic() < until:
                time.sleep(.05)
            status = read_json(root / 'status.json')
            owned = [r['pid'] for r in status['active']]
            os.kill(status['pid'], signal.SIGSTOP)
        stdout, stderr = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            from boardbench.v12.budget import kill_tree
            kill_tree(proc.pid)
            proc.wait(timeout=3)
    assert proc.returncode == 0, stdout + stderr
    assert (root / 'source/src/boardbench/v124/search.py').exists()
    outcome = read_json(root / 'watchdog_result.json')
    assert outcome['elapsed_seconds'] < 23
    assert read_json(root / 'status.json')['active_count'] == 0
    rows = list((root / 'resolved').glob('*.json'))
    assert len(rows) == 1
    assert read_json(rows[0])['status'] == ('watchdog_interrupted' if kind == 'controller_stall' else kind)
    assert (repo / 'outputs/v122' / f'{os.uname().nodename}.cpu.lock').exists()
    for pid in owned:
        stat = Path(f'/proc/{pid}/stat')
        assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
