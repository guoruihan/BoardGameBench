from copy import deepcopy
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
from boardbench.v122.campaign import incomplete, make_jobs, physical_cpus, prepare_plan
from boardbench.v122.report import environment_means, paired, publish, summarize


def plan():
    return prepare_plan({'split': 'development', 'panels': [2, 4], 'search_seeds': [11, 12, 13],
                         'arms': [{'id': 'heuristic', 'method': 'heuristic', 'multiplier': 1, 'timeout_seconds': 10},
                                  {'id': 'short_1x', 'method': 'short', 'multiplier': 1, 'timeout_seconds': 30}]},
                        make_splits(train_count=8))


def result(job, score=40):
    return {**job, 'status': 'completed', 'natural_terminal': True, 'score': score,
            'decision_seconds': 1., 'simulation_steps': 10, 'components': {'total': score, 'animals': 5}}


def test_only_declared_split_and_exact_job_matrix():
    value = plan()
    jobs = make_jobs(value)
    assert len(jobs) == 4 * (1 + 3)
    assert len({j['id'] for j in jobs}) == len(jobs)
    assert {j['environment_seed'] for j in jobs} == set(value['environment_seeds'])
    invalid = {**value, 'split': 'test'}
    with pytest.raises(ValueError, match='no confirmation/test'):
        prepare_plan(invalid, make_splits(train_count=8))
    invalid = {**value, 'panels': [4, 2]}
    with pytest.raises(ValueError, match='panel'):
        prepare_plan(invalid, make_splits(train_count=8))


def test_repetitions_are_not_pseudoreplicated_and_incomplete_panel_is_not_promoted(tmp_path):
    value, jobs = plan(), make_jobs(plan())
    rows = [result(j, score=40 + j['environment_index'] + (10 if j['method'] == 'short' else 0))
            for j in jobs if j['environment_index'] < 2]
    summary = publish(tmp_path, value, jobs, rows, {'phase': 'running'})
    assert summary['balanced_panel_size'] == 2
    assert summary['cells']['short_1x']['attempted'] == 6
    assert summary['cells']['short_1x']['complete_environments'] == 2
    comparison = summary['paired']['short_1x-vs-heuristic']
    assert comparison['n'] == 2 and comparison['mean_gap'] == 10
    assert comparison['normal_95_ci'] == [10, 10]
    assert (tmp_path / 'curve.svg').exists() and (tmp_path / 'REPORT.md').exists()
    # One missing search repetition invalidates the whole predeclared panel.
    broken = summarize(value, jobs, rows[:-1])
    assert broken['balanced_panel_size'] == 0
    assert broken['paired']['short_1x-vs-heuristic']['n'] == 0


def test_failures_cannot_disappear_or_become_terminal_scores(tmp_path):
    value, jobs = plan(), make_jobs(plan())
    fail = incomplete(jobs[0], tmp_path / 'missing', 'job_timeout')
    summary = summarize(value, jobs, [fail])
    cell = summary['cells']['heuristic']
    assert cell['attempted'] == 1 and cell['failed_or_censored'] == 1 and cell['completed'] == 0
    assert cell['completed_only_mean'] is None and cell['attempted_zero_for_incomplete_mean'] == 0
    assert fail['score'] is None and not fail['natural_terminal']
    with pytest.raises(ValueError, match='duplicate'):
        summarize(value, jobs, [fail, fail])


@pytest.mark.parametrize('kind', ('job_timeout', 'deadline', 'controller_stall'))
def test_independent_deadline_kills_only_owned_workers_and_freezes_source(tmp_path, kind):
    # Real subprocess lifecycle, source identity and output reconciliation. This
    # intentionally censors an expensive terminal64x attempt, not a fake timeout.
    if len(physical_cpus()) < 4:
        pytest.skip('requires four physical CPUs for half-machine guard')
    if os.getloadavg()[0] > len(physical_cpus()) * .5:
        pytest.skip('shared host is busy; run real lifecycle checks on the reserved node')
    repo = tmp_path / 'repo'
    repo.mkdir()
    original = Path(__file__).resolve().parents[1]
    shutil.copytree(original / 'src', repo / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(original / 'pyproject.toml', repo / 'pyproject.toml')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '--allow-empty', '-qm', 'test'], check=True)
    write_json(repo / 'splits.json', make_splits(train_count=8))
    value = {'split': 'train_calibration', 'split_manifest': 'splits.json', 'panels': [1],
             'search_seeds': [11], 'cpu_cores': 2, 'deadline_unix': time.time() + 14,
             'arms': [{'id': 'terminal_64x', 'method': 'terminal', 'multiplier': 64,
                       'timeout_seconds': 2 if kind == 'job_timeout' else 120}]}
    write_json(repo / 'plan.json', value)
    root = repo / 'outputs/v122/test'
    env = {**os.environ, 'PYTHONPATH': str(repo / 'src')}
    proc = subprocess.Popen([sys.executable, '-m', 'boardbench.v122.campaign', 'launch',
                             '--repo', str(repo), '--root', str(root), '--plan', str(repo / 'plan.json')],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    owned_pids = []
    try:
        if kind == 'controller_stall':
            ready_by = time.monotonic() + 7
            while not (root / 'status.json').exists() and time.monotonic() < ready_by:
                time.sleep(.05)
            status = read_json(root / 'status.json')
            owned_pids = [row['pid'] for row in status['active']]
            os.kill(status['pid'], signal.SIGSTOP)
        stdout, stderr = proc.communicate(timeout=25)
    finally:
        if proc.poll() is None:
            from boardbench.v12.budget import kill_tree
            kill_tree(proc.pid)
            proc.wait(timeout=3)
    assert proc.returncode == 0, stdout + stderr
    allocation, outcome = read_json(root / 'allocation.json'), read_json(root / 'watchdog_result.json')
    assert len(allocation['cpu_affinity']) == 2 and allocation['gpu_seconds'] == 0
    assert outcome['elapsed_seconds'] < 20
    attempts = list((root / 'resolved').glob('*.json'))
    assert len(attempts) == 1
    expected = 'watchdog_interrupted' if kind == 'controller_stall' else kind
    assert read_json(attempts[0])['status'] == expected
    assert read_json(root / 'status.json')['active_count'] == 0
    assert (root / 'source/src/boardbench/v122/search.py').exists()
    for pid in owned_pids:
        # A killed orphan may briefly remain a zombie until host init reaps it.
        stat = Path(f'/proc/{pid}/stat')
        assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
