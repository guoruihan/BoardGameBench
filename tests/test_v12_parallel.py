import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from boardbench.artifacts.store import read_json, write_json


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / 'scripts/parallel_v12.py'
    spec = importlib.util.spec_from_file_location('parallel_v12_test_driver', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_gpu_accounting_sums_overlapping_reservations(driver):
    assert driver.gpu_seconds([{'started': 1., 'ended': 11.}, {'started': 6., 'ended': 16.}], 20.) == 20.
    assert driver.gpu_seconds([{'started': 1., 'ended': 11.}, {'started': 6.}], 20.) == 24.


def test_real_parallel_cpu_jobs_reuse_and_evaluation_barrier(driver, tmp_path, monkeypatch):
    monkeypatch.delenv('BOARDBENCH_STARTED_MONOTONIC', raising=False)
    exp = driver.ParallelExperiment.__new__(driver.ParallelExperiment)
    exp.root, exp.repo, exp.prior = tmp_path / 'new', tmp_path, tmp_path / 'old'
    exp.root.mkdir(); exp.prior.mkdir()
    exp.gpus, exp.reused = [20, 21], []
    exp.teacher = tmp_path / 'unused_teacher.json'
    exp.splits = {'train': [10, 11], 'validation': [21, 22], 'test': [31, 32]}
    exp.plan = {'bc_ppo_learning_rate': .0001, 'workers': 1}
    exp.reserve = 1
    exp.accounting = driver.Accounting(exp.root)
    exp.ledger = driver.ParallelLedger(exp.root, 60, exp.accounting, 123.)
    monkeypatch.setattr(driver, 'idle_gpu', lambda index: f'fake_gpu_{index}')
    real_popen = subprocess.Popen
    def cpu_popen(command, **kwargs):
        command = list(command); command[command.index('--device') + 1] = 'cpu'
        return real_popen(command, **kwargs)
    monkeypatch.setattr(driver.subprocess, 'Popen', cpu_popen)
    jobs = [('ppo', seed, f'ppo_{seed}_stage0', .01, None, None) for seed in (1, 2)]
    results = exp.train_group(jobs)
    assert set(results) == {'ppo_1_stage0', 'ppo_2_stage0'}
    records = exp.accounting.snapshot()
    assert len(records) == 2 and {r['gpu'] for r in records} == {'fake_gpu_20', 'fake_gpu_21'}
    assert max(r['started'] for r in records) < min(r['ended'] for r in records)
    assert all(r['returncode'] == 0 for r in records)
    status = read_json(exp.root / 'status.json')
    assert status['cumulative_charged_gpu_seconds'] == pytest.approx(123. + driver.gpu_seconds(records))
    exp.prior = exp.root
    monkeypatch.setattr(driver.subprocess, 'Popen', lambda *a, **k: pytest.fail('completed job was rerun'))
    reused = exp.train_group(jobs)
    assert reused == results and len(exp.accounting.snapshot()) == 2
    record = exp.accounting.start('still_training', 'fake_gpu_20')
    with pytest.raises(RuntimeError, match='compete'):
        exp.evaluate({}, 'dev', [21, 22], 3)
    exp.accounting.end(record, 0)
    cfg = exp.prior / 'configs/ppo_1_stage0.json'
    value = read_json(cfg); value['epochs'] += 1; write_json(cfg, value)
    with pytest.raises(ValueError, match='configuration/source'):
        exp.train_group(jobs)


@pytest.mark.parametrize('child_valid', [True, False])
def test_handoff_waits_for_child_or_resumes_controller(driver, tmp_path, monkeypatch, child_valid):
    prior = tmp_path / 'old'; prior.mkdir()
    write_json(prior / 'allocation.json', {'host': 'testhost', 'watchdog_pid': 10})
    write_json(prior / 'status.json', {'pid': 20})
    parent_state = ['R']; child_polls = [0]; events = []
    def process(pid):
        if pid == 10:
            return {'pid': 10, 'start': 'a', 'ppid': 1, 'state': 'R', 'argv': ['boardbench.v12.launch']}
        if pid == 20:
            return {'pid': 20, 'start': 'b', 'ppid': 10, 'state': parent_state[0],
                    'argv': ['boardbench.v12.animal_experiment', str(prior)]}
        child_polls[0] += 1
        return {'pid': 30, 'start': 'c', 'ppid': 20, 'state': 'R' if child_polls[0] < 3 else 'Z',
                'argv': ['boardbench.v12.training', '--out', str(prior / 'training/job')]}
    def send(identity, sig):
        events.append((identity['pid'], sig))
        if sig == driver.signal.SIGSTOP:
            parent_state[0] = 'T'
        if sig == driver.signal.SIGTERM:
            assert child_polls[0] >= 3  # Child was allowed to finish, never signaled.
            write_json(prior / 'watchdog_result.json', {'finished': True})
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **k: '30' if str(self) == '/proc/20/task/20/children' else original_read(self, *a, **k))
    monkeypatch.setattr(driver.os, 'uname', lambda: SimpleNamespace(nodename='testhost'))
    monkeypatch.setattr(driver, 'process', process)
    monkeypatch.setattr(driver, 'signal_checked', send)
    monkeypatch.setattr(driver.time, 'sleep', lambda _: None)
    def verify(argv):
        if not child_valid:
            raise ValueError('incomplete child')
        return argv[-1]
    monkeypatch.setattr(driver, 'verify_phase_output', verify)
    if child_valid:
        driver.handoff(prior, driver.time.time() + 4000)
        assert events == [(20, driver.signal.SIGSTOP), (10, driver.signal.SIGTERM)]
        assert not read_json(prior / 'parallel_handoff.json')['active_child_was_interrupted']
    else:
        with pytest.raises(ValueError, match='incomplete'):
            driver.handoff(prior, driver.time.time() + 4000)
        assert events == [(20, driver.signal.SIGSTOP), (20, driver.signal.SIGCONT)]


def test_signal_guard_rejects_reused_pid(driver, monkeypatch):
    monkeypatch.setattr(driver, 'process', lambda pid: {'start': 'new'})
    monkeypatch.setattr(driver.os, 'kill', lambda *args: pytest.fail('must not signal reused PID'))
    with pytest.raises(RuntimeError, match='identity changed'):
        driver.signal_checked({'pid': 42, 'start': 'old'}, driver.signal.SIGTERM)


def test_stopped_prior_is_reused_without_signals_or_reopening_final_test(driver, tmp_path, monkeypatch):
    write_json(tmp_path / 'allocation.json', {'watchdog_pid': 10})
    write_json(tmp_path / 'watchdog_result.json', {'returncode': 1})
    monkeypatch.setattr(driver, 'process', lambda pid: None)
    monkeypatch.setattr(driver, 'signal_checked', lambda *args: pytest.fail('stopped run must not be signaled'))
    assert driver.prepare_prior(tmp_path, 10000) == 'reuse_stopped_run'
    write_json(tmp_path / 'frozen_selection.json', {})
    with pytest.raises(ValueError, match='final evaluation'):
        driver.prepare_prior(tmp_path, 10000)
