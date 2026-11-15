from dataclasses import replace
import json

import pytest

from boardbench.families.pilot_run import (
    Checkpoints, DevelopmentResult, PilotConfig, PilotRunner, blank_submission,
    default_queue, seed_for, tree_hash, validate_submission,
)


def config():
    return PilotConfig('frozen-rl', 'a' * 64, 'env-revision', 'codex-test', 9999999999)


def test_queue_pairs_and_domains():
    queue = default_queue(config())
    assert len(queue) == 40
    for profile in ('L10', 'L20', 'L40'):
        matches = [m for m in queue if m.profile == profile and not m.control]
        assert len(matches) == 10
        assert {m.seat for m in matches} == {0, 1}
        assert len({m.seed for m in matches}) == 5
    assert [(m.seed, m.seat) for m in queue[:10]] == [(m.seed, m.seat) for m in queue[-10:]]
    domains = [{seed_for('test', domain, i) for i in range(25)} for domain in ('acceptance', 'development', 'calibration', 'formal')]
    assert len(set.union(*domains)) == 100


def test_checkpoints_are_explicit_format_only_and_freeze_last(tmp_path):
    submission = tmp_path / 'submission'
    blank_submission(submission)
    store = Checkpoints(tmp_path / 'trusted')
    assert store.freeze() is None
    first = store.checkpoint(submission)
    (submission / 'START_HERE.md').write_text('new notes')
    last = store.checkpoint(submission)
    assert last != first
    (submission / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError):
        store.checkpoint(submission)
    assert store.freeze().name == last
    with pytest.raises(RuntimeError):
        store.checkpoint(submission)


@pytest.mark.parametrize('kind', ['symlink', 'escape', 'missing_program', 'private_dir'])
def test_reject_unsafe_submission(tmp_path, kind):
    submission = tmp_path / 'submission'
    blank_submission(submission)
    manifest = json.loads((submission / 'manifest.json').read_text())
    if kind == 'symlink':
        (submission / 'escape').symlink_to('/etc/passwd')
    elif kind == 'escape':
        manifest.update(entry_mode='program', program_entry='../outside.py')
    elif kind == 'missing_program':
        manifest.update(entry_mode='program')
    else:
        (submission / '.codex').mkdir()
    (submission / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_submission(submission)


class FakeAdapter:
    def __init__(self, checkpoint=True, finish=True, failure=False):
        self.checkpoint, self.finish, self.failure = checkpoint, finish, failure
        self.dev_calls, self.matches = [], []

    def run_development(self, context, resume_session):
        self.dev_calls.append((context, resume_session))
        if self.checkpoint and len(self.dev_calls) == 1:
            (context.workspace / 'submission' / 'START_HERE.md').write_text('developed notes')
            context.checkpoints.checkpoint(context.workspace / 'submission')
        return DevelopmentResult('explicit-session', finished=self.finish)

    def run_match(self, context, match):
        self.matches.append((context, match))
        assert not (context.workspace / 'prior_match').exists()
        assert not list(context.home.iterdir())
        notes = (context.workspace / 'submission' / 'START_HERE.md').read_text()
        assert ('developed notes' in notes) != match.control
        assert 'seed' not in (context.workspace / 'task_config.json').read_text()
        (context.workspace / 'prior_match').write_text('local memory')
        (context.home / 'session').write_text('local session')
        (context.workspace / 'submission' / 'START_HERE.md').write_text('mutated')
        return {'status': 'infrastructure_failure' if self.failure else 'natural_completion'}


def test_complete_queue_is_independent_and_resume_does_not_rerun(tmp_path):
    adapter = FakeAdapter()
    runner = PilotRunner(tmp_path, config(), adapter)
    assert runner.run()['status'] == 'complete'
    assert len(adapter.matches) == 40
    assert len({ctx.home for ctx, _ in adapter.matches}) == 40
    records = [json.loads(line) for line in (tmp_path / 'results.jsonl').read_text().splitlines()]
    assert len({r['submission_sha256'] for r in records if not r['control']}) == 1
    assert records[-1]['submission_sha256'] != records[0]['submission_sha256']
    assert PilotRunner(tmp_path, config(), adapter).run()['status'] == 'complete'
    assert len(adapter.matches) == 40


def test_two_empty_resumes_use_same_ledger_and_session(tmp_path):
    adapter = FakeAdapter(finish=False)
    PilotRunner(tmp_path, config(), adapter).run()
    assert len(adapter.dev_calls) == 3
    assert [s for _, s in adapter.dev_calls] == [None, 'explicit-session', 'explicit-session']
    assert len({c.ledger_path for c, _ in adapter.dev_calls}) == 1
    assert len({c.deadline_unix for c, _ in adapter.dev_calls}) == 1
    assert 'early_exit' in (tmp_path / 'events.jsonl').read_text()


def test_missing_submission_fails_without_testing(tmp_path):
    adapter = FakeAdapter(checkpoint=False)
    assert PilotRunner(tmp_path, config(), adapter).run()['status'] == 'no_valid_submission'
    assert not adapter.matches


def test_infrastructure_failure_stops_and_never_retries(tmp_path):
    adapter = FakeAdapter(failure=True)
    assert PilotRunner(tmp_path, config(), adapter).run()['status'] == 'infrastructure_failure'
    PilotRunner(tmp_path, config(), adapter).run()
    assert 1 <= len(adapter.matches) <= 4


def test_interrupted_phase_freezes_last_and_never_restarts(tmp_path):
    runner = PilotRunner(tmp_path, config(), FakeAdapter())
    source = tmp_path / 'submission'
    blank_submission(source)
    expected = runner.checkpoints.checkpoint(source)
    runner.state['development'] = 'running'
    state = runner.run()
    assert state['status'] == 'infrastructure_failure'
    assert state['frozen_submission'].endswith(expected)


def test_deadline_freezes_without_free_model_call(tmp_path):
    adapter = FakeAdapter()
    runner = PilotRunner(tmp_path, replace(config(), deadline_unix=10), adapter, clock=lambda: 11)
    assert runner.run()['status'] == 'deadline'
    assert not adapter.dev_calls


def test_native_development_checkpoint_control_and_resume_contract(tmp_path, monkeypatch):
    import http.client
    import socket
    import sys
    import types
    from dataclasses import dataclass
    from pathlib import Path
    from boardbench.families.pilot_run import NativePilotAdapter, DEVELOPMENT_BUDGET

    @dataclass
    class Result:
        reason: str
        stdout_path: str

    class GT:
        root = tmp_path / 'gt'
        def report(self, rid):
            return {'status': 'active', 'games': 1, 'steps': 5, 'version': 5}
        def update_budget(self, rid, budget):
            self.budget = budget
        def stop(self, rid, reason):
            self.stopped = reason

    class Supervisor:
        def run(self, spec):
            assert spec.argv[1:3] == ['exec', '--json']
            assert '--dangerously-bypass-approvals-and-sandbox' in spec.argv
            assert spec.env['PILOT_MODEL_TOKEN'] == 'account-token'
            assert spec.env['PYTHONPATH'] == '/bench'
            assert spec.cpu_ids == tuple(range(8))
            assert spec.tcp_bridges == {18080: 'gateway.sock'}
            (spec.work_dir / 'submission' / 'START_HERE.md').write_text('submitted notes')
            class Unix(http.client.HTTPConnection):
                def connect(self):
                    self.sock = socket.socket(socket.AF_UNIX)
                    self.sock.connect(str(spec.sockets['control.sock']))
            for method in ('checkpoint', 'finish'):
                conn = Unix('localhost')
                conn.request('POST', '/control', json.dumps({'method': method}),
                             {'Authorization': 'Bearer ' + spec.env['PILOT_CONTROL_TOKEN']})
                response = json.loads(conn.getresponse().read())
                conn.close()
                assert response['ok']
            assert spec.stop_requested() == 'development_finished'
            stdout = spec.ledger_path.parent / 'stdout'
            stdout.write_text(json.dumps({'type': 'thread.started', 'thread_id': 'native-session'}) + '\n')
            spec.ledger_path.write_text('{}')
            return Result('development_finished', str(stdout))

    monkeypatch.setitem(sys.modules, 'boardbench.families.pilot_supervisor', types.SimpleNamespace(
        SandboxSupervisor=Supervisor, minimal_runtime=lambda **kwargs: [],
        SandboxSpec=lambda **kwargs: types.SimpleNamespace(**kwargs),
        Limits=lambda *args: args, WRAPPER=Path('/explicit/codex'), physical_cpu_ids=lambda n: tuple(range(n)),
    ))
    monkeypatch.setitem(sys.modules, 'boardbench.families.pilot_gt', types.SimpleNamespace(
        create_run=lambda *args, **kwargs: {'run_id': 'run', 'token': 'gt-token', 'generation': 1},
    ))
    gt = GT()
    native = NativePilotAdapter(gt_service=gt, gt_socket=tmp_path/'gt.sock', gateway_socket=tmp_path/'gateway.sock',
                                sdk_dir=tmp_path/'sdk', gateway_account=lambda *args: 'account-token',
                                gateway_status=lambda _: {'remaining': {'calls': 99}})
    runner = PilotRunner(tmp_path/'pilot', config(), native)
    context = runner._context('development', DEVELOPMENT_BUDGET, None, True)
    result = native.run_development(context, None)
    assert result.finished and result.session_id == 'native-session'
    assert runner.checkpoints.freeze() is not None
    assert gt.stopped == 'development_truncated'


def test_internal_codex_logs_do_not_reset_empty_resume_counter(tmp_path):
    class InternalLogs(FakeAdapter):
        def run_development(self, context, resume_session):
            result = super().run_development(context, resume_session)
            internal = context.workspace / '.codex'
            internal.mkdir(exist_ok=True)
            (internal / 'session.jsonl').write_text(str(len(self.dev_calls)))
            return result
    adapter = InternalLogs(finish=False)
    PilotRunner(tmp_path, config(), adapter).run()
    assert len(adapter.dev_calls) == 3


def test_concurrency_never_shares_workspaces_and_stops_at_four(tmp_path):
    import threading
    import time
    class Concurrent(FakeAdapter):
        lock = threading.Lock()
        active = 0
        peak = 0
        barrier = threading.Barrier(4)
        def run_match(self, context, match):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            self.barrier.wait(timeout=5)
            result = super().run_match(context, match)
            with self.lock:
                self.active -= 1
            return result
    adapter = Concurrent()
    assert PilotRunner(tmp_path, config(), adapter).run()['status'] == 'complete'
    assert adapter.peak == 4


def test_explicit_calibration_seeds_persist_as_json(tmp_path):
    custom = replace(config(), calibration_seeds=(11, 12, 13, 14, 15))
    adapter = FakeAdapter()
    PilotRunner(tmp_path, custom, adapter).run()
    assert {m.seed for _, m in adapter.matches} == {11, 12, 13, 14, 15}
    assert PilotRunner(tmp_path, custom, adapter).run()['status'] == 'complete'


def native_match_harness(tmp_path, monkeypatch, steps, *, program_mode=False, fallback=False):
    import sys
    import types
    from dataclasses import dataclass
    from boardbench.families.pilot_run import NativePilotAdapter, DEPLOYMENT_BUDGETS

    @dataclass
    class Result:
        reason: str
        stdout_path: str

    class GT:
        config = {'seed_key': 'test'}
        state = 'active'
        version = 1
        fallbacks = 0
        def report(self, rid):
            return {'status': self.state, 'version': self.version}
        def stop(self, rid, reason):
            self.state = reason
        def fallback(self, rid, reason, finish=False):
            self.fallbacks += 1
            self.version += 1
            self.state = 'complete'
        def rotate_controller(self, rid, source):
            return {'run_id': rid, 'token': 'rotated', 'generation': 2}

    monkeypatch.setitem(sys.modules, 'boardbench.families.pilot_gt', types.SimpleNamespace(private_seed=lambda *args: 42))
    runner = PilotRunner(tmp_path, replace(config(), calibration_seeds=(42, 43, 44, 45, 46)), FakeAdapter())
    match = default_queue(runner.config)[0]
    context = runner._context(match.match_id, DEPLOYMENT_BUDGETS['L10'], None)
    manifest = json.loads((context.workspace / 'submission' / 'manifest.json').read_text())
    if fallback or program_mode:
        (context.workspace / 'submission' / 'policy.py').write_text('pass')
        manifest['fallback_entry'] = 'policy.py'
        if program_mode:
            manifest.update(entry_mode='program', program_entry='policy.py')
        (context.workspace / 'submission' / 'manifest.json').write_text(json.dumps(manifest))
    context.ledger_path.write_text(json.dumps({'cpu_seconds': 0, 'active_wall_seconds': 0}))
    adapter = NativePilotAdapter.__new__(NativePilotAdapter)
    adapter.gt = GT()
    gateway = {'remaining': {'calls': 10, 'input_tokens': 128000, 'output_tokens': 8000}, 'frozen': False}
    adapter.gateway_status = lambda _: gateway
    record = {'gt': {'run_id': 'run', 'token': 'token', 'generation': 1}, 'registered_program': None}
    adapter._open_phase = lambda *args: record
    executions = []
    def execute(ctx, rec, **kwargs):
        index = len(executions)
        executions.append((ctx, kwargs))
        step = steps[index]
        if step.get('quota'):
            gateway['remaining']['calls'] = 0
            gateway['first_binding_limit'] = 'calls'
        if step.get('frozen'):
            gateway['frozen'] = True
        if step.get('complete'):
            adapter.gt.state = 'complete'
        if step.get('progress'):
            adapter.gt.version += 1
        context.ledger_path.write_text(json.dumps({'cpu_seconds': step.get('cpu', 1), 'active_wall_seconds': step.get('wall', 1)}))
        stdout = context.workspace.parent / f'controller-{index}.jsonl'
        stdout.write_text(('NEED_AGENT\n' if step.get('need_agent') else '') +
                          (json.dumps({'type': 'thread.started', 'thread_id': 'one-session'}) if index == 0 and not program_mode else ''))
        return Result(step.get('reason', 'exited'), str(stdout))
    adapter._execute = execute
    return adapter, context, match, executions


def test_match_resumes_same_session_twice_before_default_takeover(tmp_path, monkeypatch):
    adapter, context, match, executions = native_match_harness(tmp_path, monkeypatch, [{}, {}, {}])
    result = adapter._run_match(context, match)
    assert result['status'] == 'complete'
    assert len(executions) == 3
    assert [kwargs['resume_session'] for _, kwargs in executions] == [None, 'one-session', 'one-session']
    assert executions[1][0].prompt == executions[2][0].prompt
    assert executions[1][0].prompt != executions[0][0].prompt
    assert len({ctx.ledger_path for ctx, _ in executions}) == 1
    assert adapter.gt.fallbacks == 1


def test_match_resume_progress_resets_empty_counter(tmp_path, monkeypatch):
    adapter, context, match, executions = native_match_harness(tmp_path, monkeypatch, [{}, {}, {'progress': True}, {}, {}])
    assert adapter._run_match(context, match)['status'] == 'complete'
    assert len(executions) == 5


@pytest.mark.parametrize('step,status', [({'cpu': 121}, 'cpu_exhausted'), ({'wall': 1201}, 'wall_exhausted')])
def test_normal_exit_settled_resource_overrun_cannot_get_free_completion(tmp_path, monkeypatch, step, status):
    adapter, context, match, executions = native_match_harness(tmp_path, monkeypatch, [step])
    result = adapter._run_match(context, match)
    assert result['status'] == status and result['win_points'] == 0
    assert adapter.gt.fallbacks == 0 and len(executions) == 1


@pytest.mark.parametrize('second,status', [
    ({'reason': 'execution_error'}, 'agent_failure'),
    ({'frozen': True}, 'infrastructure_failure'),
    ({'cpu': 121}, 'cpu_exhausted'),
    ({'reason': 'rss_limit'}, 'memory_exhausted'),
])
def test_registered_program_results_use_same_failure_classification(tmp_path, monkeypatch, second, status):
    adapter, context, match, executions = native_match_harness(tmp_path, monkeypatch,
        [{'reason': 'execution_error', 'quota': True}, second], fallback=True)
    result = adapter._run_match(context, match)
    assert result['status'] == status
    assert len(executions) == 2
    assert executions[1][1]['program'] == 'submission/policy.py'
    assert adapter.gt.fallbacks == 0


def test_program_need_agent_uses_available_model_budget(tmp_path, monkeypatch):
    adapter, context, match, executions = native_match_harness(tmp_path, monkeypatch,
        [{'need_agent': True}, {'complete': True}], program_mode=True)
    assert adapter._run_match(context, match)['status'] == 'complete'
    assert [kw['program'] for _, kw in executions] == ['submission/policy.py', None]
    assert adapter.gt.fallbacks == 0


def test_decision_timeout_does_not_reset_on_session_resume(tmp_path, monkeypatch):
    import sys
    import types
    import boardbench.families.pilot_run as module
    from pathlib import Path
    adapter, context, match, executions = native_match_harness(tmp_path, monkeypatch, [])
    del adapter._execute  # Exercise the real controller wiring, not harness stub.
    record = adapter._open_phase(context, match)
    record.update(gateway_token='gateway', finished=False)
    adapter.gt.update_budget = lambda *args: None
    adapter.runtime = []
    adapter.sdk_dir = tmp_path
    adapter.gt_socket = tmp_path/'gt.sock'
    adapter.gateway_socket = tmp_path/'gateway.sock'
    adapter.phase_cores = {}
    ticks = [1000.]
    monkeypatch.setattr(module.time, 'time', lambda: ticks[0])
    decisions = []
    class Supervisor:
        def run(self, spec):
            decisions.append(spec.stop_requested())
            return types.SimpleNamespace(reason='exited', stdout_path='unused')
    adapter.supervisor = Supervisor()
    class Control:
        def shutdown(self): pass
        def server_close(self): pass
    def control(*args):
        directory = tmp_path/'control'
        directory.mkdir()
        return Control(), directory/'control.sock', 'token'
    adapter._control_server = control
    monkeypatch.setitem(sys.modules, 'boardbench.families.pilot_supervisor', types.SimpleNamespace(
        SandboxSpec=lambda **kwargs: types.SimpleNamespace(**kwargs), Limits=lambda *args: args,
        WRAPPER=Path('/explicit/codex'), physical_cpu_ids=lambda n: tuple(range(n)),
    ))
    adapter._execute(context, record, match=True)
    ticks[0] += 121
    adapter._execute(context, record, resume_session='same-session', match=True)
    assert decisions == [None, 'decision_timeout']
    adapter.gt.version += 1
    ticks[0] += 1
    adapter._execute(context, record, resume_session='same-session', match=True)
    assert decisions[-1] is None


def test_completed_queue_reopens_same_tuple_seed_config_without_reexecution(tmp_path):
    cfg = replace(config(), calibration_seeds=(11, 12, 13, 14, 15))
    adapter = FakeAdapter()
    first = PilotRunner(tmp_path, cfg, adapter).run()
    assert first['status'] == 'complete'
    another = FakeAdapter()
    second = PilotRunner(tmp_path, cfg, another).run()
    assert second == first
    assert another.dev_calls == [] and another.matches == []
