"""Global admission holds work with already-running gateways' shared lock."""
import fcntl
import importlib.util
import json
from pathlib import Path
import socket
import time

import pytest

spec=importlib.util.spec_from_file_location('api_guard',Path(__file__).resolve().parents[2]/'scripts/guard_harmonies_api.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


def test_shared_hold_blocks_every_gateway_and_resets_spacing(tmp_path):
    path=tmp_path/'clock';path.write_text('0')
    hold=guard.AdmissionHold(path,time.time()+30)
    hold.hold()
    try:
        assert float(path.read_text())>time.monotonic()+120
        for _ in range(3):
            with path.open('a+') as gateway:
                with pytest.raises(BlockingIOError):fcntl.flock(gateway,fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally:hold.release(True)
    with path.open('a+') as gateway:fcntl.flock(gateway,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert abs(float(path.read_text())-time.monotonic())<1


def test_crashed_hold_marker_fails_closed(tmp_path):
    path=tmp_path/'clock';path.write_text('0')
    hold=guard.AdmissionHold(path,time.time()+30);hold.hold()
    hold.release(False)
    assert float(path.read_text())>time.monotonic()+120


def test_transport_scan_deduplicates_and_ignores_waiting_timeouts(tmp_path):
    p=tmp_path/'dev/gateway/requests/request';p.mkdir(parents=True)
    (p/'input_tokens_error.json').write_text('{"kind":"URLError"}')
    (p/'generate_error.json').write_text('{"kind":"GatewayError","detail":"request_timeout"}')
    (p/'input_tokens_attempt_1_error.json').write_text('{"kind":"URLError"}')
    seen=set();errors=guard.transport_errors(tmp_path,seen)
    assert len(errors)==1 and errors[0]['kind']=='URLError'
    assert guard.transport_errors(tmp_path,seen)==[]


def test_one_recovery_controller_waits_for_all_without_model_probe(tmp_path):
    started=time.time();clock=[started]
    (tmp_path/'private').mkdir();(tmp_path/'private/api_clock').write_text('0')
    (tmp_path/'preregistered_plan.json').write_text(json.dumps({'deadline':started+200}))
    config=tmp_path/'private/upstream.json';config.write_text('{"base_url":"https://test.invalid/v1"}')
    error=tmp_path/'dev/gateway/requests/request/input_tokens_error.json';error.parent.mkdir(parents=True)
    error.write_text('{"kind":"URLError"}')
    (tmp_path/'dev/experiment').mkdir()
    (tmp_path/'dev/experiment/state.json').write_text('{"status":"complete"}')
    dns=[];blocked=[]
    def sleep(seconds):
        if clock[0]<started+120:
            for _ in range(3):
                with (tmp_path/'private/api_clock').open('a+') as stream:
                    with pytest.raises(BlockingIOError):fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            blocked.append(clock[0])
        else:(tmp_path/'finished.json').write_text('{}')
        clock[0]+=seconds
    def resolve(*args,**kwargs):
        dns.append(clock[0])
        if len(dns)==1:raise socket.gaierror(-2,'injected')
        return [('resolved',)]
    guard.guard(tmp_path,config,clock=lambda:clock[0],sleep=sleep,resolve=resolve)
    assert dns==[started+60,started+120] and len(blocked)==120
    assert json.loads((tmp_path/'api_guard_state.json').read_text())['mode']=='finished'
    assert 'test.invalid' not in (tmp_path/'api_guard_events.jsonl').read_text()


def test_guard_outlives_coordinator_while_parallel_worker_active(tmp_path):
    (tmp_path/'finished.json').write_text('{"status":"paused_infrastructure"}')
    task=tmp_path/'dev-b200-r0';task.mkdir()
    assert not guard.study_finished(tmp_path)
    (task/'experiment').mkdir();state=task/'experiment/state.json'
    state.write_text('{"status":"running"}')
    assert not guard.study_finished(tmp_path)
    state.write_text('{"status":"complete"}')
    assert guard.study_finished(tmp_path)
