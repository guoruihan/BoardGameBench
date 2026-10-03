import os
import struct
import time

import pytest

from boardbench.environments.micro import MicroTiles
from boardbench.solvers.board import ranked_actions
from boardbench.v12.timed import aggregate, receive, run_timed, send


def blocked_hook_worker(sock, spec, task):
    from boardbench.v12 import timed
    from boardbench.solvers.board import BoardSolver
    class Blocked(BoardSolver):
        def on_transition(self, transition, context):
            time.sleep(30)
    timed.build_policy = lambda spec: Blocked(1, method='heuristic')
    timed.worker(sock, spec, task)


def fixture_worker(sock, spec, task):
    os.setsid(); sock.setblocking(False)
    mode = spec.get('mode')
    if mode == 'startup_block':
        time.sleep(20)
    send(sock, {'kind': 'ready'}, time.monotonic() + 5)
    while True:
        message = receive(sock, time.monotonic() + 5)
        obs = message['observation']
        if mode == 'block' or (mode == 'partial_score' and message['request_id'] >= 3):
            time.sleep(20)
        if mode == 'partial':
            sock.send(struct.pack('!I', 200) + b'{')
            time.sleep(20)
        if mode == 'crash':
            return
        action = {'type': 'stop_episode'} if mode == 'stop' else ranked_actions(obs)[0][1]
        if mode == 'invalid':
            action = {'offer_index': 999, 'cell_id': 999}
        send(sock, {'kind': 'action', 'request_id': message['request_id'], 'action': action}, time.monotonic() + 5)


@pytest.mark.parametrize('mode', ['block', 'partial', 'partial_score'])
def test_parent_deadline_does_not_wait_for_worker(mode):
    result = run_timed('micro_tiles', {'mode': mode}, 7, .15, _worker=fixture_worker)
    assert result['reason'] == 'deadline' and result['episode_finished'] and not result['game_terminated']
    assert result['score'] == result['final_observation']['score_breakdown']['total']
    assert result['online_seconds'] < .65 and result['worker_reaped']
    assert result['actions'] == (3 if mode == 'partial_score' else 0)
    if mode == 'partial_score':
        assert result['score'] > 0


@pytest.mark.parametrize('mode,reason', [('stop', 'stopped'), ('crash', 'failed'),
                                       ('invalid', 'failed'), ('startup_block', 'startup_timeout')])
def test_stop_failure_and_startup_are_distinct(mode, reason):
    result = run_timed('micro_tiles', {'mode': mode}, 9, .2,
                       startup_seconds=.3 if mode == 'startup_block' else 5., _worker=fixture_worker)
    assert result['reason'] == reason and result['score'] == 0 and result['worker_reaped']


@pytest.mark.parametrize('game', ['micro_tiles', 'take_it_easy', 'harmonies'])
def test_natural_ending_and_all_attempts_aggregation(game):
    row = run_timed(game, {'solver': {'id': 'board', 'params': {'method': 'heuristic'}}}, 31, 3.)
    assert row['reason'] == 'natural' and row['game_terminated']
    assert row['score'] == row['final_observation']['score']
    summary = aggregate([row, {**row, 'score': 0, 'failed': True, 'reason': 'failed'}])
    assert summary['score_mean'] == row['score'] / 2
    assert summary['natural_completion_rate'] == .5 and summary['failures'] == 1


class SlowStep(MicroTiles):
    def step(self, action):
        time.sleep(.1)
        return super().step(action)


def test_accepted_atomic_step_may_finish_after_deadline():
    row = run_timed('micro_tiles', {}, 1, .05, _worker=fixture_worker, _environment=SlowStep())
    assert row['actions'] == 1 and row['reason'] == 'deadline'
    assert row['atomic_tail_seconds'] > 0
    assert row['frames'][0]['received_seconds'] < .05


def test_blocked_feedback_cannot_block_parent_settlement():
    spec = {'solver': {'id': 'board', 'params': {'method': 'heuristic'}}}
    row = run_timed('micro_tiles', spec, 10, .15, _worker=blocked_hook_worker)
    assert row['reason'] == 'deadline' and row['actions'] == 1
    assert row['online_seconds'] < .6 and row['worker_reaped']
