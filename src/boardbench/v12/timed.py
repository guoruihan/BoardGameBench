"""Parent-owned scoring with killable resident policies and deadline-aware IPC.

This is a local cooperative-code boundary, not an adversarial sandbox or a
hard-real-time operating system. No solver callback executes in the parent.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import select
import signal
import socket
import statistics
import struct
import time
from types import SimpleNamespace

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.contracts import EpisodeResult, StepResult, Transition
from boardbench.environments import TASKS
from boardbench.environments.adapters import SearchAccess
from boardbench.solvers import make_solver
from .encoding import STOP

PROTOCOL = 'timed_score_v1'
MAX_MESSAGE = 4 * 1024 * 1024


def _wait(sock, deadline, writing=False):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('deadline')
    ready = select.select([] if writing else [sock], [sock] if writing else [], [], remaining)
    if not ready[1 if writing else 0] or time.monotonic() >= deadline:
        raise TimeoutError('deadline')


def send(sock, value, deadline):
    data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    if len(data) > MAX_MESSAGE:
        raise ValueError('IPC message exceeds limit')
    pending = memoryview(struct.pack('!I', len(data)) + data)
    while pending:
        _wait(sock, deadline, True)
        try:
            n = sock.send(pending)
        except BlockingIOError:
            continue
        if not n:
            raise EOFError('worker socket closed')
        pending = pending[n:]


def receive(sock, deadline):
    def exact(n):
        data = bytearray()
        while len(data) < n:
            _wait(sock, deadline)
            try:
                part = sock.recv(n - len(data))
            except BlockingIOError:
                continue
            if not part:
                raise EOFError('worker socket closed')
            data.extend(part)
        return data
    size = struct.unpack('!I', exact(4))[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError('invalid IPC frame length')
    payload = exact(size)
    if time.monotonic() >= deadline:
        raise TimeoutError('late complete message')
    return json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def build_policy(spec):
    spec = deepcopy(spec)
    solver = make_solver(spec['solver'], spec.get('policy_seed', 90210))
    if spec.get('weights'):
        if digest(spec['weights']) != spec['weights_sha256']:
            raise ValueError('policy weight hash mismatch')
        solver.load_weights(spec['weights'])
    return solver


def worker(sock, spec, task):
    os.setsid()
    sock.setblocking(False)
    try:
        if spec['solver']['id'] in ('ppo', 'compact'):
            import torch
            torch.set_num_threads(1)
        solver = build_policy(spec)
        context = SimpleNamespace(task_spec=deepcopy(TASKS[task].spec), reference_simulator=None,
                                  allow_learning=False, time_budget_seconds=None)
        solver.start_task(context)
        send(sock, {'kind': 'ready'}, time.monotonic() + 60)
        first = True
        previous = action = None
        while True:
            message = receive(sock, time.monotonic() + 3600)
            if message.get('kind') == 'settlement':
                if message.get('feedback') is not None:
                    solver.on_transition(Transition(previous, action, StepResult(**message['feedback'])), context)
                solver.end_episode(EpisodeResult(**message['result']), context)
                send(sock, {'kind': 'settled'}, time.monotonic() + .05)
                return
            obs, deadline = message['observation'], message['deadline']
            context.time_budget_seconds = message['time_budget_seconds']
            context.remaining_seconds = lambda: max(0., deadline - time.monotonic())
            context.seconds_left = context.remaining_seconds
            counter = [0]
            def count():
                counter[0] += 1
            context.reference_simulator = SearchAccess(
                lambda seed: TASKS[task].factory.from_observation(obs, seed), count)
            if message.get('feedback') is not None:
                solver.on_transition(Transition(previous, action, StepResult(**message['feedback'])), context)
            if first:
                solver.start_episode(deepcopy(obs), context)
                first = False
            action = solver.decide(deepcopy(obs), context)
            previous = deepcopy(obs)
            send(sock, {'kind': 'action', 'request_id': message['request_id'],
                        'action': action, 'simulation_steps': counter[0]}, deadline)
    except (EOFError, BrokenPipeError, TimeoutError):
        pass
    except Exception as exc:
        try:
            send(sock, {'kind': 'error', 'error': f'{type(exc).__name__}: {exc}'}, time.monotonic() + .1)
        except (OSError, TimeoutError):
            pass
    finally:
        sock.close()


def _reap(process):
    # setsid is the first worker instruction; kill only this task's group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        if process.is_alive():
            process.kill()
    process.join(timeout=.5)
    return not process.is_alive()


def run_timed(task, spec, seed, seconds, *, startup_seconds=30., _worker=worker,
              _environment=None):
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(startup_seconds) or startup_seconds <= 0:
        raise ValueError('positive finite timing budgets required')
    begin = time.monotonic()
    env = _environment if _environment is not None else TASKS[task].factory()
    obs, _ = env.reset(seed)
    prepared = time.monotonic()
    parent, child = socket.socketpair()
    parent.setblocking(False)
    process = mp.get_context('spawn').Process(target=_worker, args=(child, spec, task))
    # The child receives no environment, test seed or private snapshot at startup.
    process.start()
    child.close()
    frames = []
    started = None
    deadline = None
    reason, error, failed = 'deadline', None, False
    steps = simulations = animals = takes = 0
    atomic_tail = 0.
    feedback = None
    try:
        ready = receive(parent, prepared + startup_seconds)
        if ready.get('kind') != 'ready':
            raise RuntimeError(ready.get('error', 'invalid startup handshake'))
        started = time.monotonic()
        deadline = started + seconds
        while not obs['terminated']:
            if steps >= 256:
                raise RuntimeError('finite game action bound exceeded')
            send(parent, {'request_id': steps, 'observation': obs, 'deadline': deadline,
                          'time_budget_seconds': seconds, 'feedback': feedback}, deadline)
            feedback = None
            reply = receive(parent, deadline)
            received = time.monotonic()
            if received >= deadline:
                raise TimeoutError('late action')
            if reply.get('kind') != 'action' or reply.get('request_id') != steps:
                raise RuntimeError(reply.get('error', 'unexpected or stale action response'))
            action = reply['action']
            simulations += reply.get('simulation_steps', 0)
            if action == STOP:
                reason = 'stopped'
                break
            previous = obs
            result = env.step(deepcopy(action))
            feedback = asdict(result)
            obs = result.observation
            committed = time.monotonic()
            atomic_tail = max(atomic_tail, max(0., committed - deadline))
            steps += 1
            animals += action.get('type') == 'place_animal'
            takes += action.get('type') == 'take_card'
            frames.append({'action': action, 'observation': deepcopy(obs),
                           'received_seconds': received - started, 'committed_seconds': committed - started,
                           'score_delta': obs['score_breakdown']['total'] - previous['score_breakdown']['total']})
            if obs['terminated']:
                reason = 'natural'
                break
            if committed >= deadline:
                reason = 'deadline'
                break
    except TimeoutError:
        if started is None:
            failed, reason, error = True, 'startup_timeout', 'startup limit'
        else:
            reason = 'deadline'
    except Exception as exc:
        failed, reason, error = True, 'failed', f'{type(exc).__name__}: {exc}'
    finally:
        settled = time.monotonic()
        # Post-settlement hooks are advisory and bounded, never allowed to change
        # the committed score or prolong the primary online clock.
        cleanup_ack = False
        try:
            end = EpisodeResult(1, None, 'failed' if failed else 'completed', reason,
                                0 if failed else obs['score_breakdown']['total'], steps, steps)
            cleanup_deadline = settled + .05
            send(parent, {'kind': 'settlement', 'result': asdict(end), 'feedback': feedback}, cleanup_deadline)
            cleanup_ack = receive(parent, cleanup_deadline).get('kind') == 'settled'
        except (OSError, EOFError, TimeoutError, ValueError):
            pass
        parent.close()
        reaped = _reap(process)
    # The last committed public observation is authoritative, not a child score.
    breakdown = obs['score_breakdown']
    components = {'terrain': breakdown['total'] - breakdown.get('animals', 0),
                  'animals': breakdown.get('animals', 0), 'animal_placements': animals,
                  'cards_taken': takes, 'completed_cards': len(obs.get('completed_cards', []))}
    if not reaped:
        failed, reason, error = True, 'cleanup_failed', 'worker survived bounded cleanup'
    return {'protocol': PROTOCOL, 'task': task, 'seed': seed,
            'score': 0 if failed else breakdown['total'], 'diagnostic_current_score': breakdown['total'],
            'reason': reason, 'failed': failed, 'error': error,
            'game_terminated': obs['terminated'], 'episode_finished': True,
            'time_budget_seconds': seconds, 'startup_seconds': (started or settled) - prepared,
            'preparation_seconds': prepared - begin, 'online_seconds': settled - started if started else 0.,
            'settlement_overrun_seconds': max(0., settled - deadline) if deadline else 0.,
            'atomic_tail_seconds': atomic_tail, 'cleanup_seconds': time.monotonic() - settled,
            'worker_reaped': reaped, 'actions': steps, 'simulation_steps': simulations,
            'simulation_steps_scope': 'reported accepted replies; in-flight killed work is covered by wall-clock reservation',
            'terminal_hook_acknowledged': cleanup_ack,
            'components': components, 'frames': frames, 'final_observation': obs}


def aggregate(rows):
    if not rows:
        raise ValueError('no timed evaluation attempts')
    return {'protocol': PROTOCOL, 'attempts': len(rows),
            'score_mean': statistics.mean(r['score'] for r in rows),
            'score_std': statistics.pstdev(r['score'] for r in rows),
            'natural_completion_rate': statistics.mean(r['reason'] == 'natural' for r in rows),
            'failures': sum(r['failed'] for r in rows),
            'online_seconds_mean': statistics.mean(r['online_seconds'] for r in rows),
            'startup_seconds_mean': statistics.mean(r['startup_seconds'] for r in rows),
            'score_policy': 'all attempts; natural/deadline/STOP current score, failures zero',
            'components': {k: statistics.mean(r['components'][k] for r in rows) for k in rows[0]['components']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    with (out / 'episodes.jsonl').open('x') as stream:
        for seed in config['seeds']:
            row = run_timed(config['task'], config['policy'], seed, config['seconds'])
            rows.append(row)
            stream.write(json.dumps(row) + '\n'); stream.flush()
    write_json(out / 'summary.json', aggregate(rows))


if __name__ == '__main__':
    main()
