"""Synchronous, complete-game collection with batched inference.

Workers own private environments. Only public encodings and transition statistics
cross the shared-memory boundary. A pool reset starts exactly one fixed cohort;
finished slots never auto-reset or contribute samples to the next policy version.
"""
import ctypes
import multiprocessing as mp
import os
import time
import traceback

import numpy as np

from boardbench.environments.harmonies import Harmonies
from boardbench.v12.encoding import STOP, decode, encode

FEATURES, ACTIONS = 1346, 241
# Last transition: reward, total score, animal score, completed cards, natural,
# STOP, animal placement. Only live-at-start slots are consumed by the collector.
REWARD, SCORE, ANIMAL, COMPLETED, NATURAL, STOPPED, PLACEMENT = range(7)


def _views(buffers, count):
    return {
        'features': np.frombuffer(buffers['features'], np.float32).reshape(count, FEATURES),
        'masks': np.frombuffer(buffers['masks'], np.bool_).reshape(count, ACTIONS),
        'actions': np.frombuffer(buffers['actions'], np.int64),
        'done': np.frombuffer(buffers['done'], np.bool_),
        'stats': np.frombuffer(buffers['stats'], np.float64).reshape(count, 7),
    }


class _Shard:
    def __init__(self, slots, arrays):
        self.slots, self.arrays = slots, arrays
        self.games, self.observations, self.steps = {}, {}, {}

    def _publish(self, slot):
        if self.arrays['done'][slot]:
            self.arrays['features'][slot].fill(0)
            self.arrays['masks'][slot].fill(False)
        else:
            data = encode(self.observations[slot])
            self.arrays['features'][slot] = data['features']
            self.arrays['masks'][slot] = data['action_mask']

    def reset(self, seeds):
        for slot in self.slots:
            env = Harmonies()
            obs, _ = env.reset(seeds[slot])
            self.games[slot], self.observations[slot], self.steps[slot] = env, obs, 0
            self.arrays['done'][slot] = False
            self.arrays['stats'][slot].fill(0)
            self._publish(slot)

    def step(self):
        for slot in self.slots:
            if self.arrays['done'][slot]:
                continue
            obs = self.observations[slot]
            action = decode(obs, int(self.arrays['actions'][slot]), 'slots240_v1', True)
            stopped = action == STOP
            nxt = obs if stopped else self.games[slot].step(action).observation
            self.steps[slot] += 1
            done = stopped or nxt['terminated']
            if self.steps[slot] >= 256 and not done:
                raise RuntimeError('game action limit')
            self.arrays['stats'][slot] = (
                (nxt['score_breakdown']['total'] - obs['score_breakdown']['total']) / 150.,
                nxt['score_breakdown']['total'], nxt['score_breakdown']['animals'],
                len(nxt['completed_cards']), nxt['terminated'], stopped,
                action.get('type') == 'place_animal',
            )
            self.observations[slot] = nxt
            self.arrays['done'][slot] = done
            self._publish(slot)


def _worker(connection, buffers, count, slots, core):
    try:
        if core is not None:
            os.sched_setaffinity(0, {core})
        shard = _Shard(slots, _views(buffers, count))
        connection.send(('ready', None))
        while True:
            command, value = connection.recv()
            if command == 'close':
                break
            if command == 'reset':
                shard.reset(value)
            elif command == 'step':
                shard.step()
            else:
                raise ValueError('unknown environment worker command')
            connection.send(('ok', None))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class EnvironmentPool:
    def __init__(self, count=16, workers=0, timeout=60.):
        if type(count) is not int or count < 1 or type(workers) is not int or not 0 <= workers <= count:
            raise ValueError('positive cohort size and workers in [0, count] required')
        cores = sorted(os.sched_getaffinity(0))
        if workers and workers > len(cores) - 1:
            raise ValueError('workers require distinct cores plus one learner core')
        self.count, self.timeout = count, timeout
        self.connections, self.processes = [], []
        self.closed = False
        ctx = mp.get_context('spawn')
        self.buffers = {
            'features': ctx.RawArray(ctypes.c_float, count * FEATURES),
            'masks': ctx.RawArray(ctypes.c_bool, count * ACTIONS),
            'actions': ctx.RawArray(ctypes.c_int64, count),
            'done': ctx.RawArray(ctypes.c_bool, count),
            'stats': ctx.RawArray(ctypes.c_double, count * 7),
        }
        self.arrays = _views(self.buffers, count)
        self.local = _Shard(list(range(count)), self.arrays) if workers == 0 else None
        self.arrays['done'].fill(True)
        self.started = False
        try:
            for i in range(workers):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker, args=(child, self.buffers, count,
                    list(range(i, count, workers)), cores[i + 1]), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self._receive('ready')
        except BaseException:
            self.close()
            raise

    def _receive(self, expected):
        deadline = time.monotonic() + self.timeout
        for pipe in self.connections:
            if not pipe.poll(max(0., deadline - time.monotonic())):
                raise TimeoutError('environment worker did not respond')
            try:
                tag, value = pipe.recv()
            except EOFError as exc:
                raise RuntimeError('environment worker exited unexpectedly') from exc
            if tag != expected:
                raise RuntimeError(f'environment worker failed: {value}')

    def _command(self, name, value=None):
        if self.closed:
            raise RuntimeError('environment pool is closed')
        if self.local is not None:
            self.local.reset(value) if name == 'reset' else self.local.step()
        else:
            for pipe in self.connections:
                pipe.send((name, value))
            self._receive('ok')

    def reset(self, seeds):
        if len(seeds) != self.count:
            raise ValueError('one explicit seed per cohort slot required')
        if self.started and not self.arrays['done'].all():
            raise ValueError('cannot discard unfinished cohort')
        self._command('reset', list(seeds))
        self.started = True
        return self.arrays

    def step(self, slots, actions):
        if not self.started or not np.array_equal(slots, np.flatnonzero(~self.arrays['done'])):
            raise ValueError('exact live-slot ordering required')
        if len(slots) == 0 or len(slots) != len(actions):
            raise ValueError('one action per live slot required')
        if any(isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, np.integer)) for a in actions):
            raise ValueError('integer actions required')
        actions = np.asarray(actions, dtype=np.int64)
        if (actions < 0).any() or (actions >= ACTIONS).any() or not self.arrays['masks'][slots, actions].all():
            raise ValueError('action outside recorded legal mask')
        self.arrays['actions'][slots] = actions
        self._command('step')
        return self.arrays

    def close(self):
        if self.closed:
            return
        self.closed = True
        for pipe in self.connections:
            try:
                pipe.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2.)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.)
        for pipe in self.connections:
            pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def collect(trainer, pool):
    """Return original PPO row format in episode-major order, not completion order."""
    import torch
    from boardbench.v12.training import policy_distribution, returns

    count = pool.count
    seeds = [trainer.splits['train'][(trainer.episodes + i) % len(trainer.splits['train'])] for i in range(count)]
    tick = time.monotonic()
    arrays = pool.reset(seeds)
    environment_seconds = time.monotonic() - tick
    model_seconds = 0.
    trajectories = [[] for _ in seeds]
    logps, values = [], []
    decisions = stops = placements = actions = forwards = 0
    while not arrays['done'].all():
        slots = np.flatnonzero(~arrays['done'])
        # Copies are necessary: workers overwrite the shared buffers after step().
        x, mask = arrays['features'][slots].copy(), arrays['masks'][slots].copy()
        tick = time.monotonic()
        with torch.no_grad():
            logits, value = trainer.solver.model(torch.as_tensor(x, device=trainer.device),
                                                 torch.as_tensor(mask, device=trainer.device))
            dist = policy_distribution(logits)
            sampled = dist.sample()
            logps.append(dist.log_prob(sampled))
            values.append(value)
            chosen = sampled.cpu().numpy()
        model_seconds += time.monotonic() - tick
        forwards += 1
        tick = time.monotonic()
        pool.step(slots, chosen)
        environment_seconds += time.monotonic() - tick
        stats = arrays['stats'][slots]
        for j, slot in enumerate(slots):
            trajectories[slot].append((x[j], mask[j], int(chosen[j]), decisions + j, stats[j, REWARD]))
        decisions += len(slots)
        stops += int(stats[:, STOPPED].sum())
        actions += len(slots) - int(stats[:, STOPPED].sum())
        placements += int(stats[:, PLACEMENT].sum())
    tick = time.monotonic()
    old = torch.cat(logps).cpu().numpy()
    val = torch.cat(values).cpu().numpy()
    model_seconds += time.monotonic() - tick
    rows = []
    for trajectory in trajectories:
        targets = returns([r[-1] for r in trajectory])
        for (x, mask, chosen, flat, _), target in zip(trajectory, targets):
            rows.append((x, mask, chosen, float(old[flat]), target, target - float(val[flat])))
    end = arrays['stats']
    metrics = dict(environment_seconds=environment_seconds, model_seconds=model_seconds,
        inference_calls=forwards, mean_inference_batch=decisions / forwards,
        mean_training_score=float(end[:, SCORE].mean()), mean_training_animal_score=float(end[:, ANIMAL].mean()),
        mean_score_gain=float(end[:, SCORE].mean()), animal_placements_per_episode=placements / count,
        completed_cards_per_episode=float(end[:, COMPLETED].mean()), natural_fraction=float(end[:, NATURAL].mean()),
        real_actions_this_batch=actions, stop_decisions_this_batch=stops, decisions_this_batch=decisions)
    return rows, metrics
