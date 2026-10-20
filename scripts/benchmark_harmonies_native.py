"""Bounded CPU-only C++ encoding feasibility probe; never changes active training.

Reference trajectories/encoding stay authoritative. Include dict->packed conversion
and the FFI call in headline times; kernel-only timing is separately labeled.
"""
import argparse
import cProfile
import ctypes
import io
import json
import os
from pathlib import Path
import pstats
import random
import shutil
import statistics
import subprocess
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.environments.harmonies import CARDS, COLORS
from boardbench.v12.encoding import CARD_VECTORS, MAX_CUBES, action_map, feature_vector
from boardbench.v125.policies import GoalContinuation
from boardbench.v128.environment import FastHarmonies

WIDTH = 161
FEATURES = 1346


def pack(observations):
    rows = []
    for obs in observations:
        active = obs['active_cards']
        owned = {c['card_id']: c['placed_count'] for c in active + obs['completed_cards']}
        remaining = set(obs['remaining_cards'])
        row = [c['stack_id'] for c in obs['board']] + [c['animal'] or 0 for c in obs['board']]
        row += [(bundle or []).count(color) for bundle in obs['token_market'] for color in COLORS]
        for field in ('pending_tokens', 'remaining_tokens', 'discarded_tokens'):
            row += [obs[field][color] for color in COLORS]
        row += [c or 0 for c in obs['animal_market']]
        row += [c['card_id'] for c in active] + [0] * (4-len(active))
        row += [c['placed_count'] for c in active] + [0] * (4-len(active))
        row += [owned.get(c, -1) for c in range(1, 33)]
        row += [int(c in remaining) for c in range(1, 33)]
        row += [obs['turn_index'], obs['bundle_taken'], obs['card_taken'], obs['terminated']]
        if len(row) != WIDTH:
            raise ValueError('public state layout changed')
        rows.append(row)
    return np.asarray(rows, dtype=np.int16).reshape(len(rows), WIDTH)


class NativeEncoder:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        self.short = ctypes.POINTER(ctypes.c_int16)
        self.single = ctypes.POINTER(ctypes.c_float)
        self.fn = self.lib.harmonies_encode
        self.fn.argtypes = [ctypes.c_size_t, self.short, self.single, self.short, ctypes.c_int, self.single]
        self.fn.restype = ctypes.c_int
        self.cards = np.asarray([(0.,)*83] + [CARD_VECTORS[i] for i in range(1, 33)], dtype=np.float32)
        if self.cards.shape != (33, 83):
            raise ValueError('card encoding changed')
        self.cubes = np.asarray([0] + [CARDS[i]['cube_count'] for i in range(1, 33)], dtype=np.int16)
        self.card_pointer = self.cards.ctypes.data_as(self.single)
        self.cube_pointer = self.cubes.ctypes.data_as(self.short)

    def encode(self, states):
        if states.dtype != np.int16 or states.ndim != 2 or states.shape[1] != WIDTH or not states.flags.c_contiguous:
            raise ValueError('contiguous public state array required')
        out = np.empty((len(states), FEATURES), dtype=np.float32)
        code = self.fn(len(states), states.ctypes.data_as(self.short), self.card_pointer,
                       self.cube_pointer, MAX_CUBES, out.ctypes.data_as(self.single))
        if code:
            raise ValueError(f'native encoder rejected state: {code}')
        return out


def collect(seeds):
    snapshots, traces = [], []
    rng = random.Random(128)
    heuristic = GoalContinuation()
    for i, seed in enumerate(seeds):
        env = FastHarmonies.__new__(FastHarmonies)
        obs, _ = env.reset(seed)
        trace = []
        while not obs['terminated']:
            snapshots.append(obs)
            action = heuristic.decide(obs) if i % 2 else rng.choice(obs['legal_actions'])
            trace.append(action)
            obs = env.step(action).observation
        snapshots.append(obs)
        traces.append((seed, trace))
    return snapshots, traces


def timed(fn, repeats=5):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter()-start)
    return {'median_seconds': statistics.median(values), 'seconds': values}


def run(out, games):
    if not 2 <= games <= 64:
        raise ValueError('bounded probe: 2 to64 train games')
    out = Path(out).resolve()
    repo = Path(__file__).resolve().parents[1]
    if not out.is_relative_to(repo/'outputs/v128'):
        raise ValueError('project prototype output required')
    out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    began = time.monotonic()
    cpp = Path(__file__).with_name('native')/'harmonies_encode.cpp'
    library = out/'harmonies_encode.so'
    compiler = shutil.which('g++')
    if compiler is None:
        raise RuntimeError('g++ required; no automatic installs')
    command = [compiler, '-O3', '-std=c++17', '-shared', '-fPIC', str(cpp), '-o', str(library)]
    subprocess.run(command, check=True, timeout=120)
    encoder = NativeEncoder(library)
    manifest_path = repo/'outputs/v121/factorial_f64_20261004_c/plan/splits.json'
    seeds = read_json(manifest_path)['sets']['train'][3000:3000+games]
    observations, traces = collect(seeds)
    expected = np.asarray([feature_vector(o) for o in observations], dtype=np.float32)
    states = pack(observations)
    actual = encoder.encode(states)
    np.testing.assert_array_equal(actual, expected)
    # Exercise empty calls and reject invalid IDs before any out-of-bounds index.
    assert encoder.encode(states[:0]).shape == (0, FEATURES)
    bad = states[:1].copy(); bad[0, 0] = 15
    try:
        encoder.encode(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid terrain accepted')
    baseline = timed(lambda: np.asarray([feature_vector(o) for o in observations], dtype=np.float32))
    cases = []
    for size in (1, 16, 128):
        groups = [observations[i:i+size] for i in range(0, len(observations), size)]
        packed = [pack(g) for g in groups]
        def end_to_end():
            for group in groups:
                encoder.encode(pack(group))
        def kernel():
            for group in packed:
                encoder.encode(group)
        for group, state in zip(groups, packed):
            np.testing.assert_array_equal(encoder.encode(state), np.asarray([feature_vector(o) for o in group], dtype=np.float32))
        timing = timed(end_to_end)
        cases.append({'batch': size, 'packing_ffi_and_encoding': timing, 'prepacked_ffi_and_encoding': timed(kernel),
                      'speedup_vs_reference_encoding': baseline['median_seconds']/timing['median_seconds']})
    def replay(native=False):
        env = FastHarmonies.__new__(FastHarmonies)
        for seed, trace in traces:
            obs, _ = env.reset(seed)
            for action in trace:
                if native:
                    encoder.encode(pack([obs]))
                else:
                    np.asarray(feature_vector(obs), dtype=np.float32)
                action_map(obs, 'slots240_v1', True)
                obs = env.step(action).observation
    replay_reference = timed(replay)
    replay_native = timed(lambda: replay(True))
    profiler = cProfile.Profile(); profiler.runcall(replay)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(25)
    profiler.dump_stats(str(out/'replay.prof'))
    (out/'profile.txt').write_text(stream.getvalue())
    shutil.copy2(cpp, out/cpp.name)
    shutil.copy2(__file__, out/Path(__file__).name)
    summary = {'status': 'completed', 'source_id': source_id(), 'prototype_sha256': digest(__file__),
               'cpp_sha256': digest(cpp), 'manifest_sha256': digest(manifest_path), 'train_seeds': seeds,
               'games': games, 'states': len(states), 'array_equal': True, 'packed_bytes_per_state': WIDTH*2,
               'encoded_bytes_per_state': FEATURES*4, 'reference_encoding': baseline, 'cases': cases,
               'environment_replay_reference': replay_reference, 'environment_replay_native': replay_native,
               'environment_replay_speedup': replay_reference['median_seconds']/replay_native['median_seconds'],
               'compiler_command': command, 'host': os.uname().nodename, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
               'cpu_core_seconds': time.monotonic()-began, 'gpu_seconds': 0,
               'scope': 'CPU encoder and fixed-trace environment replay incl packing/FFI; not live PPO speedup; no production backend'}
    write_json(out/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--games', type=int, default=32)
    args = parser.parse_args()
    run(args.out, args.games)
