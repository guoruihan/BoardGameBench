"""Bounded CPU-only C++ encoding feasibility probe; never changes active training.

Reference trajectories/encoding stay authoritative. Include dict->packed conversion
and the FFI call in headline times; kernel-only timing is separately labeled.
"""
import argparse
import cProfile
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
from boardbench.v12.encoding import action_map, feature_vector
from boardbench.v125.policies import GoalContinuation
from boardbench.v128.environment import FastHarmonies
from boardbench.v129.encoding import CPP_SOURCE, FEATURES, WIDTH, NativeEncoder, pack


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
    cpp = CPP_SOURCE
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
