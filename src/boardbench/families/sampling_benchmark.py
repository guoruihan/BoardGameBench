"""Warmed replay benchmark, also executable against frozen 0.6.2 PYTHONPATH.

Jobs, output paths and native artifact are explicit. No formal trainer is
resumed. Save complete trajectories and post-update weights for exact pairing.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from boardbench.families.neural import load_model
from boardbench.families.pool_evaluator import bounded_jobs
from boardbench.families.pool_training import compact_rollout, audit_behavior
from boardbench.families.ppo import update
from boardbench.families.registry import game_class
from boardbench.families.trajectory_io import write_episode_batch, read_episode_batch, encode_json


def initialize(kernel):
    torch.set_num_threads(1)
    if kernel != 'reference':
        from boardbench.families.harmonies_kernel import configure
        configure(None if kernel == 'python' else json.loads(Path(kernel).read_text()))


def measured_rollout(job):
    started = time.process_time()
    episode = compact_rollout(job)
    episode['benchmark_worker_cpu_seconds'] = time.process_time()-started
    return episode


def semantic_digest(archive):
    games = read_episode_batch(archive)['games']
    # Omit timing only. Include both players, masks, returns, full observations,
    # termination, errors, policy identities and complete final score breakdown.
    for episode in games:
        for field in ('seconds', 'policy_seconds', 'policy_max_seconds'):
            episode['result'].pop(field, None)
    return hashlib.sha256(json.dumps(games, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def benchmark(jobs, output, kernel='python', workers=6, device='cuda'):
    output = Path(output);output.mkdir(parents=True, exist_ok=False)
    if not jobs:raise ValueError('Nonempty jobs required')
    if len({j['path'] for j in jobs}) != 1:raise ValueError('One learner model per batch')
    initialize(kernel)
    torch.set_num_threads(1)
    if device == 'cuda':
        torch.ones(32, device=device).sum().item();torch.cuda.synchronize()
    cls = game_class(jobs[0]['game'])
    def fresh():
        model, _ = load_model(jobs[0]['path'], cls, device)
        return model, torch.optim.Adam(model.parameters(), lr=1e-4)
    began = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(kernel,)) as pool:
        warm = bounded_jobs(pool, compact_rollout, jobs[:workers], time.time()+120, workers)
        model, optimizer = fresh()
        update(model, optimizer, [r for e in warm for r in e['records']], device,
               np.random.default_rng(1701), 4, 128, .01, .01)
        if device == 'cuda':torch.cuda.synchronize()
        warmup = time.perf_counter()-began
        model, optimizer = fresh()
        start = time.perf_counter();cpu_start = time.process_time()
        if device == 'cuda':torch.cuda.reset_peak_memory_stats()
        episodes = bounded_jobs(pool, measured_rollout, jobs, time.time()+300, workers)
        sampled = time.perf_counter()
        assert len(episodes) == len(jobs) and all(e['result']['status'] == 'complete' for e in episodes)
        archive = output/'trajectories.jsonl.gz'
        write_episode_batch(archive, dict(jobs_sha256=hashlib.sha256(encode_json(jobs)).hexdigest()), episodes)
        archived = time.perf_counter()
        rows = [r for episode in episodes for r in episode['records']]
        behavior_error = audit_behavior(model, rows, device)
        metric = update(model, optimizer, rows, device, np.random.default_rng(1701), 4, 128, .01, .01)
        if device == 'cuda':torch.cuda.synchronize()
        updated = time.perf_counter()
        coordinator_cpu = time.process_time()-cpu_start
        torch.save(dict(model={k: v.cpu() for k, v in model.state_dict().items()},
                        optimizer=optimizer.state_dict()), output/'updated.pt')
    summary = dict(kernel=kernel, device=device, workers=workers, games=len(jobs),
        learner_decisions=len(rows), environment_decisions=sum(e['result']['decisions'] for e in episodes),
        warmup_seconds=warmup, sampling_seconds=sampled-start, saving_seconds=archived-sampled,
        update_seconds=updated-archived, batch_seconds=updated-start,
        behavior_error=behavior_error, loss=metric['loss'],
        worker_cpu_seconds=sum(e['benchmark_worker_cpu_seconds'] for e in episodes),
        coordinator_cpu_seconds=coordinator_cpu,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device == 'cuda' else 0,
        trajectory_sha256=semantic_digest(archive), total_wall_seconds=time.perf_counter()-began)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kernel', default='python', help='reference, python, or pinned artifact.json')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.workers < 1:raise ValueError('Positive workers required')
    benchmark(json.loads(args.jobs.read_text()), args.output, args.kernel, args.workers, args.device)


if __name__ == '__main__':main()
