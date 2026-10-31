"""Resumable population policy search with public features and paired matches.

This is evolutionary policy optimization, not PPO. Every measured game, rejected
candidate and incomplete run is retained. A final confirmation set is not used
to choose checkpoints. Spawn workers receive policy weights, never hidden states.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

import numpy as np

from .base import FEATURE_DIM
from .play import run_match, summarize, write_json, action_features
from .registry import game_class

STOP = False


def stop_requested(*_):
    global STOP
    STOP = True


def match_worker(job):
    began = time.monotonic()
    try:
        return run_match(**job)
    except Exception as error:
        # Errors are evidence, not quietly discarded training examples.
        import traceback
        return dict(game=job['game_id'], seed=job['seed'], first_player=job['first_player'],
                    policies=job['specs'], status='error', rewards=None,
                    seconds=time.monotonic()-began, error=repr(error), traceback=traceback.format_exc())


def paired_jobs(game, candidate, opponent, seeds, record=False, max_seconds=120.):
    jobs = []
    for seed in seeds:
        for seat in (0, 1):
            specs = [opponent, opponent]
            specs[seat] = candidate
            jobs.append(dict(game_id=game, specs=specs, seed=int(seed), first_player=0,
                             max_seconds=max_seconds, record=record))
    return jobs


def evaluate(pool, jobs, directory, label):
    results = list(pool.map(match_worker, jobs, chunksize=1))
    write_json(directory / f'{label}.json', results)
    return summarize(results, [i % 2 for i in range(len(results))]), results


def artifact(cls, weights, generation, source_commit):
    return dict(format='boardbench_family_linear_v1', game=cls.game_id,
                rules_version=cls.rules_version, config=cls.config_id, kind='linear',
                weights=list(map(float, weights)), generation=generation, source_commit=source_commit,
                method='cross_entropy_policy_search', policy_inputs='public_observation_only')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--population', type=int, default=12)
    parser.add_argument('--generations', type=int, default=64)
    parser.add_argument('--train-seeds', type=int, default=2)
    parser.add_argument('--gate-seeds', type=int, default=8)
    parser.add_argument('--confirm-seeds', type=int, default=32)
    parser.add_argument('--hours', type=float, default=3.)
    parser.add_argument('--deadline-unix', type=float, default=float('inf'))
    parser.add_argument('--max-match-seconds', type=float, default=120.)
    parser.add_argument('--seed', type=int, default=160)
    parser.add_argument('--source-commit', default='unrecorded')
    args = parser.parse_args(argv)
    if args.population < 4 or min(args.workers, args.train_seeds, args.gate_seeds, args.confirm_seeds) < 1:
        raise ValueError('Positive workers/seeds and population >=4 required')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_requested)
    args.output.mkdir(parents=True, exist_ok=True)
    cls = game_class(args.game)
    rng = np.random.default_rng(args.seed)
    baseline = dict(kind='heuristic', weights=cls.default_weights)
    start = time.time()
    deadline = min(start + args.hours * 3600, args.deadline_unix)
    state_file = args.output / 'state.json'
    if (args.output/'confirmation_summary.json').exists():
        print('Run already confirmed; use a new output directory for new training.', flush=True)
        return
    mean = np.asarray(cls.default_weights, dtype=float)
    scales = np.ones(FEATURE_DIM)
    sigma = np.full(FEATURE_DIM, .65)
    sigma[0] = .2
    best = artifact(cls, mean, -1, args.source_commit)
    first_generation = 0
    records = []
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state['game'] != args.game or state['source_commit'] != args.source_commit:
            raise ValueError('Resume requires the same game and source snapshot')
        mean, sigma, scales = map(np.asarray, (state['mean'], state['sigma'], state['scales']))
        best, records = state['best'], state['records']
        first_generation = state['next_generation']
        rng.bit_generator.state = state['rng']
        deadline = min(deadline, state['deadline'])
    write_json(args.output / 'config.json', {**vars(args), 'output': str(args.output),
                                            'deadline_unix': deadline, 'pid': os.getpid()})
    def save(next_generation, phase):
        write_json(state_file, dict(game=args.game, source_commit=args.source_commit,
                   mean=mean.tolist(), sigma=sigma.tolist(), scales=scales.tolist(),
                   best=best, records=records, next_generation=next_generation,
                   rng=rng.bit_generator.state, deadline=deadline, phase=phase,
                   updated_unix=time.time(), wall_seconds=time.time()-start))
        write_json(args.output / 'best.json', best)

    # BLAS/OpenMP threads are also constrained by the cluster launcher.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        if not state_file.exists():
            boot_jobs = paired_jobs(args.game, baseline, dict(kind='random'), [10_000_001], True,
                                    args.max_match_seconds)
            boot, results = evaluate(pool, boot_jobs, args.output, 'bootstrap')
            rows = []
            for result in results:
                for item in result.get('trajectory', []):
                    rows.append(cls.features(item['observation'], item['action']))
            if rows:
                scales = np.maximum(np.std(np.asarray(rows), axis=0), 1.)
            # A heuristic with unresolved/error runs is not eligible for training.
            if boot['completed'] != boot['games']:
                save(0, 'blocked_preflight')
                raise RuntimeError(f'Game preflight incomplete: {boot}')
            save(0, 'training')
        next_generation = first_generation
        for generation in range(first_generation, args.generations):
            if STOP or time.time() + 2*args.max_match_seconds >= deadline:
                break
            candidates = mean + rng.normal(size=(args.population, FEATURE_DIM))*sigma/scales
            candidates[0] = best['weights']
            candidates[1] = cls.default_weights
            candidates[:, 0] = np.clip(candidates[:, 0], .1, 3.)
            # Fresh training environments each generation; confirmation has a disjoint namespace.
            seeds = [10_100_000 + generation*1000 + i for i in range(args.train_seeds)]
            jobs = []
            for weights in candidates:
                spec = artifact(cls, weights, generation, args.source_commit)
                jobs += paired_jobs(args.game, spec, baseline, seeds, max_seconds=args.max_match_seconds)
                jobs += paired_jobs(args.game, spec, best, seeds, max_seconds=args.max_match_seconds)
            results = list(pool.map(match_worker, jobs, chunksize=1))
            write_json(args.output / f'g{generation:03d}_population.json', results)
            batch = 4*len(seeds)
            summaries = [summarize(results[i*batch:(i+1)*batch], [j%2 for j in range(batch)])
                         for i in range(args.population)]
            # A candidate with an incomplete game receives no selection credit for that batch.
            fitness = np.asarray([s['completed_win_rate'] if s['completed'] == s['games'] else -1.
                                  for s in summaries])
            order = np.argsort(-fitness, kind='stable')
            elite = candidates[order[:max(2, args.population//4)]]
            mean = .3*mean + .7*elite.mean(axis=0)
            sigma = np.maximum(.10, .5*sigma + .5*elite.std(axis=0)*scales)
            proposed = artifact(cls, candidates[order[0]], generation, args.source_commit)
            gate = None
            promoted = False
            if fitness[order[0]] >= .5 and not np.allclose(proposed['weights'], best['weights']):
                gate_seeds = [11_000_000 + generation*1000 + i for i in range(args.gate_seeds)]
                gate, _ = evaluate(pool, paired_jobs(args.game, proposed, best, gate_seeds,
                                    max_seconds=args.max_match_seconds), args.output, f'g{generation:03d}_gate')
                if gate['completed'] == gate['games'] and gate['completed_win_rate'] >= .625:
                    best = proposed
                    promoted = True
                    write_json(args.output / f'bot_g{generation:03d}.json', best)
            record = dict(generation=generation, fitness=fitness.tolist(), gate=gate,
                          promoted=promoted, best_generation=best['generation'], unix=time.time())
            records.append(record)
            next_generation = generation+1
            save(next_generation, 'training')
            print(json.dumps(record), flush=True)
        save(next_generation, 'stopping' if STOP else 'confirmation')
        # Holdout is assessed once per completed run. Its result never changes the selected bot.
        if not STOP and time.time() + args.max_match_seconds < deadline and not (args.output/'confirmation.json').exists():
            seeds = [12_000_000+i for i in range(args.confirm_seeds)]
            summary, _ = evaluate(pool, paired_jobs(args.game, best, baseline, seeds, True,
                                  args.max_match_seconds), args.output, 'confirmation')
            write_json(args.output / 'confirmation_summary.json', summary)
            print(json.dumps(dict(confirmation=summary)), flush=True)
        save(next_generation, 'stopped' if STOP else 'complete')


if __name__ == '__main__':
    main()
