"""Frozen solo deployment evaluation; fresh seeds never select the model."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np
import torch

from boardbench.artifacts.store import digest, write_json
from boardbench.v12.timed import run_timed, aggregate
from boardbench.v134.collector import EnvironmentPool
from .baseline import load_model


def timed(args):
    spec, seed = args
    row = run_timed('harmonies', spec, seed, 3.)
    return {k: v for k, v in row.items() if k not in ('frames', 'final_observation')}


def validate(weights, native, out, count=512, workers=4):
    torch.set_num_threads(1); model, _ = load_model(weights)
    seeds = list(range(91000000, 91000000+count))
    start = time.monotonic(); games = []
    with EnvironmentPool(min(64, count), workers, native_artifact=native, arm='persistent') as pool:
        if count % pool.count:
            raise ValueError('panel must divide cohort')
        for offset in range(0, count, pool.count):
            arrays = pool.reset(seeds[offset:offset+pool.count])
            while not arrays['done'].all():
                slots = np.flatnonzero(~arrays['done']); ext = pool.extra
                plans = ext['plan'][slots].copy()
                for j, slot in enumerate(slots):
                    if ext['request'][slot]:
                        choice = int(ext['plan_mask'][slot, 1]); ext['choice'][slot] = choice
                        plans[j] = ext['plans'][slot, choice]
                x = np.concatenate((arrays['features'][slots], plans), -1)
                with torch.inference_mode():
                    logits, _ = model(torch.as_tensor(x, dtype=torch.float64),
                                      torch.as_tensor(arrays['masks'][slots, :240]))
                pool.step(slots, logits.argmax(-1).numpy())
            games += [dict(seed=seed, score=float(s[1]), animal_score=float(s[2]), natural=bool(s[4]))
                      for seed, s in zip(seeds[offset:offset+pool.count], arrays['stats'])]
    spec = dict(solver=dict(id='harmonies_baseline'), weights=str(weights), weights_sha256=digest(weights))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as executor:
        deployment = list(executor.map(timed, [(spec, s) for s in seeds[:32]]))
    result = dict(weights_sha256=digest(weights), selection_uses_these_seeds=False,
                  panel='fresh_validation_not_the_sealed_formal_test', games=games,
                  model_only_mean=float(np.mean([g['score'] for g in games])),
                  timed=aggregate(deployment), timed_games=deployment,
                  wall_seconds=time.monotonic()-start)
    write_json(out, result)
    print({k: v for k, v in result.items() if k not in ('games', 'timed_games')}, flush=True)


if __name__ == '__main__':
    from boardbench.artifacts.store import read_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True); p.add_argument('--native', required=True)
    p.add_argument('--out', required=True); p.add_argument('--count', type=int, default=512)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    validate(args.weights, read_json(args.native), args.out, args.count, args.workers)
