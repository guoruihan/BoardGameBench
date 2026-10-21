"""Current-policy planning states and paired terminal Q with independent futures."""
import argparse
from contextlib import nullcontext
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v12.encoding import decode
from boardbench.v121.collector import SCORE, ANIMAL, NATURAL, STOPPED
from boardbench.v128.environment import FastHarmonies
from boardbench.v132.collector import CurriculumShard
from boardbench.v134.collector import EnvironmentPool, PlanShard
from boardbench.v134.plans import candidates, encode_plan, K, PLAN_DIM
from boardbench.v129.encoding import NativeEncoder, pack


class PublicPlanShard(PlanShard):
    def reset(self, specifications):
        if not all(isinstance(s, dict) for s in specifications):
            return super().reset(specifications)
        self.active.clear(); self.started_turn.clear(); self.bank.clear()
        CurriculumShard.reset(self, specifications)


def reader(spec, device='cuda', workers=6):
    """Explicit read-only, cross-source architecture-compatible policy import."""
    path = Path(spec['checkpoint']); meta = read_json(path.with_suffix('.json'))
    if digest(path) != spec['sha256'] or meta['sha256'] != spec['sha256'] or meta['source_id'] != spec['source_id']:
        raise ValueError('reader parent hash/source mismatch')
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved['source_id'] != spec['source_id'] or saved['version'] not in (136, 137):
        raise ValueError('V136/V137 parent required')
    if saved['version'] == 136:
        from boardbench.v136.training import Trainer
    else:
        from .training import Trainer
    t = Trainer({**saved['config'], 'workers': workers, 'record_games_per_batch': 128}, device)
    t.model.load_state_dict(saved['model']); t.manager.load_state_dict(saved['manager'])
    return t


def own_states(trainer, environment_count=64, offset=0):
    seeds = trainer.train_seeds[offset:offset+environment_count]
    if len(seeds) != environment_count:
        raise ValueError('insufficient training seeds for own-state panel')
    with trainer.make_pool(len(seeds), min(trainer.config['workers'], len(seeds))) as pool:
        _, metrics = trainer.collect(pool, seeds, greedy=True, keep_rows=True)
    traces = trainer.last_trajectories
    if len(traces) != len(seeds):
        raise ValueError('all own-state episodes must be recorded')
    rows = []
    for trace in traces:
        env = FastHarmonies(); obs, _ = env.reset(trace['environment_seed'])
        points, strata = set(trace['planning_steps']), {}
        for i, action in enumerate(trace['actions']):
            if i in points and len(candidates(obs)) > 1:
                phase = min(3, obs['turn_index']//4)
                strata.setdefault(phase, []).append({'environment_seed': trace['environment_seed'],
                    'decision': i, 'observation': obs, 'phase': phase})
            obs = env.step(decode(obs, action, 'slots240_v1', False)).observation
        if not obs['terminated']:
            raise ValueError('own-state trajectory did not end naturally')
        for phase, options in sorted(strata.items()):
            key = lambda r: hashlib.sha256(f"v137:{r['environment_seed']}:{r['decision']}".encode()).hexdigest()
            rows.append(min(options, key=key))
    if len({r['environment_seed'] for r in rows}) < 4:
        raise ValueError('too few own-state environments')
    return rows, traces, metrics


def branch_batch(trainer, states, specifications, timeout=180, pool=None):
    """Forced root option, then this policy's actual planner to natural terminal."""
    reset = [{'observation': states[s['state']]['observation'], 'future_seed': s['future_seed'],
              'environment_seed': states[s['state']]['environment_seed'], 'source': 'v137_public_counterfactual'}
             for s in specifications]
    tick = time.monotonic()
    context = nullcontext(pool) if pool is not None else EnvironmentPool(len(reset), min(trainer.config['workers'], len(reset)),
                         native_artifact=trainer.config['native_artifact'], arm=trainer.config['arm'],
                         horizon=trainer.config['plan_horizon'], shard_class=PublicPlanShard)
    with context as pool:
        arrays = pool.reset(reset); first = np.ones(len(reset), bool); steps = 0
        while not arrays['done'].all():
            if time.monotonic()-tick > timeout:
                raise TimeoutError('counterfactual batch deadline')
            slots = np.flatnonzero(~arrays['done']); extra = pool.extra
            selected = extra['plan'][slots].copy()
            indices = np.flatnonzero(extra['request'][slots]); requesting = slots[indices]
            if len(requesting):
                bank, mask = extra['plans'][requesting].copy(), extra['plan_mask'][requesting].copy()
                if trainer.config['arm'] == 'manager':
                    with torch.no_grad():
                        logits, _ = trainer.manager(torch.as_tensor(arrays['features'][requesting], device=trainer.device, dtype=torch.float64),
                            torch.as_tensor(bank, device=trainer.device, dtype=torch.float64), torch.as_tensor(mask, device=trainer.device))
                        choices = logits.argmax(-1).cpu().numpy()
                else:
                    choices = np.where(mask[:, 1], 1, 0)
                for j, slot in enumerate(requesting):
                    if first[slot]:
                        choices[j] = specifications[slot]['candidate']; first[slot] = False
                if not mask[np.arange(len(choices)), choices].all():
                    raise ValueError('forced root candidate not legal')
                extra['choice'][requesting] = choices
                selected[indices] = bank[np.arange(len(choices)), choices]
            with torch.no_grad():
                logits, _ = trainer.model(torch.as_tensor(np.c_[arrays['features'][slots], selected], device=trainer.device, dtype=torch.float64),
                                           torch.as_tensor(arrays['masks'][slots, :240], device=trainer.device))
                action = logits.argmax(-1).cpu().numpy()
            pool.step(slots, action); steps += len(slots)
        stats = arrays['stats'].copy()
    if stats[:, STOPPED].any() or not stats[:, NATURAL].all():
        raise RuntimeError('only natural terminal scores can label goals')
    return [{**s, 'score': float(row[SCORE]), 'animal_score': float(row[ANIMAL]), 'natural_terminal': True}
            for s, row in zip(specifications, stats)], steps


def interval(values):
    values = np.asarray(values, float)
    mean = float(values.mean()); half = float(1.96*values.std(ddof=1)/len(values)**.5) if len(values)>1 else None
    return {'n_environments': len(values), 'mean': mean,
            'descriptive95': [mean-half, mean+half] if half is not None else None}


def diagnostic(data):
    q, mask, seeds = data['q_fit'], data['mask'], data['seed']
    choice = np.where(mask, q.mean(-1), -np.inf).argmax(-1)
    baseline = np.where(mask[:, 1], 1, 0)
    ids = np.arange(len(seeds)); audit = data['q_verify'].mean(-1)
    # Exact ties are not evidence to abandon the existing option.
    choice = np.where(q.mean(-1)[ids, choice] > q.mean(-1)[ids, baseline], choice, baseline)
    spread = np.max(np.where(mask, q.mean(-1), -np.inf), -1)-np.min(np.where(mask, q.mean(-1), np.inf), -1)
    gains = audit[ids, choice]-audit[ids, baseline]
    env = [float(gains[seeds == s].mean()) for s in sorted(set(seeds.tolist()))]
    return {'independent_future_gain': interval(env), 'selected_nonbaseline_fraction': float((choice != baseline).mean()),
            'identical_candidate_mean_fraction': float((spread == 0).mean()),
            'mean_candidate_score_spread': float(spread.mean()),
            'fit_selected_gain': float((q.mean(-1)[ids, choice]-q.mean(-1)[ids, baseline]).mean()),
            'meaning': 'root option selected using fitting futures, evaluated on independent futures; not an optimality bound or full-policy score'}


def build(spec, out, settings, device='cuda'):
    if (not 4 <= settings['environments'] <= 128 or settings['offset'] < 0 or settings['workers'] < 0
            or min(settings['fit_futures'], settings['verify_futures'], settings['cohort']) < 1
            or settings['fit_futures']+settings['verify_futures'] > 100):
        raise ValueError('bounded own-state sampling settings required')
    out = Path(out)
    if (out/'dataset.json').exists():
        old = read_json(out/'dataset.json')
        if old['parent'] != spec or old['settings'] != settings or digest(old['path']) != old['sha256']:
            raise ValueError('immutable goal labels changed')
        return old
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); t = reader(spec, device, settings['workers'])
    try:
        states, traces, metrics = own_states(t, settings['environments'], settings['offset'])
        state_spec = {'parent': spec, 'settings': settings, 'states': states}
        if (out/'states.json').exists() and read_json(out/'states.json') != state_spec:
            raise ValueError('replayed student states changed')
        write_json(out/'states.json', state_spec)
        (out/'trajectories.json.gz').write_bytes(gzip.compress(json.dumps(traces, allow_nan=False).encode(), mtime=0))
        banks = [candidates(r['observation']) for r in states]
        observations = [r['observation'] for r in states]
        plans, mask = np.zeros((len(states), K, PLAN_DIM), np.float32), np.zeros((len(states), K), bool)
        for i, (obs, bank) in enumerate(zip(observations, banks)):
            for j, plan in enumerate(bank):
                plans[i, j] = encode_plan(obs, plan); mask[i, j] = True
        specs = []
        for i, bank in enumerate(banks):
            for repeat in range(settings['fit_futures']+settings['verify_futures']):
                for j in range(len(bank)):
                    specs.append({'state': i, 'candidate': j, 'repeat': repeat,
                                  'future_seed': settings['future_seed'] + i*100 + repeat})
        results, actions = [], 0
        state_sha = digest(out/'states.json')
        cohort = min(settings['cohort'], len(specs))
        with EnvironmentPool(cohort, min(settings['workers'], cohort), native_artifact=t.config['native_artifact'],
                             arm=t.config['arm'], horizon=t.config['plan_horizon'], shard_class=PublicPlanShard) as pool:
            for start in range(0, len(specs), cohort):
                part = specs[start:start+cohort]; path = out/'batches'/f'{start:06d}.json'
                if path.exists():
                    batch = read_json(path)
                    if batch['specifications'] != part or batch['states_sha256'] != state_sha:
                        raise ValueError('changed counterfactual batch')
                else:
                    rows, count = branch_batch(t, states, part, pool=pool if len(part)==cohort else None)
                    batch = {'specifications': part, 'rows': rows, 'actions': count, 'states_sha256': state_sha}
                    write_json(path, batch)
                if (len(batch['rows']) != len(part) or any(
                        any(row[k] != expected[k] for k in expected) or not row['natural_terminal']
                        or not np.isfinite(row['score']) for row, expected in zip(batch['rows'], part))):
                    raise ValueError('incomplete or mismatched counterfactual results')
                results.extend(batch['rows']); actions += batch['actions']
                write_json(out/'progress.json', {'completed_branches': len(results), 'total_branches': len(specs),
                           'elapsed_seconds': time.monotonic()-started, 'updated_unix': time.time()})
        q = np.zeros((len(states), K, settings['fit_futures']+settings['verify_futures']), np.float32)
        for row in results:
            q[row['state'], row['candidate'], row['repeat']] = row['score']
        seeds = np.asarray([r['environment_seed'] for r in states])
        environments = sorted(set(seeds.tolist()), key=lambda s: hashlib.sha256(f'v137-hold-{s}'.encode()).hexdigest())
        held = set(environments[:max(1, len(environments)//8)])
        data = {'x': NativeEncoder.from_artifact(t.config['native_artifact']).encode(pack(observations)),
                'plans': plans, 'mask': mask, 'q_fit': q[:, :, :settings['fit_futures']],
                'q_verify': q[:, :, settings['fit_futures']:], 'seed': seeds,
                'validation': np.asarray([s in held for s in seeds]),
                'start_score': np.asarray([o['score_breakdown']['total'] for o in observations], np.float32)}
        with (out/'goals.partial').open('wb') as stream:
            np.savez_compressed(stream, **data)
        (out/'goals.partial').replace(out/'goals.npz')
        info = {'path': str((out/'goals.npz').resolve()), 'sha256': digest(out/'goals.npz'), 'parent': spec, 'settings': settings,
                'states': len(states), 'environments': len(environments), 'branches': len(results), 'actions': actions,
                'collection': metrics, 'diagnostic': diagnostic(data), 'elapsed_seconds': time.monotonic()-started,
                'continuation': 'current worker and goal selector, greedy deployment policy; conditional public future',
                'verification_futures_used_for_training': False}
        write_json(out/'dataset.json', info)
        return info
    finally:
        t.close()


def load(spec, allowed):
    if digest(spec['path']) != spec['sha256']:
        raise ValueError('goal dataset checksum mismatch')
    with np.load(spec['path'], allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    n = len(data['seed']); hold = data['validation']
    if not set(data['seed'].tolist()) <= set(allowed) or set(data['seed'][hold]) & set(data['seed'][~hold]):
        raise ValueError('goal dataset split leakage')
    if (hold.dtype != np.bool_ or data['mask'].dtype != np.bool_ or not hold.any() or hold.all()
            or data['x'].shape != (n, 1346) or data['plans'].shape != (n, K, PLAN_DIM)
            or data['mask'].shape != (n, K) or not data['mask'][:, 0].all()
            or data['q_fit'].shape[:2] != (n, K) or data['q_verify'].shape[:2] != (n, K)
            or min(data['q_fit'].shape[-1], data['q_verify'].shape[-1]) < 1
            or any(not np.isfinite(v).all() for v in data.values())):
        raise ValueError('goal dataset shape/mask/finite failure')
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True); p.add_argument('--out', required=True); p.add_argument('--device', default='cuda')
    a = p.parse_args(); job = read_json(a.job)
    result = build(job['parent'], a.out, job['settings'], a.device)
    print(json.dumps({k: result[k] for k in ('states', 'branches', 'elapsed_seconds', 'diagnostic')}), flush=True)
