"""Paired conditional-public futures: test execution, then label goal values."""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from boardbench.artifacts.store import digest, read_json
from boardbench.v12.encoding import decode
from boardbench.v128.environment import FastHarmonies
from boardbench.v129.encoding import NativeEncoder, pack
from boardbench.v132.data import load as load_corpus
from boardbench.v134.plans import K, PLAN_DIM, achieved, candidates, encode_plan, refresh_goal
from .data import segments


def continuations(trainer, specifications, timeout=240):
    """The prescribed first goal lasts <=2 turns; then the same heuristic planner.

    Every branch resamples from PUBLIC state. Matched future seeds pair randomness
    but cannot force identical draws after policies diverge in card consumption.
    No future stream or outcome is an input to the worker.
    """
    started = time.monotonic()
    games = [FastHarmonies.from_observation(s['observation'], s['future_seed']) for s in specifications]
    observations = [g.observe() for g in games]
    if any(obs != spec['observation'] for obs, spec in zip(observations, specifications)):
        raise ValueError('invalid public probe state')
    plans = [deepcopy(s['plan']) for s in specifications]
    turns = [obs['turn_index'] for obs in observations]
    initial_turns = list(turns)
    hit = [False] * len(games)
    first = [True] * len(games)
    actions = 0
    encoder = NativeEncoder.from_artifact(trainer.config['native_artifact'])
    for _ in range(256):
        if time.monotonic() - started > timeout:
            raise TimeoutError('bounded goal probe exhausted')
        slots = [i for i, obs in enumerate(observations) if not obs['terminated']]
        if not slots:
            break
        vectors = []
        for i in slots:
            obs, plan = observations[i], plans[i]
            age = obs['turn_index'] - turns[i]
            if age >= (2 if plan else 1) or any(refresh_goal(obs, g) is None for g in plan):
                first[i] = False
                bank = candidates(obs)
                plans[i] = bank[1] if len(bank) > 1 else []
                turns[i] = obs['turn_index']
                plan, age = plans[i], 0
            vectors.append(encode_plan(obs, plan, age))
        live = [observations[i] for i in slots]
        base = encoder.encode(pack(live))
        masks = np.zeros((len(slots), 240), bool)
        from boardbench.v12.encoding import action_map
        for j, obs in enumerate(live):
            masks[j, list(action_map(obs, 'slots240_v1', False))] = True
        with torch.no_grad():
            logits, _ = trainer.model(torch.as_tensor(np.c_[base, np.asarray(vectors)], device=trainer.device, dtype=torch.float64),
                                      torch.as_tensor(masks, device=trainer.device))
            choices = logits.argmax(-1).cpu().tolist()
        for i, choice in zip(slots, choices):
            observations[i] = games[i].step(decode(observations[i], choice, 'slots240_v1', False)).observation
            target = specifications[i].get('target')
            # Same target outcome measured in free/heuristic and prescribed branches.
            if target and observations[i]['turn_index'] - initial_turns[i] < 2 and achieved(observations[i], target):
                hit[i] = True
        actions += len(slots)
    if any(not obs['terminated'] for obs in observations):
        raise RuntimeError('probe requires natural terminal, no truncated score')
    return {'rows': [{**{k: s[k] for k in ('environment_seed', 'future_seed', 'branch', 'state')},
                      'score': obs['score'], 'animal_score': obs['score_breakdown']['animals'],
                      'target_completed': bool(success), 'natural_terminal': True}
                     for s, obs, success in zip(specifications, observations, hit)],
            'actions': actions, 'elapsed_seconds': time.monotonic() - started}


def execution_probe(trainer, max_states=24, futures=2):
    spec = trainer.config['skill_dataset']
    if digest(spec['probes']) != spec['probes_sha256']:
        raise ValueError('probe state hash mismatch')
    selected, seen = [], set()
    for row in read_json(spec['probes']):
        obs = row['observation']
        goal = {**row['goal'], 'targets': {int(k): v for k, v in row['goal']['targets'].items()}}
        updated = refresh_goal(obs, goal)
        if (row['environment_seed'] in seen or row['environment_seed'] not in trainer.train_seeds
                or updated is None or not 1 <= updated['missing'] <= 4):
            continue
        seen.add(row['environment_seed'])
        selected.append({**row, 'goal': goal})
        if len(selected) == max_states:
            break
    if len(selected) < 2:
        raise ValueError('at least two feasible probe environments required')
    specifications = []
    for state, row in enumerate(selected):
        bank = candidates(row['observation'])
        for repeat in range(futures):
            for branch, plan in (('target', [row['goal']]), ('free', []),
                                 ('heuristic', bank[1] if len(bank) > 1 else [])):
                specifications.append({'environment_seed': row['environment_seed'], 'state': state,
                    'future_seed': 136000000 + 100 * state + repeat, 'branch': branch,
                    'observation': row['observation'], 'plan': plan, 'target': row['goal']})
    result = continuations(trainer, specifications)
    result['means'] = {branch: {key: float(np.mean([r[key] for r in result['rows'] if r['branch'] == branch]))
                       for key in ('score', 'animal_score', 'target_completed')} for branch in ('target', 'free', 'heuristic')}
    result.update(scope='training states withheld from skill BC, not independent policy performance',
                  states=len(selected), futures=futures, selection='first eligible environment; no outcome selection',
                  definition='target completion within two turns; total score at natural terminal')
    return result


def execution_gate(control, skills):
    """Exploration gate, deliberately not a statistical superiority claim."""
    c, s = control['means'], skills['means']
    ids = lambda r: [(v['environment_seed'], v['state'], v['future_seed'], v['branch']) for v in r['rows']]
    if ids(control) != ids(skills):
        raise ValueError('execution gate pairing mismatch')
    gain = s['target']['target_completed'] - c['target']['target_completed']
    conditioned = s['target']['target_completed'] - s['free']['target_completed']
    total = s['target']['score'] - c['target']['score']
    return {'passed': bool(gain > 0 and conditioned > 0 and total >= -1),
            'completion_gain': gain, 'goal_conditioning_gain': conditioned, 'target_total_gain': total,
            'rule': 'completion gain >0, conditioned >free, target total >= control-1; training diagnostic only'}


def bootstrap_manager(trainer):
    """Goal labels use this trained worker, never expert continuation or test states."""
    corpus = load_corpus(trainer.config['skill_corpus'], set(trainer.train_seeds))
    held = set(trainer.skills['seed'][trainer.skills['validation']].tolist())
    selected, seen = [], set()
    for row in segments(corpus):
        seed = row['environment_seed']
        if seed in seen or seed in held:
            continue
        seen.add(seed); selected.append(row)
        if len(selected) == trainer.config['manager_label_states']:
            break
    if len(selected) < 2:
        raise ValueError('insufficient manager training states')
    specifications, banks = [], []
    for state, row in enumerate(selected):
        obs = row['observations'][0]
        bank = candidates(obs)
        banks.append(bank)
        for repeat in range(2):
            for branch, plan in enumerate(bank):
                specifications.append({'state': state, 'environment_seed': row['environment_seed'],
                    'future_seed': 136100000 + 100 * state + repeat, 'branch': branch,
                    'observation': obs, 'plan': plan})
    result = continuations(trainer, specifications, timeout=360)
    encoder = NativeEncoder.from_artifact(trainer.config['native_artifact'])
    obs = [r['observations'][0] for r in selected]
    x = torch.as_tensor(encoder.encode(pack(obs)), device=trainer.device, dtype=torch.float64)
    plans, mask, q = np.zeros((len(obs), K, PLAN_DIM)), np.zeros((len(obs), K), bool), np.zeros((len(obs), K))
    for i, (o, bank) in enumerate(zip(obs, banks)):
        for j, plan in enumerate(bank):
            plans[i, j] = encode_plan(o, plan); mask[i, j] = True
            q[i, j] = np.mean([r['score'] for r in result['rows'] if r['state'] == i and r['branch'] == j])
    plans, mask, q = (torch.as_tensor(v, device=trainer.device) for v in (plans, mask, q))
    with torch.no_grad():
        prior, _ = trainer.manager(x, plans, mask)
        # Conservative policy improvement target: reweight existing prior with true Q/5.
        target = (prior + q / 5.).masked_fill(~mask, -1e9).softmax(-1)
    losses = []
    for _ in range(trainer.config['manager_label_updates']):
        logits, _ = trainer.manager(x, plans, mask)
        loss = F.kl_div(logits.log_softmax(-1), target, reduction='batchmean')
        trainer.manager_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.manager.parameters(), .5, error_if_nonfinite=True)
        trainer.manager_optimizer.step(); losses.append(float(loss.detach()))
    return {**result, 'states': len(obs), 'fit_updates': len(losses), 'first_loss': losses[0], 'last_loss': losses[-1],
            'public_observations': obs, 'goal_values': q.cpu().tolist(),
            'target': 'current-worker two-future true terminal score, prior reweighted by exp(Q/5)',
            'limits': 'small noisy training-only initialization; manager then learns from real on-policy returns'}
