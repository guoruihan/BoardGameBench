"""Public-observation policies and auditable matches for original two-player games."""
from copy import deepcopy
import hashlib
import json
import math
import random
import time

from .base import FEATURE_DIM
from .registry import game_class


def action_features(cls, observation, actions):
    batch = getattr(cls, 'features_batch', None)
    rows = batch(observation, actions) if batch else [cls.features(observation, a) for a in actions]
    if len(rows) != len(actions) or any(len(r) != FEATURE_DIM or
            not all(math.isfinite(float(x)) for x in r) for r in rows):
        raise ValueError('Invalid public candidate features')
    return rows


def policy_seed(seed, seat):
    digest = hashlib.sha256(f'policy:{seed}:{seat}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')


class FeaturePolicy:
    """No engine reference or snapshot is supplied to a policy."""
    def __init__(self, cls, spec, seed=0):
        self.cls, self.spec, self.rng = cls, deepcopy(spec), random.Random(seed)
        for key, expected in (('game', cls.game_id), ('config', cls.config_id), ('rules_version', cls.rules_version)):
            if key in spec and spec[key] != expected:
                raise ValueError('Policy game/config mismatch')
        self.weights = spec.get('weights', cls.default_weights)
        if len(self.weights) != FEATURE_DIM or not all(math.isfinite(float(w)) for w in self.weights):
            raise ValueError('Expected 24 finite weights')
        if spec.get('kind', 'heuristic') not in ('random', 'heuristic', 'linear'):
            raise ValueError('Unsupported feature policy')

    def decide(self, observation, actions):
        if self.spec.get('kind') == 'random':
            return self.rng.choice(actions)
        rows = action_features(self.cls, observation, actions)
        values = [sum(x * w for x, w in zip(row, self.weights)) for row in rows]
        best = max(values)
        return actions[self.rng.choice([i for i, v in enumerate(values) if v == best])]


def make_policy(cls, spec, seed=0):
    if spec.get('kind') == 'neural':
        from .neural import NeuralPolicy
        return NeuralPolicy(cls, spec, seed)
    return FeaturePolicy(cls, spec, seed)


def run_match(game_id, specs, seed=0, first_player=0, max_decisions=10000,
              max_seconds=300., record=False, validate=False, policies=None):
    """Caps are incomplete runs, never house-rule endings or fabricated rewards."""
    if len(specs) != 2:
        raise ValueError('Exactly two policies required')
    cls = game_class(game_id)
    env = cls()
    env.reset(seed=seed, first_player=first_player)
    policies = policies or [make_policy(cls, spec, policy_seed(seed, p))
                            for p, spec in enumerate(specs)]
    began = time.monotonic()
    trace = []
    count = 0
    status = None
    while not env.outcome()['terminated']:
        outcome = env.outcome()
        if outcome.get('unresolved'):
            status = 'unresolved_rule'
            break
        if count >= max_decisions or time.monotonic() - began >= max_seconds:
            status = 'truncated'
            break
        seat = env.current_player
        observation = env.observe(seat)
        actions = env.legal_actions(seat)
        if not actions:
            status = 'unresolved_rule' if observation.get('blocked_reason') else 'no_legal_action'
            break
        # Defend engine-owned legal dictionaries against accidental policy mutation.
        action = policies[seat].decide(observation, deepcopy(actions))
        if record:
            trace.append(dict(player=seat, observation=deepcopy(observation),
                              action=deepcopy(action)))
        env.step(action)
        if validate:
            env.validate()
        count += 1
    outcome = env.outcome()
    status = status or ('complete' if outcome['terminated'] else 'incomplete')
    rewards = None
    if status == 'complete':
        rewards = [0., 0.] if len(outcome['winners']) != 1 else [
            1. if p in outcome['winners'] else -1. for p in (0, 1)]
    result = dict(game=game_id, config=cls.config_id, rules_version=cls.rules_version,
                  seed=seed, first_player=first_player, policies=specs, status=status,
                  decisions=count, seconds=time.monotonic()-began, outcome=outcome,
                  score_breakdown=env.score_breakdown(), rewards=rewards)
    if record:
        result['trajectory'] = trace
    return result


def summarize(results, candidate_seats=None):
    """Never silently drop incomplete games from win-rate denominators."""
    candidate_seats = candidate_seats or [0] * len(results)
    if len(candidate_seats) != len(results):
        raise ValueError('Seat/result alignment error')
    completed = [(r, p) for r, p in zip(results, candidate_seats) if r['status'] == 'complete']
    points = [(.5 + .5*r['rewards'][p]) for r, p in completed]
    statuses = {s: sum(r['status'] == s for r in results) for s in sorted({r['status'] for r in results})}
    return dict(games=len(results), completed=len(completed), statuses=statuses,
                win_points=sum(points), completed_win_rate=sum(points)/len(points) if points else None,
                all_scheduled_win_rate=sum(points)/len(results) if results else None,
                mean_seconds=sum(r['seconds'] for r in results)/len(results) if results else None)


def write_json(path, data):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    temporary.replace(path)
