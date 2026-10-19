"""Public-state continuations; no rule, reward, or simulator modifications."""
from collections import OrderedDict
import hashlib
import json

from boardbench.environments.harmonies import COLORS
from boardbench.solvers.board import ranked_actions
from boardbench.v123.policies import Continuation, available_tokens, card_goals, rank_actions
from .compiled import card_goals_compiled


class GoalContinuation:
    """Memoize exactly the inputs read by card_goals, independent of history.

    Goal objects are read-only in the original ranker. Resource-aware adjustments
    use copies and explicitly change the heuristic, not game legality.
    """
    def __init__(self, resource_aware=False, capacity=512, compiled=False):
        self.resource_aware = resource_aware
        self.capacity = capacity
        self.cache = OrderedDict()
        self.actions = OrderedDict()
        self.calls = self.hits = self.goal_hits = 0
        self.goal_function = card_goals_compiled if compiled else card_goals

    def goals(self, obs, card_id, placed_count=0):
        supply = available_tokens(obs)
        board = tuple((cell['stack_id'], cell['animal'] is not None) for cell in obs['board'])
        key = (board, tuple(supply[c] for c in COLORS), card_id, placed_count)
        if key in self.cache:
            self.goal_hits += 1
            self.cache.move_to_end(key)
            goals = self.cache[key]
        else:
            goals = self.goal_function(obs, card_id, placed_count)
            self.cache[key] = goals
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        if not self.resource_aware:
            return goals
        # Only roughly a third of the public future bag can actually be selected.
        # This is a planning discount, never an assertion of impossibility.
        accessible = sum(obs['pending_tokens'].values()) + (0 if obs['bundle_taken'] else 3)
        accessible += 3 * (sum(obs['remaining_tokens'].values()) // 9)
        adjusted = []
        for goal in goals:
            g = dict(goal)
            need = g['missing']
            discount = min(1., accessible / max(1, need))
            # Scarce required colours and very long projects are less actionable.
            scarcity = sum(max(0., n - obs['pending_tokens'].get(c, 0)) /
                           max(1., supply[c] / 3) for c, n in g['need'].items())
            g['value'] *= discount / (1 + .12 * scarcity)
            adjusted.append(g)
        return sorted(adjusted, key=lambda g: -g['value'])

    def rank(self, obs):
        scores = rank_actions(obs, 'habitat', goal_function=self.goals)
        if not self.resource_aware:
            return scores
        base = {action_key(a): v for v, a in ranked_actions(obs)}
        # Preserve a larger fraction of terrain incentives when very little
        # construction time remains; do not globally multiply animal reward.
        turns = sum(obs['remaining_tokens'].values()) // 9
        animal_fraction = .75 if turns >= 3 else .45
        return sorted(((base[action_key(a)] + animal_fraction * (v - base[action_key(a)]), a)
                       for v, a in scores), key=lambda pair: -pair[0])

    def decide(self, obs):
        self.calls += 1
        key = hashlib.sha256(json.dumps(obs, sort_keys=True, separators=(',', ':')).encode()).digest()
        if key in self.actions:
            self.hits += 1
            self.actions.move_to_end(key)
            return dict(self.actions[key])
        action = dict(self.rank(obs)[0][1])
        self.actions[key] = action
        if len(self.actions) > 32768:
            self.actions.popitem(last=False)
        return dict(action)


def action_key(action):
    return tuple(sorted(action.items()))


def mixed_candidates(obs, width=6):
    """Interleave terrain and habitat recommendations; old top1 stays fallback."""
    original = ranked_actions(obs)
    habitat = rank_actions(obs, 'habitat')
    merged, seen = [], set()
    for left, right in zip(original, habitat):
        for score, action in (left, right):
            key = action_key(action)
            if key not in seen:
                merged.append((score, action))
                seen.add(key)
            if len(merged) >= width:
                return merged
    return merged


class ComplementaryContinuation:
    """Combine existing players, either per rollout round or by public game phase."""
    def __init__(self, mode):
        if mode not in ('portfolio','finish_terrain'):
            raise ValueError('unknown complementary mode')
        self.mode=mode
        self.original=Continuation('original',capacity=32768)
        self.habitat=GoalContinuation(compiled=True)
        self.round=0
        self.calls=self.hits=0

    def select_round(self, number):
        self.round=number

    def decide(self, obs):
        if self.mode=='portfolio':
            use_original=self.round%2==0
        else:
            use_original=(sum(obs['remaining_tokens'].values())//9<=3 or
                          sum(c['stack_id']==0 for c in obs['board'])<=4)
        policy=self.original if use_original else self.habitat
        hits=policy.hits
        action=policy.decide(obs)
        self.calls+=1;self.hits+=policy.hits-hits
        return action
