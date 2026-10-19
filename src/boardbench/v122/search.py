"""Public-information heuristic rollout to natural terminal.

The old short-horizon search is deliberately unchanged. This solver changes only
the horizon/leaf, keeping its root ranking, common random numbers and tie order.
Finite sampling and root pruning do not provide a policy-improvement guarantee.
"""
from collections import OrderedDict
import hashlib
import json
import random

from boardbench.artifacts.store import read_json, write_json
from boardbench.contracts import Solver
from boardbench.environments.board import tuples
from boardbench.solvers.board import ranked_actions


class CachedHeuristic:
    """Cache public-observation -> greedy action, never future-dependent values.

    Full public keys intentionally favor safety over maximum reuse. Copies prevent
    callers mutating cached actions. The cache changes runtime, not decisions.
    """
    def __init__(self, capacity=32768):
        self.capacity = int(capacity)
        if self.capacity < 0:
            raise ValueError('negative cache capacity')
        self.cache = OrderedDict()
        self.calls = self.hits = 0

    def decide(self, obs):
        self.calls += 1
        key = hashlib.sha256(json.dumps(obs, sort_keys=True, separators=(',', ':')).encode()).digest()
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return dict(self.cache[key])
        action = ranked_actions(obs)[0][1]
        if self.capacity:
            self.cache[key] = dict(action)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return dict(action)


class TerminalRollout(Solver):
    def __init__(self, seed, width=6, rollouts=4, branch_cap=256, cache_capacity=32768):
        self.width, self.rollouts, self.branch_cap = int(width), int(rollouts), int(branch_cap)
        if min(self.width, self.rollouts, self.branch_cap) < 1:
            raise ValueError('positive search budgets required')
        self.rng = random.Random(seed)
        self.policy = CachedHeuristic(cache_capacity)
        self.last_search = {}

    def decide(self, observation, context):
        if observation['task_id'] != 'harmonies':
            raise ValueError('terminal-rollout diagnostic is registered for Harmonies only')
        if context.reference_simulator is None:
            raise ValueError('public reference simulator required')
        ranked = ranked_actions(observation)
        if not ranked:
            raise ValueError('terminal observation has no action')
        seeds = [self.rng.getrandbits(63) for _ in range(self.rollouts)]
        choices, candidates = [], []
        used = completed = 0
        before_calls, before_hits = self.policy.calls, self.policy.hits
        for rank, (_, action) in enumerate(ranked[:self.width]):
            returns = []
            for seed in seeds:
                branch = context.reference_simulator.fork(seed)
                next_action = action
                for _ in range(self.branch_cap):
                    result = branch.step(next_action)
                    used += 1
                    if result.terminated:
                        break
                    # Neither a sampled private future nor its RNG/seed is given
                    # to the continuation policy; it sees the then-public state.
                    next_action = self.policy.decide(result.observation)
                else:
                    raise RuntimeError('terminal rollout exceeded safety bound; no leaf substituted')
                returns.append(result.observation['score'])
                completed += 1
            mean = sum(returns) / len(returns)
            candidates.append({'rank': rank, 'action': action, 'mean': mean,
                               'samples': len(returns),
                               'sample_variance': (sum((x - mean)**2 for x in returns) / (len(returns) - 1)
                                                   if len(returns) > 1 else None)})
            choices.append((mean, action))
        selected = max(range(len(choices)), key=lambda i: choices[i][0])
        self.last_search = {'simulation_steps': used, 'rollouts': completed,
                            'horizon_completed_rollouts': completed, 'terminal_rollouts': completed,
                            'depth_unit': 'terminal', 'candidate_count': len(choices),
                            'legal_count': len(ranked), 'selected_rank': selected,
                            'candidates': candidates, 'cache_calls': self.policy.calls - before_calls,
                            'cache_hits': self.policy.hits - before_hits}
        return choices[selected][1]

    def save(self, directory):
        write_json(directory / 'terminal_rollout.json', {
            'width': self.width, 'rollouts': self.rollouts, 'branch_cap': self.branch_cap,
            'cache_capacity': self.policy.capacity, 'rng': self.rng.getstate()})

    def load(self, directory):
        state = read_json(directory / 'terminal_rollout.json')
        rng = state.pop('rng')
        self.__init__(0, **state)
        self.rng.setstate(tuples(rng))


def short_parameters(multiplier):
    """Historical selected V1.1 teacher, scaled in SAMPLES and proportional cap."""
    if type(multiplier) is not int or multiplier < 1:
        raise ValueError('positive integer multiplier required')
    return {'method': 'search', 'width': 6, 'rollouts': 4 * multiplier,
            'depth': 2, 'depth_unit': 'turn', 'max_simulation_steps': 384 * multiplier,
            'leaf': 'potential'}
