"""Common-budget terminal search; only complete common-sample rounds count.

The budget is accumulated solver wall time, not game engine / replay I/O time.
Each non-forced decision receives 2/H of remaining time, where H estimates
remaining actions from previous public rollouts. Linear-in-horizon allocation
accounts for longer early simulations without starving later moves. Exact legacy
sample-budget search is a separate control, not mislabeled as this algorithm.
"""
import math
import random
import time

from boardbench.contracts import derive_seed
from boardbench.solvers.board import ranked_actions
from boardbench.v123.policies import Continuation


class Greedy:
    def __init__(self, continuation):
        self.policy = Continuation(continuation, capacity=32768)
        self.last_search = {}

    def decide(self, observation, context):
        action = self.policy.decide(observation)
        self.last_search = {'candidate_count': 1, 'simulation_steps': 0}
        return action


class TimedRollout:
    def __init__(self, seed, continuation, seconds, *, width=6, branch_cap=256,
                 cache_capacity=32768, clock=None):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('finite positive time budget required')
        if type(width) is not int or width != 6 or type(branch_cap) is not int or branch_cap < 1:
            raise ValueError('top6 and positive branch safety bound required')
        self.seed, self.decisions = seed, 0
        self.policy = Continuation(continuation, capacity=cache_capacity)
        self.seconds, self.spent, self.horizon = seconds, 0., 64.
        self.width, self.branch_cap = width, branch_cap
        self.clock = clock or time.monotonic
        self.last_search = {}

    def decide(self, observation, context):
        begin = self.clock()
        self.rng = random.Random(derive_seed(self.seed, f'v124:decision:{self.decisions}'))
        self.decisions += 1
        try:
            return self._decide(observation, context, begin)
        finally:
            elapsed = self.clock() - begin
            self.spent += elapsed
            self.last_search.update(solver_seconds=elapsed, cumulative_solver_seconds=self.spent,
                                    budget_seconds=self.seconds)

    def _decide(self, observation, context, begin):
        if observation['task_id'] != 'harmonies' or context.reference_simulator is None:
            raise ValueError('public Harmonies simulator required')
        # Root ranking and fallback never change with the continuation policy.
        ranked = ranked_actions(observation)
        if not ranked:
            raise ValueError('terminal state has no action')
        actions = [a for _, a in ranked[:self.width]]
        remaining = max(0., self.seconds - self.spent)
        fraction = min(1., 2 / max(2., self.horizon))
        allowance = remaining * fraction
        deadline = begin + allowance
        sums = [0.] * len(actions)
        rounds = steps = terminals = attempted = terminal_steps = 0
        interrupted = False
        before_calls, before_hits = self.policy.calls, self.policy.hits
        # Even an incomplete last round is charged and logged, but never used to
        # compare candidates with unequal sample counts or fake partial scores.
        while len(actions) > 1 and self.clock() < deadline:
            seed, returns = self.rng.getrandbits(63), {}
            offset = rounds % len(actions)  # rotate wasted partial-round work
            for k in range(len(actions)):
                rank = (offset + k) % len(actions)
                if self.clock() >= deadline:
                    interrupted = True
                    break
                branch = context.reference_simulator.fork(seed)
                action = actions[rank]
                attempted += 1
                for branch_step in range(self.branch_cap):
                    if self.clock() >= deadline:
                        interrupted = True
                        break
                    result = branch.step(action)
                    steps += 1
                    if result.terminated:
                        terminals += 1
                        terminal_steps += branch_step + 1
                        returns[rank] = result.observation['score']
                        break
                    if self.clock() >= deadline:
                        interrupted = True
                        break
                    action = self.policy.decide(result.observation)
                else:
                    raise RuntimeError('terminal rollout exceeded safety bound; no leaf substituted')
                if interrupted:
                    break
            if len(returns) != len(actions):
                break
            for rank, value in returns.items():
                sums[rank] += value
            rounds += 1
        selected = max(range(len(actions)), key=lambda i: sums[i]) if rounds else 0
        self.last_search = {
            'candidate_count': len(actions), 'legal_count': len(ranked), 'selected_rank': selected,
            'completed_rounds': rounds, 'accepted_terminal_rollouts': rounds * len(actions),
            'terminal_rollouts': terminals, 'rollouts': attempted, 'simulation_steps': steps,
            'horizon_completed_rollouts': terminals, 'discarded_terminal_rollouts': terminals - rounds * len(actions),
            'interrupted_round': interrupted, 'fallback': rounds == 0 and len(actions) > 1,
            'forced_action': len(actions) == 1, 'allowance_seconds': allowance,
            'estimated_remaining_actions': self.horizon,
            'cache_calls': self.policy.calls - before_calls, 'cache_hits': self.policy.hits - before_hits,
            'candidates': [{'rank': i, 'action': a, 'samples': rounds,
                            'mean': sums[i] / rounds if rounds else None} for i, a in enumerate(actions)],
        }
        # Only public simulated lengths; never use the real hidden game length.
        # Smooth the estimate, decrement for the action we are about to commit.
        estimate = .2 * self.horizon + .8 * terminal_steps / terminals if terminals else self.horizon
        self.horizon = max(1., estimate - 1)
        return actions[selected]
