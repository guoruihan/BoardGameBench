"""900-second cumulative-budget search variants with terminal target logging.

The time allocation and random-future streams deliberately match V1.2.4.
Only complete balanced candidate rounds contribute to Q estimates.
"""
from boardbench.solvers.board import ranked_actions
from boardbench.v124.search import TimedRollout
from .policies import GoalContinuation, mixed_candidates

VARIANTS = ('original', 'habitat', 'habitat_fast', 'mixed_roots', 'resource_habitat')


class Search(TimedRollout):
    def __init__(self, seed, variant, seconds, **kwargs):
        if variant not in VARIANTS:
            raise ValueError('unknown v125 variant')
        super().__init__(seed, 'original' if variant == 'original' else 'habitat', seconds, **kwargs)
        self.variant = variant
        if variant in ('habitat_fast', 'mixed_roots', 'resource_habitat'):
            self.policy = GoalContinuation(resource_aware=variant == 'resource_habitat')

    def _decide(self, observation, context, begin):
        if observation['task_id'] != 'harmonies' or context.reference_simulator is None:
            raise ValueError('public Harmonies simulator required')
        ranked = (mixed_candidates(observation, self.width) if self.variant == 'mixed_roots'
                  else ranked_actions(observation))
        if not ranked:
            raise ValueError('terminal state has no action')
        actions = [a for _, a in ranked[:self.width]]
        remaining = max(0., self.seconds - self.spent)
        allowance = remaining * min(1., 2 / max(2., self.horizon))
        deadline = begin + allowance
        sums, squares = [0.] * len(actions), [0.] * len(actions)
        rounds = steps = terminals = attempted = terminal_steps = 0
        interrupted = False
        before_calls, before_hits = self.policy.calls, self.policy.hits
        while len(actions) > 1 and self.clock() < deadline:
            seed, returns = self.rng.getrandbits(63), {}
            offset = rounds % len(actions)
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
                squares[rank] += value * value
            rounds += 1
        selected = max(range(len(actions)), key=lambda i: sums[i]) if rounds else 0
        self.last_search = {
            'variant': self.variant, 'candidate_count': len(actions),
            'legal_count': len(observation['legal_actions']), 'selected_rank': selected,
            'completed_rounds': rounds, 'accepted_terminal_rollouts': rounds * len(actions),
            'terminal_rollouts': terminals, 'rollouts': attempted, 'simulation_steps': steps,
            'horizon_completed_rollouts': terminals, 'discarded_terminal_rollouts': terminals - rounds * len(actions),
            'interrupted_round': interrupted, 'fallback': rounds == 0 and len(actions) > 1,
            'forced_action': len(actions) == 1, 'allowance_seconds': allowance,
            'estimated_remaining_actions': self.horizon,
            'cache_calls': self.policy.calls - before_calls, 'cache_hits': self.policy.hits - before_hits,
            'candidates': [{'rank': i, 'action': a, 'samples': rounds,
                            'mean': sums[i] / rounds if rounds else None,
                            'sample_variance': max(0., (squares[i] - sums[i] ** 2 / rounds) / (rounds - 1))
                            if rounds > 1 else None} for i, a in enumerate(actions)],
        }
        estimate = .2 * self.horizon + .8 * terminal_steps / terminals if terminals else self.horizon
        self.horizon = max(1., estimate - 1)
        return actions[selected]
