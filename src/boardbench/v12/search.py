"""A solver-owned time allocation rule; the runtime does not choose a method."""
import time

from boardbench.solvers.board import BoardSolver, ranked_actions


class TimedSearch(BoardSolver):
    def __init__(self, seed, max_simulation_steps=384):
        super().__init__(seed, method='search', width=6, rollouts=2, depth=2,
                         depth_unit='turn', leaf='potential', max_simulation_steps=max_simulation_steps)
        self.ceiling = max_simulation_steps
        self.rate = 400.

    def decide(self, observation, context):
        if not hasattr(context, 'remaining_seconds'):
            self.max_simulation_steps = self.ceiling
            return super().decide(observation, context)
        left = context.remaining_seconds()
        if observation['task_id'] == 'harmonies':
            future = sum(observation['remaining_tokens'].values()) // 9
            decisions = 6 * future + sum(observation['pending_tokens'].values()) + 8
        else:
            decisions = (9 if observation['task_id'] == 'micro_tiles' else 19) - observation['placed_count']
        allowance = left / max(1, decisions) * .6
        budget = min(self.ceiling, int(allowance * self.rate))
        if budget < self.width * self.rollouts:
            return ranked_actions(observation)[0][1]
        self.max_simulation_steps = budget
        tick = time.monotonic()
        action = super().decide(observation, context)
        spent = time.monotonic() - tick
        if spent:
            self.rate = .7 * self.rate + .3 * self.last_search['simulation_steps'] / spent
        return action

    def save(self, directory):
        from boardbench.artifacts.store import write_json
        from pathlib import Path
        super().save(directory)
        write_json(Path(directory) / 'timed_search.json', {'rate': self.rate, 'ceiling': self.ceiling})

    def load(self, directory):
        from boardbench.artifacts.store import read_json
        from pathlib import Path
        super().load(directory)
        state = read_json(Path(directory) / 'timed_search.json')
        self.rate, self.ceiling = state['rate'], state['ceiling']
