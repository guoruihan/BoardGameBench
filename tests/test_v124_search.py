from copy import deepcopy
from types import SimpleNamespace

import pytest

from boardbench.environments.harmonies import Harmonies
from boardbench.environments.adapters import SearchAccess
from boardbench.solvers.board import ranked_actions
from boardbench.v124.search import TimedRollout, Greedy
from boardbench.v124.episode import make_policy, run_episode
from boardbench.v122.search import TerminalRollout


@pytest.fixture(scope='module')
def states():
    engine = Harmonies()
    obs, _ = engine.reset(1751)
    rows = []
    while not obs['terminated']:
        rows.append(obs)
        obs = engine.step(ranked_actions(obs)[0][1]).observation
    return rows


class Clock:
    def __init__(self, step=.001):
        self.now, self.step = 0., step

    def __call__(self):
        self.now += self.step
        return self.now


def context(obs, calls):
    return SimpleNamespace(reference_simulator=SearchAccess(
        lambda seed: Harmonies.from_observation(obs, seed), lambda: calls.append(1)))


@pytest.mark.parametrize('name', ['original', 'space', 'habitat'])
def test_common_roots_natural_samples_no_observation_mutation(states, name):
    obs = states[-5]
    frozen = deepcopy(obs)
    solver = TimedRollout(11, name, 10, clock=Clock())
    calls = []
    action = solver.decide(obs, context(obs, calls))
    assert action in obs['legal_actions'] and obs == frozen
    stats = solver.last_search
    assert stats['simulation_steps'] == len(calls)
    assert [c['action'] for c in stats['candidates']] == [a for _, a in ranked_actions(obs)[:6]]
    assert {c['samples'] for c in stats['candidates']} == {stats['completed_rounds']}
    assert stats['terminal_rollouts'] >= stats['accepted_terminal_rollouts']
    assert stats['rollouts'] >= stats['terminal_rollouts']
    assert stats['accepted_terminal_rollouts'] + stats['discarded_terminal_rollouts'] == stats['terminal_rollouts']


def test_no_fake_leaf_incomplete_round_or_expired_budget(states):
    obs = states[0]
    solver = TimedRollout(11, 'original', .08, clock=Clock())
    calls = []
    assert solver.decide(obs, context(obs, calls)) == ranked_actions(obs)[0][1]
    assert solver.last_search['completed_rounds'] == 0
    assert all(c['mean'] is None for c in solver.last_search['candidates'])
    assert solver.last_search['fallback']
    solver.spent = 1.
    calls = []
    solver.decide(obs, context(obs, calls))
    assert not calls


def test_safety_bound_fails_instead_of_substituting_leaf(states):
    solver = TimedRollout(11, 'original', 10, branch_cap=1, clock=Clock())
    with pytest.raises(RuntimeError, match='no leaf'):
        solver.decide(states[0], context(states[0], []))


def test_exact_legacy_control_and_public_equivalence(states):
    legacy = make_policy('terminal', 16, 12, mode='fixed', continuation='original')
    assert isinstance(legacy, TerminalRollout) and legacy.rollouts == 64 and legacy.width == 6
    obs = states[-5]
    a, b = Harmonies.from_observation(obs, 11), Harmonies.from_observation(obs, 12)
    left = TimedRollout(11, 'habitat', 1, clock=Clock())
    right = TimedRollout(11, 'habitat', 1, clock=Clock())
    assert left.decide(a.observe(), context(a.observe(), [])) == right.decide(b.observe(), context(b.observe(), []))
    assert left.last_search == right.last_search


def test_greedy_original_exact_and_full_game_metrics(states, tmp_path):
    solver = Greedy('original')
    for obs in states:
        assert solver.decide(obs, None) == ranked_actions(obs)[0][1]
    for name in ['original', 'space', 'habitat']:
        job = {'id': name, 'method': 'heuristic', 'multiplier': 1, 'environment_seed': 1751,
               'search_seed': 11, 'params': {'mode': 'greedy', 'continuation': name}}
        result = run_episode(job, tmp_path / name)
        assert result['status'] == 'completed' and result['natural_terminal']
        assert result['score'] == result['terminal_metrics']['terrain'] + result['terminal_metrics']['animals']
        assert result['terminal_metrics']['tokens'] % 3 == 0
        assert result['simulation_steps'] == 0
    job.update(method='terminal', params={'mode': 'timed', 'continuation': 'habitat', 'seconds': .1})
    result = run_episode(job, tmp_path / 'timed')
    assert result['status'] == 'completed'
    assert result['solver_budget_utilization'] > 0 and result['decision_seconds'] < 1
    fail = run_episode(job, tmp_path / 'failed', max_decisions=1)
    assert fail['status'] == 'failed' and fail['score'] is None


def test_only_complete_round_used_and_partial_returns_discarded(monkeypatch):
    import boardbench.v124.search as module
    monkeypatch.setattr(module, 'ranked_actions', lambda obs: [(1, {'id': 0}), (0, {'id': 1})])
    clock = Clock(step=0)
    counts = [0]
    class Branch:
        def step(self, action):
            counts[0] += 1
            # First round prefers rank1 (10 vs20); an incomplete next round
            # has an arbitrarily large score but cannot alter that result.
            values = [10, 20, -1000]
            clock.now += .03
            return SimpleNamespace(terminated=True, observation={'score': values[counts[0] - 1]})
    ctx = SimpleNamespace(reference_simulator=SimpleNamespace(fork=lambda seed: Branch()))
    solver = TimedRollout(11, 'original', .64, clock=clock)
    solver.horizon = 16.  # .08 allowance
    assert solver.decide({'task_id': 'harmonies'}, ctx) == {'id': 1}
    assert solver.last_search['completed_rounds'] == 1
    assert solver.last_search['discarded_terminal_rollouts'] == 1
    assert [c['mean'] for c in solver.last_search['candidates']] == [10, 20]
