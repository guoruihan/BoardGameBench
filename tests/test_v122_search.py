from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from boardbench.environments import TASKS
from boardbench.environments.adapters import SearchAccess
from boardbench.solvers.board import BoardSolver, ranked_actions
from boardbench.v122.episode import run_episode
from boardbench.v122.search import CachedHeuristic, TerminalRollout, short_parameters


@pytest.fixture(scope='module')
def public_states():
    engine = TASKS['harmonies'].factory()
    obs, _ = engine.reset(1751)
    states = []
    while not obs['terminated']:
        states.append(deepcopy(obs))
        obs = engine.step(ranked_actions(obs)[0][1]).observation
    return states


def context(obs, counter):
    return SimpleNamespace(reference_simulator=SearchAccess(
        lambda seed: TASKS['harmonies'].factory.from_observation(obs, seed),
        lambda: counter.append(1)))


def test_old_configuration_is_exact_and_scale_consumes_work(public_states):
    assert short_parameters(1) == dict(method='search', width=6, rollouts=4,
        depth=2, depth_unit='turn', max_simulation_steps=384, leaf='potential')
    # Identical root, before the first future event: fixed candidates, more samples.
    obs = public_states[-5]
    measurements = []
    for scale in (1, 4):
        calls = []
        solver = BoardSolver(19, **short_parameters(scale))
        action = solver.decide(obs, context(obs, calls))
        assert action in obs['legal_actions']
        assert len(calls) == solver.last_search['simulation_steps'] <= 384 * scale
        measurements.append(solver.last_search)
    assert measurements[1]['rollouts'] == measurements[0]['rollouts'] * 4
    assert measurements[1]['simulation_steps'] > measurements[0]['simulation_steps']


def test_terminal_public_only_exact_restore_and_cache(public_states, tmp_path):
    obs = public_states[-4]
    before = deepcopy(obs)
    solver = TerminalRollout(19, width=3, rollouts=2)
    uncached = TerminalRollout(19, width=3, rollouts=2, cache_capacity=0)
    calls, other = [], []
    action = solver.decide(obs, context(obs, calls))
    assert action == uncached.decide(obs, context(obs, other))
    assert obs == before and len(calls) == len(other)
    assert solver.last_search['terminal_rollouts'] == solver.last_search['rollouts']
    assert solver.last_search['simulation_steps'] == len(calls)
    assert action in obs['legal_actions']
    solver.save(tmp_path)
    restored = TerminalRollout(0)
    restored.load(tmp_path)
    assert restored.decide(obs, context(obs, [])) == solver.decide(obs, context(obs, []))
    assert restored.last_search['candidates'] == solver.last_search['candidates']


def test_terminal_bound_never_substitutes_partial_score(public_states):
    obs = public_states[0]
    solver = TerminalRollout(19, width=1, rollouts=1, branch_cap=1)
    with pytest.raises(RuntimeError, match='no leaf substituted'):
        solver.decide(obs, context(obs, []))


def test_public_cache_matches_greedy_and_does_not_alias(public_states):
    cache = CachedHeuristic(3)
    for obs in public_states:
        expected = ranked_actions(obs)[0][1]
        first = cache.decide(obs)
        first['mutated'] = True
        assert cache.decide(obs) == expected
        assert len(cache.cache) <= 3
    assert cache.hits == len(public_states)


def test_public_equivalence_ignores_real_hidden_supply(public_states):
    obs = public_states[-8]
    a = TASKS['harmonies'].factory.from_observation(obs, 11)
    b = TASKS['harmonies'].factory.from_observation(obs, 12)
    assert a.observe() == b.observe()
    left, right = TerminalRollout(91, width=2, rollouts=2), TerminalRollout(91, width=2, rollouts=2)
    assert left.decide(a.observe(), context(a.observe(), [])) == right.decide(b.observe(), context(b.observe(), []))
    assert left.last_search['candidates'] == right.last_search['candidates']


def test_full_attempt_and_failure_retained(tmp_path):
    job = {'id': 'test', 'environment_seed': 1751, 'method': 'heuristic',
           'multiplier': 1, 'search_seed': 90210}
    complete = run_episode(job, tmp_path / 'complete')
    assert complete['status'] == 'completed' and complete['score'] == complete['partial_score']
    assert complete['simulation_steps'] == 0 and complete['natural_terminal']
    failed = run_episode(job, tmp_path / 'failed', max_decisions=1)
    assert failed['status'] == 'failed' and failed['score'] is None
    assert (tmp_path / 'failed/result.json').exists()
