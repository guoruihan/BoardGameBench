from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from boardbench.environments.harmonies import Harmonies
from boardbench.environments.adapters import SearchAccess
from boardbench.v121.registry import make_splits
from boardbench.v123.policies import Continuation, rank_actions, card_goals
from boardbench.v124.search import TimedRollout
from boardbench.v125.campaign import prepare_plan, make_jobs
from boardbench.v125.episode import run_episode
from boardbench.v125.policies import GoalContinuation, mixed_candidates, action_key
from boardbench.v125.search import Search, VARIANTS


class Clock:
    def __init__(self, step=.001):
        self.now, self.step = 0., step

    def __call__(self):
        self.now += self.step
        return self.now


@pytest.fixture(scope='module')
def states():
    rows = []
    for seed in (7, 113, 1751):
        e = Harmonies(); obs, _ = e.reset(seed)
        p = Continuation('habitat')
        while not obs['terminated']:
            rows.append(obs)
            obs = e.step(p.decide(obs)).observation
    return rows


def context(obs, calls):
    return SimpleNamespace(reference_simulator=SearchAccess(
        lambda seed: Harmonies.from_observation(obs, seed), lambda: calls.append(1)))


def test_goal_cache_equivalent_and_context_sensitive(states):
    fast = GoalContinuation()
    for obs in states:
        original = deepcopy(obs)
        assert fast.rank(obs) == rank_actions(obs, 'habitat')
        assert fast.decide(obs) == fast.decide(deepcopy(obs))
        for c in obs['active_cards']:
            assert fast.goals(obs, c['card_id'], c['placed_count']) == card_goals(obs, c['card_id'], c['placed_count'])
        assert obs == original
    assert fast.hits > 0 and fast.goal_hits > 0
    assert len(fast.cache) <= fast.capacity


@pytest.mark.parametrize('variant', VARIANTS)
def test_legal_public_budget_balanced_samples(states, variant):
    obs = states[-5]; before = deepcopy(obs)
    p = Search(11, variant, 10, clock=Clock())
    calls = []
    action = p.decide(obs, context(obs, calls))
    s = p.last_search
    assert action in obs['legal_actions'] and obs == before
    assert s['simulation_steps'] == len(calls)
    assert {c['samples'] for c in s['candidates']} == {s['completed_rounds']}
    assert s['accepted_terminal_rollouts'] + s['discarded_terminal_rollouts'] == s['terminal_rollouts']
    assert all(c['sample_variance'] is None or c['sample_variance'] >= 0 for c in s['candidates'])
    p.spent = 20
    calls = []
    p.decide(obs, context(obs, calls))
    assert not calls and p.last_search['completed_rounds'] == 0
    assert all(c['mean'] is None for c in p.last_search['candidates'])


@pytest.mark.parametrize('variant', ['original', 'habitat'])
def test_old_baseline_fixed_clock_exact(states, variant):
    for obs in states[-8:-3]:
        old = TimedRollout(11, variant, 10, clock=Clock())
        new = Search(11, variant, 10, clock=Clock())
        assert old.decide(obs, context(obs, [])) == new.decide(obs, context(obs, []))
        for k in ('completed_rounds','simulation_steps','terminal_rollouts','allowance_seconds','solver_seconds'):
            assert old.last_search[k] == new.last_search[k]
        assert [c['mean'] for c in old.last_search['candidates']] == [c['mean'] for c in new.last_search['candidates']]


def test_mixed_roots_no_duplicates_legal_and_baseline_fallback(states):
    from boardbench.solvers.board import ranked_actions
    for obs in states[::5]:
        candidates = mixed_candidates(obs)
        assert len(candidates) == min(6, len(obs['legal_actions']))
        assert len({action_key(a) for _, a in candidates}) == len(candidates)
        assert candidates[0][1] == ranked_actions(obs)[0][1]
        assert all(a in obs['legal_actions'] for _, a in candidates)


def raw_plan():
    return {'split': 'development', 'environment_indices': [96,97,98,99], 'panels':[4],
            'search_seeds':[12520261020], 'references':['original','habitat'],
            'arms':[{'id':v,'method':'terminal','multiplier':1,'timeout_seconds':1020,
                     'params':{'mode':'timed','continuation':v,'seconds':900}} for v in VARIANTS]}


def test_plan_seed_pairing_and_caps():
    raw = raw_plan(); manifest = make_splits(train_count=8)
    plan = prepare_plan(raw, manifest); jobs = make_jobs(plan)
    assert len(jobs) == 20
    assert plan['environment_seeds'] == manifest['sets']['development'][96:100]
    pairs = {a['id']:{(j['environment_seed'],j['search_seed']) for j in jobs if j['arm']==a['id']} for a in raw['arms']}
    assert all(p==pairs['original'] for p in pairs.values())
    with pytest.raises(ValueError, match='no confirmation/test'):
        prepare_plan({**raw,'split':'test'}, manifest)
    raw['arms'][0]['params']['seconds'] = 901
    with pytest.raises(ValueError, match='900-second'):
        prepare_plan(raw,manifest)


def test_episode_learning_alignment_and_true_return(tmp_path):
    job = {'id':'small','method':'terminal','multiplier':1,'environment_seed':7,'search_seed':11,
           'params':{'mode':'timed','continuation':'habitat_fast','seconds':.05}}
    result = run_episode(job,tmp_path/'episode')
    assert result['status']=='completed' and result['natural_terminal']
    rows=[json.loads(x) for x in (tmp_path/'episode/decisions.jsonl').read_text().splitlines()]
    assert sum(r['raw_score_delta'] for r in rows)==result['score']
    for i,row in enumerate(rows):
        assert row['action'] in row['observation_before']['legal_actions']
        assert row['raw_score_delta']==row['observation']['score_breakdown']['total']-row['observation_before']['score_breakdown']['total']
        if i:
            assert row['observation_before']==rows[i-1]['observation']
    assert rows[-1]['observation']['terminated']
