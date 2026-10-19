from collections import Counter
from copy import deepcopy
import json

import pytest

from boardbench.artifacts.store import read_json
from boardbench.environments.harmonies import Harmonies
from boardbench.solvers.board import ranked_actions
from boardbench.v121.registry import make_splits
from boardbench.v123.bank import object_hash, select_state, validate_bank, validate_public
from boardbench.v123.diagnostic import rollout, run_state, sample_seeds, select_roots
from boardbench.v123.policies import (POLICIES, Continuation, available_tokens,
                                      compatible_goals, rank_actions, reachable_pattern)
from boardbench.v123.campaign import prepare_plan, make_jobs, incomplete
from boardbench.v123.report import component_effects, effects, publish, summarize


@pytest.fixture(scope='module')
def states():
    engine = Harmonies()
    obs, _ = engine.reset(make_splits(train_count=8)['sets']['train'][0])
    result = []
    while not obs['terminated']:
        result.append(obs)
        obs = engine.step(ranked_actions(obs)[0][1]).observation
    return result


def test_reachability_aggregate_supply_locks_and_shared_support():
    board = [{'stack_id': 0, 'animal': None} for _ in range(23)]
    pattern = ((0, (2,)), (1, (2,)))
    assert reachable_pattern(board, 0, pattern, Counter(yellow=1)) is None
    goal = reachable_pattern(board, 0, pattern, Counter(yellow=2))
    assert goal['missing'] == 2 and goal['need'] == {'yellow': 2}
    board[1] = {'stack_id': 2, 'animal': 7}
    assert reachable_pattern(board, 0, pattern, Counter(yellow=1))['missing'] == 1
    board[0]['animal'] = 9
    assert reachable_pattern(board, 0, pattern, Counter(yellow=1)) is None
    board[0] = {'stack_id': 1, 'animal': None}  # water cannot become field
    assert reachable_pattern(board, 0, pattern, Counter(yellow=9)) is None
    board[0] = {'stack_id': 6, 'animal': 9}  # locked brown cannot become tree
    assert reachable_pattern(board, 1, ((0, (9,)),), Counter(green=9)) is None


def test_building_alternatives_and_joint_resource_accounting(states):
    obs = deepcopy(states[0])
    goal = reachable_pattern(obs['board'], 0, ((0, (12, 13, 14)),), Counter(brown=1, red=1))
    assert goal['targets'][0] == 13
    obs['remaining_tokens'] = {'blue': 1}
    obs['bundle_taken'] = True
    obs['pending_tokens'] = {}
    first = dict(card_id=1, anchor=0, targets={2: 1}, value=3, missing=1)
    second = dict(card_id=2, anchor=1, targets={2: 1}, value=2, missing=1)
    assert len(compatible_goals(obs, {1: [first], 2: [second]})) == 2  # shared support counted once
    second['targets'] = {3: 1}
    assert len(compatible_goals(obs, {1: [first], 2: [second]})) == 1
    second['targets'], second['anchor'] = {2: 1}, 0
    assert len(compatible_goals(obs, {1: [first], 2: [second]})) == 1


def test_inaccessible_market_bundles_not_counted(states):
    obs = deepcopy(states[0])
    before = available_tokens(obs)
    obs['bundle_taken'] = True
    assert sum(before.values()) - sum(available_tokens(obs).values()) == 9


def test_unreachable_card_is_not_preferred_to_ending_turn(states, monkeypatch):
    obs = deepcopy(states[0])
    engine = Harmonies.from_observation(obs, 0)
    engine.s.update(bundle_taken=True, pending_tokens=dict.fromkeys(obs['pending_tokens'], 0))
    obs = engine.observe()
    monkeypatch.setattr('boardbench.v123.policies.card_goals', lambda *args: [])
    assert rank_actions(obs, 'habitat')[0][1] == {'type': 'end_turn'}


@pytest.mark.parametrize('name', POLICIES)
def test_policy_is_legal_public_deterministic_and_cache_exact(states, name):
    policy, uncached = Continuation(name, 3), Continuation(name, 0)
    for obs in states[::4]:
        before = deepcopy(obs)
        action = policy.decide(obs)
        assert action in obs['legal_actions'] and action == uncached.decide(obs)
        action['mutated'] = True
        assert policy.decide(obs) == uncached.decide(obs)
        assert obs == before and len(policy.cache) <= 3
        if name == 'original':
            assert policy.decide(obs) == ranked_actions(obs)[0][1]


@pytest.mark.parametrize('name', POLICIES)
def test_full_legal_terminal_and_public_future_equivalence(states, name):
    obs = states[12]
    a, b = Harmonies.from_observation(obs, 4), Harmonies.from_observation(obs, 5)
    assert a.observe() == b.observe()
    action = ranked_actions(obs)[0][1]
    left = rollout(a.observe(), action, Continuation(name), 20)
    right = rollout(b.observe(), action, Continuation(name), 20)
    assert left == right and left['natural_terminal']
    assert left['score'] == left['terrain'] + left['animals']
    with pytest.raises(RuntimeError, match='no partial leaf'):
        rollout(states[0], ranked_actions(states[0])[0][1], Continuation(name), 20, branch_cap=1)


def test_public_bank_rejects_hidden_or_derived_corruption_and_selection_is_fixed(states):
    validate_public(states[0])
    corrupt = {**states[0], '_bag': ['blue']}
    with pytest.raises(ValueError, match='canonical public'):
        validate_public(corrupt)
    with pytest.raises(ValueError, match='legality or score'):
        validate_public({**states[0], 'legal_actions': []})
    index, obs = select_state(states, 'middle', 'place_token')
    assert 'place_token' in {a['type'] for a in obs['legal_actions']}
    assert select_state(deepcopy(states), 'middle', 'place_token') == (index, obs)


def make_bank(obs, environments=1):
    manifest = make_splits(train_count=8)
    bank = {'split': 'train_calibration', 'split_sha256': manifest['sha256'], 'environments': environments,
            'states': [{'id': f'e{i}_s{j}', 'environment_index': i,
                        'environment_seed': manifest['sets']['train'][i], 'producer': 'test',
                        'phase': 'late', 'observation': obs, 'observation_sha256': object_hash(obs)}
                       for i in range(environments) for j in range(2)]}
    bank['sha256'] = object_hash(bank)
    return bank, manifest


def make_plan(bank, manifest, **overrides):
    return prepare_plan({'cpu_cores': 2, 'width': 6, 'discovery_samples': 2, 'confirmation_samples': 3,
                         'seed': 121, 'run_seconds': 60, 'job_timeout_seconds': 50,
                         'panels': [bank['environments']], **overrides}, bank, manifest)


def test_split_and_bank_integrity(states):
    bank, manifest = make_bank(states[-4])
    validate_bank(bank, manifest)
    changed = deepcopy(bank)
    changed['states'][0]['observation']['score'] = 999
    with pytest.raises(ValueError, match='digest'):
        validate_bank(changed, manifest)
    changed = {**bank, 'split': 'test'}
    changed['sha256'] = object_hash({k: v for k, v in changed.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='no confirmation/test'):
        validate_bank(changed, manifest)


def test_all_actions_independent_seeds_committed_selection_and_failure(states, tmp_path):
    bank, manifest = make_bank(states[-3])
    plan = make_plan(bank, manifest)
    job = make_jobs(plan, bank)[0]
    result = run_state(job, tmp_path / 'complete')
    assert result['status'] == 'completed', result['error']
    selection = read_json(tmp_path / 'complete/selection.json')
    assert not set(selection['seeds']['discovery']) & set(selection['seeds']['confirmation'])
    assert selection['choices'] == result['choices']
    samples = [json.loads(line) for line in (tmp_path / 'complete/rollouts.jsonl').read_text().splitlines()]
    for policy in POLICIES:
        rows = result['discovery'][policy]
        assert len(rows) == len(job['observation']['legal_actions'])
        assert {r['rank'] for r in result['confirmation'][policy]} == set(selection['roots'])
        assert all(s['natural_terminal'] for s in samples)
        assert all(len(r['scores']) == plan['confirmation_samples'] for r in result['confirmation'][policy])
    assert result['simulation_steps'] == sum(s['steps'] for s in samples)
    assert result['terminal_rollouts'] == len(samples)
    components = component_effects(result)
    assert 'space_minus_original:terrain' in components
    fixed_gap = effects(result)['fixed_root:space_minus_original']
    assert components['space_minus_original:terrain'] + components['space_minus_original:animals'] == pytest.approx(fixed_gap)
    # Domain separation is stable across process/policy order.
    assert sample_seeds(121, job['id'], 'discovery', 2) == selection['seeds']['discovery']
    bad = {**job, 'observation': states[0], 'observation_sha256': object_hash(states[0]), 'branch_cap': 1}
    failure = run_state(bad, tmp_path / 'failed')
    assert failure['status'] == 'failed' and 'no partial leaf' in failure['error']
    assert (tmp_path / 'failed/result.json').exists()


def synthetic(job, shift=0):
    discovery = {p: [{'rank': rank, 'mean': score, 'action': {'type': 'place_token'}}
                     for rank, score in enumerate([0, 0, 0, 0, 0, 1, 3])] for p in POLICIES}
    choices, roots = select_roots(discovery, 6)
    # Discovery chose outside rank6; independent confirmation can be WORSE.
    confirmation = {p: [{'rank': rank, 'mean': (2 if rank == 5 else -1) + shift,
                          'scores': [(2 if rank == 5 else -1) + shift] * 3} for rank in roots] for p in POLICIES}
    return {**{k: v for k, v in job.items() if k != 'observation'}, 'status': 'completed', 'legal_count': 7,
            'choices': choices, 'discovery': discovery, 'confirmation': confirmation}


def test_no_confirmation_reselection_no_pseudoreplication_or_partial_panel(states, tmp_path):
    bank, manifest = make_bank(states[-3], environments=2)
    plan = make_plan(bank, manifest)
    jobs = make_jobs(plan, bank)
    rows = [synthetic(j, shift=i * 10) for i, j in enumerate(jobs)]
    summary = publish(tmp_path, plan, bank, jobs, rows, {'phase': 'completed'})
    gap = summary['effects']['original:all_minus_top6']
    assert gap['n_environments'] == 2 and gap['mean'] == -3
    assert gap['normal_95_ci'] == [-3, -3]
    assert effects(rows[0])['space:preference_interaction'] == 0
    assert summarize(plan, bank, jobs, rows[:-1])['balanced_environment_panel'] == 0
    fail = incomplete(jobs[-1], tmp_path / 'missing', 'job_timeout')
    partial = summarize(plan, bank, jobs, rows[:-1] + [fail])
    assert partial['failed_or_censored'] == 1 and partial['balanced_environment_panel'] == 0
    assert (tmp_path / 'REPORT.md').exists() and (tmp_path / 'states.csv').exists()
    assert 'train_calibration' in (tmp_path / 'REPORT.md').read_text()
    assert '不用于方法优劣结论' in (tmp_path / 'REPORT.md').read_text()
    with pytest.raises(ValueError, match='duplicate'):
        summarize(plan, bank, jobs, rows + [rows[0]])
    corrupt = deepcopy(rows)
    corrupt[0]['confirmation']['space'] = corrupt[0]['confirmation']['space'][:-1]
    with pytest.raises(ValueError, match='different confirmation roots'):
        summarize(plan, bank, jobs, corrupt)
