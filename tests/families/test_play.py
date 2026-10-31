import json

import pytest

from boardbench.families.play import FeaturePolicy, run_match, summarize
from boardbench.families.registry import available_games, game_class


def test_seeded_match_and_terminal_rewards():
    specs = [{'kind': 'heuristic'}, {'kind': 'random'}]
    a = run_match('jaipur', specs, seed=80, record=True, validate=True)
    b = run_match('jaipur', specs, seed=80, record=True, validate=True)
    assert a['status'] == 'complete'
    for key in ('outcome', 'trajectory', 'decisions', 'rewards'):
        assert a[key] == b[key]
    assert sum(a['rewards']) == 0
    assert all('rng' not in step['observation'] for step in a['trajectory'])
    json.dumps(a, allow_nan=False)


def test_caps_are_incomplete_without_rewards():
    result = run_match('jaipur', [{'kind': 'random'}]*2, max_decisions=1)
    assert result['status'] == 'truncated'
    assert result['outcome']['terminated'] is False
    assert result['rewards'] is None
    summary = summarize([result])
    assert summary['completed_win_rate'] is None
    assert summary['all_scheduled_win_rate'] == 0
    assert summary['statuses'] == {'truncated': 1}


def test_policy_rejects_nonfinite_and_registry_does_not_alias():
    with pytest.raises(ValueError):
        FeaturePolicy(game_class('jaipur'), {'weights': [float('nan')]*24})
    with pytest.raises(ValueError):
        game_class('invented')
    assert 'jaipur' in available_games()


def test_summary_preserves_seats_and_incomplete_denominator():
    rows = [{'status': 'complete', 'rewards': [-1, 1], 'seconds': 1},
            {'status': 'truncated', 'rewards': None, 'seconds': 2}]
    summary = summarize(rows, [1, 0])
    assert summary['completed_win_rate'] == 1
    assert summary['all_scheduled_win_rate'] == .5
    with pytest.raises(ValueError):
        summarize(rows, [0])


@pytest.mark.parametrize('flag', ['unresolved', 'unresolved_rule'])
def test_unresolved_original_rule_never_becomes_a_draw(monkeypatch, flag):
    from boardbench.families.base import DuoGame
    from boardbench.families import play
    class Blocked(DuoGame):
        game_id = 'blocked_test'
        def _reset(self, first_player):
            pass
        def _observe(self, player):
            return {}
        def scores(self):
            return [0, 0]
        def outcome(self):
            return {**super().outcome(), flag: True}
        def _legal_actions(self):
            raise AssertionError('An unresolved game must not ask the policy to move')
    monkeypatch.setattr(play, 'game_class', lambda _: Blocked)
    result = run_match('blocked_test', [{'kind': 'random'}]*2)
    assert result['status'] == 'unresolved_rule'
    assert result['rewards'] is None
    assert not result['outcome']['terminated']
    assert result['outcome']['winners'] == []
