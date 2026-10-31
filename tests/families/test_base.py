import json
import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.base import DuoGame


class Tiny(DuoGame):
    game_id = "contract_test"

    def _reset(self, first_player):
        self.state.update(points=[0, 0], hidden=self.rng.randrange(100))

    def _observe(self, player):
        return {"points": list(self.state["points"])}

    def _legal_actions(self):
        return [{"take": 1}, {"take": 2}]

    def scores(self):
        return self.state["points"]

    def _step(self, action):
        self.state["points"][self.current_player] += self.rng.randint(1, 6)
        if action["take"] == 2:
            raise ValueError("failed effect must roll back RNG and state")
        self.finish()


def test_failed_and_illegal_actions_preserve_state_and_rng():
    game = Tiny(); game.reset(73)
    before = game.snapshot()
    for action in ({"take": True}, {"take": 1.0}, {"take": 3}, {"take": 2}):
        with pytest.raises((InvalidAction, ValueError)):
            game.step(action)
        assert game.snapshot() == before


def test_json_replay_terminal_rewards_and_private_boundary():
    game = Tiny(); observation = game.reset(19, 1)
    assert "hidden" not in observation and "rng" not in observation
    restored = Tiny(); restored.restore(json.loads(json.dumps(game.snapshot())))
    result = game.step({"take": 1})
    assert restored.step({"take": 1}) == result
    assert game.snapshot() == restored.snapshot()
    assert result["rewards"] == [-1., 1.]
    assert game.current_actor() is None and game.legal_actions() == []
    before = restored.snapshot()
    malformed = game.snapshot(); malformed["state"]["current_player"] = 7
    with pytest.raises(ValueError):
        restored.restore(malformed)
    assert restored.snapshot() == before


def test_machine_readable_coverage_is_separate_from_installation():
    from boardbench.families.registry import available_games, game_class
    assert len(available_games()) == 17
    jaipur = game_class('jaipur').describe()
    five_tribes = game_class('five_tribes').describe()
    assert jaipur['coverage_status'] == 'known_unresolved_edges'
    assert jaipur['known_unresolved_edges'] == ['unresolved_round_tie']
    assert five_tribes['coverage_status'] == 'research_only'
    assert len(five_tribes['known_unresolved_edges']) == 2
    jaipur['known_unresolved_edges'].clear()
    assert game_class('jaipur').describe()['known_unresolved_edges']
