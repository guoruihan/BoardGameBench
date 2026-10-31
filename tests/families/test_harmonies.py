import json

from boardbench.environments.harmonies_duo import FastHarmoniesDuo
from boardbench.families.games.harmonies import Harmonies


def test_adapter_preserves_original_transitions_and_private_replay():
    old, new = FastHarmoniesDuo(), Harmonies()
    old.reset(123, 1)
    new.reset(123, 1)
    for _ in range(30):
        assert old.legal_actions() == new.legal_actions()
        action = old.legal_actions()[0]
        old.step(action)
        new.step(action)
        assert old.save_state() == new.engine.save_state()
    snap = json.loads(json.dumps(new.snapshot()))
    restored = Harmonies()
    restored.restore(snap)
    assert restored.observe() == new.observe()
    assert 'private' not in restored.observe()
    assert '_deck' not in restored.observe()


def test_public_features_batch_matches_single():
    env = Harmonies()
    obs = env.reset(2)
    actions = env.legal_actions()
    assert env.features_batch(obs, actions) == [env.features(obs, a) for a in actions]
