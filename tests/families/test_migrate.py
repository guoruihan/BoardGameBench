import hashlib
import json

import pytest
import torch

from boardbench.families.migrate import migrate_policy
from boardbench.families.neural import CandidateActorCritic, load_model, save_model
from boardbench.families.registry import game_class


@pytest.mark.parametrize('game', ['splendor', 'gizmos'])
def test_rule_migration_preserves_weights_and_source(tmp_path, game):
    cls = game_class(game)
    old = type('Old', (cls,), {'rules_version': '1'})
    source = tmp_path / 'old.pt'
    save_model(source, CandidateActorCritic(16), old, cls.default_weights)
    before = source.read_bytes()
    spec = dict(kind='neural', game=game, config=cls.config_id,
                rules_version='1', path=str(source))
    with pytest.raises(ValueError):
        load_model(source, cls)
    result, audit = migrate_policy(game, spec, tmp_path / 'new.json')
    _, migrated = load_model(result['path'], cls)
    _, original = load_model(source, old)
    assert source.read_bytes() == before
    assert audit['source_model_sha256'] == hashlib.sha256(before).hexdigest()
    assert audit['evaluation_required'] and not audit['inference_parameters_changed']
    assert migrated['metadata']['rules_migration']['source_rules_version'] == '1'
    assert migrated['prior_weights'] == original['prior_weights']
    assert all(torch.equal(v, migrated['model'][k]) for k, v in original['model'].items())
    with pytest.raises(FileExistsError):
        migrate_policy(game, spec, tmp_path / 'new.json')
    with pytest.raises(ValueError):
        migrate_policy(game, {**spec, 'sha256': 'bad'}, tmp_path / 'bad.json')
    assert not (tmp_path / 'bad.json').exists()


def test_linear_migration_is_narrow_and_records_audit(tmp_path):
    cls = game_class('splendor')
    spec = dict(kind='linear', game='splendor', config=cls.config_id,
                rules_version='1', weights=list(cls.default_weights))
    result, audit = migrate_policy('splendor', spec, tmp_path / 'new.json')
    assert result['weights'] == spec['weights'] and result['rules_version'] == '2'
    assert json.loads((tmp_path / 'new.migration.json').read_text()) == audit
    for bad in ({**spec, 'rules_version': '2'}, {**spec, 'kind': 'random'},
                {**spec, 'game': 'gizmos'}, {**spec, 'decision_guard': 'unknown'}):
        with pytest.raises(ValueError):
            migrate_policy('splendor', bad, tmp_path / 'bad.json')
    with pytest.raises(ValueError):
        migrate_policy('jaipur', spec, tmp_path / 'other.json')
