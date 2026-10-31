import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from boardbench.families import bundle, evaluate as ev
from boardbench.families.neural import CandidateActorCritic, save_model
from boardbench.families.play import write_json
from boardbench.families.registry import game_class


@pytest.fixture
def source(tmp_path):
    game = 'machi_koro'
    cls = game_class(game)
    artifacts = tmp_path/'private-inputs'
    artifacts.mkdir()
    model = artifacts/'network.pt'
    save_model(model, CandidateActorCritic(hidden=8), cls, cls.default_weights)
    # These source-only fields must never be copied to public JSONs.
    write_json(artifacts/'linear.json', dict(kind='linear', weights=list(cls.default_weights),
        metadata={'host': 'private-host.example', 'api_key': 'private-secret-fixture'},
        original_directory='/private/experiment/directory'))
    bots = artifacts/'bots.json'
    write_json(bots, {'linear': 'linear.json', 'neural': dict(kind='neural', path='network.pt'),
                     'heuristic': dict(kind='heuristic'), 'random': dict(kind='random')})
    ev.prepare_evaluation(game, bots, artifacts/'dev', [15, 16], 'dev')
    plan = tmp_path/'plan.json'
    write_json(plan, {'games': {game: dict(bots='private-inputs/bots.json',
        dev_manifest='private-inputs/dev/manifest.json', status='selected',
        recommended=['linear'], research=['neural'])}})
    return game, plan, artifacts


def test_moved_pool_loads_real_policies_and_evaluator_from_any_cwd(source, tmp_path, monkeypatch):
    game, plan, artifacts = source
    original = tmp_path/'exported'
    manifest = bundle.export_bundle(plan, original)
    assert manifest['selection_performed'] is False
    assert manifest['games'][game]['bots']['linear']['status'] == 'selected'
    assert manifest['games'][game]['bots']['neural']['status'] == 'research'
    assert manifest['games'][game]['bots']['random']['status'] == 'retained'
    moved = tmp_path/'new-location'/'pool'
    moved.parent.mkdir()
    shutil.move(original, moved)
    # The source tree is unavailable and CWD supplies a misleading same-name model.
    shutil.rmtree(artifacts)
    elsewhere = tmp_path/'elsewhere'
    elsewhere.mkdir()
    (elsewhere/'network.pt').write_bytes(b'not a model')
    monkeypatch.chdir(elsewhere)
    loaded = bundle.load_bundle(moved, manifest['manifest_sha256'])
    cls = game_class(game)
    env = cls(); env.reset(23)
    for name in loaded['games'][game]['bots']:
        policy = bundle.load_bot(moved, game, name, seed=29)
        assert policy.decide(env.observe(), env.legal_actions()) in env.legal_actions()
    refrozen = ev.freeze_bots(moved/loaded['games'][game]['bots_file'], tmp_path/'refrozen', game)
    assert set(refrozen) == {'linear', 'neural', 'heuristic', 'random'}
    assert all(not p.is_symlink() for p in moved.rglob('*'))
    for path in moved.rglob('*.json'):
        text = path.read_text()
        assert not any(value in text for value in ('private-host', 'private-secret',
            '/private/', str(artifacts), str(original), 'original_path', 'api_key'))
    provenance = ev.read_json(moved/loaded['games'][game]['dev_provenance'])
    assert provenance['phase'] == 'dev' and provenance['seeds'] == [15, 16]
    assert len(provenance['source_manifest_sha256']) == 64


@pytest.mark.parametrize('target', ['model', 'bot', 'manifest', 'extra', 'symlink'])
def test_modified_missing_or_external_package_content_rejected(source, tmp_path, target):
    game, plan, _ = source
    output = tmp_path/'exported'
    manifest = bundle.export_bundle(plan, output)
    bot = manifest['games'][game]['bots']['neural']
    model = output/bot['model']['path']
    if target == 'model':
        model.chmod(0o644); model.write_bytes(b'tampered')
    elif target == 'bot':
        (output/bot['path']).write_text('{}')
    elif target == 'manifest':
        manifest['games'][game]['recommended'] = ['random']
        write_json(output/'manifest.json', manifest)
    elif target == 'extra':
        (output/'private-launch-config.json').write_text('{}')
    else:
        copy = tmp_path/'copy.pt'
        shutil.copyfile(model, copy)
        model.unlink(); model.symlink_to(copy)
    with pytest.raises(ValueError):
        bundle.load_bundle(output)


def test_confirmation_is_not_selection_provenance_and_export_is_atomic(source, tmp_path):
    _, plan, artifacts = source
    path = artifacts/'dev'/'manifest.json'
    dev = ev.read_json(path)
    dev['phase'] = 'confirmation'
    dev = bundle._signed({k: v for k, v in dev.items() if k != 'manifest_sha256'})
    write_json(path, dev)
    output = tmp_path/'exported'
    with pytest.raises(ValueError, match='never confirmation'):
        bundle.export_bundle(plan, output)
    assert not output.exists()
    assert not list(tmp_path.glob('.bot-bundle-*'))


def test_changed_live_candidate_cannot_replace_dev_frozen_identity(source, tmp_path):
    _, plan, artifacts = source
    path = artifacts/'linear.json'
    value = ev.read_json(path); value['weights'][0] += 1; write_json(path, value)
    with pytest.raises(ValueError, match='differs from the frozen dev'):
        bundle.export_bundle(plan, tmp_path/'exported')


def test_code_config_and_external_manifest_pin_checks(source, tmp_path, monkeypatch):
    game, plan, _ = source
    output = tmp_path/'exported'
    manifest = bundle.export_bundle(plan, output)
    with pytest.raises(ValueError, match='externally pinned'):
        bundle.load_bundle(output, expected_sha256='0'*64)
    changed = bundle.runtime_identity(); changed['content_sha256'] = '0'*64
    monkeypatch.setattr(bundle, 'runtime_identity', lambda: changed)
    with pytest.raises(ValueError, match='Code/data'):
        bundle.load_bundle(output)
    assert bundle.load_bundle(output, verify_code=False) == manifest
    monkeypatch.setattr(game_class(game), 'rules_version', 'not-the-bundled-rules')
    with pytest.raises(ValueError, match='Installed game/config/rules'):
        bundle.load_bundle(output, verify_code=False)


def test_source_tree_and_installed_layout_have_same_runtime_identity(source, tmp_path):
    _, plan, _ = source
    output = tmp_path/'exported'
    manifest = bundle.export_bundle(plan, output)
    installed = tmp_path/'installed'/'site-packages'
    package = installed/'boardbench'
    shutil.copytree(Path(bundle.__file__).resolve().parents[1], package,
                    ignore=shutil.ignore_patterns('__pycache__'))
    # This optional repository metadata changes the old, layout-dependent hash.
    (installed.parent/'pyproject.toml').write_text('[project]\nname="different-layout"\n')
    # Generated Python bytecode is also deliberately excluded from the digest.
    cache = package/'__pycache__'; cache.mkdir()
    (cache/'ignored.pyc').write_bytes(b'generated bytecode')
    env = dict(os.environ, PYTHONPATH=str(installed), PYTHONNOUSERSITE='1')
    script = '''
import json, sys
from pathlib import Path
from boardbench.families import bundle, evaluate
assert Path(bundle.__file__).resolve().is_relative_to(Path(sys.argv[2]))
metadata = bundle.load_bundle(sys.argv[1])
print(json.dumps({'runtime': bundle.runtime_identity()['content_sha256'],
                  'source': evaluate.code_identity()['content_sha256']}))
'''
    command = [sys.executable, '-c', script, str(output), str(installed)]
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    identities = json.loads(result.stdout)
    assert identities['runtime'] == manifest['code']['runtime']['content_sha256']
    assert identities['source'] != manifest['code']['content_sha256']
    assert all(not name.startswith(('src/', 'boardbench/'))
               for name in manifest['code']['runtime']['files'])
    assert 'families/games/machi_koro.py' in manifest['code']['runtime']['files']
    assert 'pyproject.toml' not in manifest['code']['runtime']['files']
    # Even a one-line runtime source edit must be detected in the installed copy.
    changed = package/'families'/'games'/'machi_koro.py'
    with changed.open('a') as handle:
        handle.write('\n# changed installed runtime source\n')
    rejected = subprocess.run(command, cwd=tmp_path, env=env, text=True,
                              capture_output=True, timeout=30)
    assert rejected.returncode != 0 and 'runtime identity mismatch' in rejected.stderr


def test_explicit_recommendations_and_multiple_game_statuses(source, tmp_path):
    _, plan, artifacts = source
    second = 'century_a_new_world'
    write_json(artifacts/'second.json', {'reference': {'kind': 'heuristic'}, 'random': {'kind': 'random'}})
    ev.prepare_evaluation(second, artifacts/'second.json', artifacts/'second-dev', [19], 'dev')
    config = ev.read_json(plan)
    config['games'][second] = dict(bots='private-inputs/second.json',
        dev_manifest='private-inputs/second-dev/manifest.json', status='research',
        recommended=[], research=['reference'])
    write_json(plan, config)
    output = tmp_path/'exported'
    manifest = bundle.export_bundle(plan, output)
    assert manifest['games'][second]['status'] == 'research'
    assert manifest['games'][second]['recommended'] == []
    with pytest.raises(ValueError, match='new output'):
        bundle.export_bundle(plan, output)
    config['games'][second]['recommended'] = ['reference']
    write_json(plan, config)
    with pytest.raises(ValueError, match='research-only'):
        bundle.export_bundle(plan, tmp_path/'bad')


@pytest.mark.parametrize('game,guard,valid', [
    ('targi', 'targi_trader_acyclic_v1', True),
    ('targi', 'unrecognized_guard', False),
    ('machi_koro', 'targi_trader_acyclic_v1', False),
])
def test_explicit_decision_guard_is_preserved_or_rejected(tmp_path, game, guard, valid):
    bots = tmp_path/'bots.json'
    write_json(bots, {'reference': dict(kind='heuristic'),
                     'guarded': dict(kind='heuristic', decision_guard=guard)})
    ev.prepare_evaluation(game, bots, tmp_path/'dev', [26], 'dev')
    plan = tmp_path/'plan.json'
    write_json(plan, dict(games={game: dict(bots='bots.json', dev_manifest='dev/manifest.json',
        status='selected', recommended=['guarded'])}))
    if valid:
        manifest = bundle.export_bundle(plan, tmp_path/'pool')
        bot = manifest['games'][game]['bots']['guarded']
        assert ev.read_json(tmp_path/'pool'/bot['path'])['decision_guard'] == guard
        assert bundle.load_bundle(tmp_path/'pool') == manifest
    else:
        with pytest.raises(ValueError, match='decision_guard'):
            bundle.export_bundle(plan, tmp_path/'pool')
