from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from boardbench.artifacts.store import read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.solvers.board import ranked_actions
from boardbench.ui.server import PlaySession
from boardbench.v12.budget import Ledger
from boardbench.v12.deployment import export, publish, resolve_release
from boardbench.v12.diagnostics import exact_values
from boardbench.v12.evaluation import paired
from boardbench.v12.experiment import family_gate, fixed_policy_gate
from boardbench.v12.timed import run_timed


def test_shared_budget_stops_child_and_preserves_logs(tmp_path, monkeypatch):
    monkeypatch.delenv('BOARDBENCH_STARTED_MONOTONIC', raising=False)
    ledger = Ledger(tmp_path, .3)
    with pytest.raises(TimeoutError):
        ledger.run('slow', [sys.executable, '-c', 'import time; print("started", flush=True); time.sleep(30)'], 10)
    assert ledger.remaining() == 0 and ledger.phases[0]['timeout']
    assert (tmp_path / 'logs/slow.log').read_text() == 'started\n'
    assert ledger.phases[0]['seconds'] < 2
    with pytest.raises(TimeoutError):
        ledger.run('cannot_reset_clock', [sys.executable, '-c', 'pass'], 10)


def test_gate_requires_all_repetitions_and_uses_paired_rows():
    base = [{'seed': i, 'score': 3, 'failed': False} for i in range(8)]
    higher = [{**row, 'score': 10} for row in base]
    assert family_gate([higher] * 3, {'incumbent': base})['passed']
    assert not family_gate([higher] * 2, {'incumbent': base})['passed']
    assert not family_gate([base] * 3, {'incumbent': higher})['passed']
    assert not family_gate([higher] * 3, {'incumbent': [{**row, 'failed': True} for row in base]})['passed']
    assert fixed_policy_gate(higher, {'incumbent': base})['passed']
    assert not fixed_policy_gate(base, {'incumbent': higher})['passed']
    with pytest.raises(ValueError):
        paired(base + [base[0]], base)
    with pytest.raises(ValueError):
        paired(base, base[:-1])


def test_source_verified_release_and_game_boundary_refresh(tmp_path):
    pointer, release = tmp_path / 'current.json', tmp_path / 'release1'
    spec = {'id': 'first', 'method': 'heuristic', 'label': 'first',
            'solver': {'id': 'board', 'params': {'method': 'heuristic'}}}
    export('micro_tiles', spec, release / 'micro_tiles/policies/first')
    publish(release, pointer, {'micro_tiles': 'first'}, kind='legacy_import')
    config = {'task': {'id': 'micro_tiles', 'version': 'micro_tiles_v1'}, 'deployment_pointer': str(pointer)}
    session = PlaySession(config)
    assert session.policy_id == 'first'
    release2 = tmp_path / 'release2'
    spec['id'] = 'second'
    export('micro_tiles', spec, release2 / 'micro_tiles/policies/second')
    with pytest.raises(ValueError, match='gate'):
        publish(release2, pointer, {'micro_tiles': 'second'}, kind='promotion')
    assert read_json(pointer)['release'] == str(release)
    publish(release2, pointer, {'micro_tiles': 'second'}, kind='promotion', gate={'passed': True})
    assert session.policy_id == 'first'
    session.new_game()
    assert session.policy_id == 'second'
    assert read_json(release2 / 'publication.json')['previous']['release'] == str(release)
    path = release2 / 'micro_tiles/policies/second/manifest.json'
    value = read_json(path); value['label'] = 'tampered'; write_json(path, value)
    with pytest.raises(ValueError, match='manifest changed'):
        resolve_release(pointer, 'micro_tiles')
    session.registry.close()


def test_ui_stop_deadline_save_and_restore(tmp_path, monkeypatch):
    config = {'task': {'id': 'harmonies', 'version': Harmonies.rules_version},
              'human_time_budget_seconds': 10, 'save_directory': str(tmp_path)}
    session = PlaySession(config)
    action = session.observation['legal_actions'][0]
    session.step(action)
    snapshot = session.env.save_state()
    session.step({'type': 'stop_episode'})
    assert session.env.save_state() == snapshot
    state = session.state()
    assert state['episode_finished'] and not state['observation']['terminated']
    saved = session.save_game(); session.new_game(); session.load_game(saved)
    assert session.end_reason == 'stopped'
    with pytest.raises(ValueError, match='finished'):
        session.step(action)
    session.new_game()
    saved = session.save_game()
    value = read_json(tmp_path / (saved + '.json'))
    value['deadline_unix'] = time.time() - 1; write_json(tmp_path / (saved + '.json'), value)
    session.load_game(saved)
    assert session.state()['episode_end_reason'] == 'deadline'
    session.registry.close()


def test_no_chance_oracle_matches_brute_force_and_leaves_snapshot_unchanged():
    env = Harmonies(); obs, _ = env.reset(107)
    for _ in range(256):
        if sum(c['stack_id'] == 0 for c in obs['board']) <= 2:
            break
        obs = env.step(ranked_actions(obs)[0][1]).observation
    snapshot = env.save_state(); original = deepcopy(snapshot)
    obs, values = exact_values(snapshot, depth=1)
    assert snapshot == original
    expected = [obs['score_breakdown']['total']]
    for action in obs['legal_actions']:
        other = Harmonies(); other.load_state(snapshot)
        expected.append(other.step(action).observation['score_breakdown']['total'])
    assert sorted(values.values()) == sorted(expected)


@pytest.mark.parametrize('task', ['micro_tiles', 'take_it_easy', 'harmonies'])
def test_solver_time_allocation_is_legal_and_lifecycle_finishes(task):
    row = run_timed(task, {'solver': {'id': 'timed_search', 'params': {'max_simulation_steps': 12}}}, 20, 2)
    assert not row['failed'], row['error']
    assert row['worker_reaped']
    assert row['reason'] in ('natural', 'deadline')
    if row['reason'] == 'natural':
        assert row['terminal_hook_acknowledged']


def test_real_training_cli_evaluation_and_export(tmp_path):
    from boardbench.artifacts.store import digest
    from boardbench.ui.policies import PolicyRegistry
    cfg = {'training_seed': 191, 'network': {'hidden': 16, 'action_encoding': 'slots240_v1',
            'observation_encoding': 'cards_v1', 'allow_stop': True},
           'splits': {'train': [10, 11], 'validation': [21, 22], 'test': [31, 32]},
           'batch_episodes': 2, 'epochs': 1, 'threads': 1}
    config = tmp_path / 'config.json'; write_json(config, cfg)
    out = tmp_path / 'training'
    subprocess.run([sys.executable, '-m', 'boardbench.v12.training', '--config', str(config),
                    '--out', str(out), '--seconds', '.01'], check=True, timeout=30, capture_output=True)
    weights = out / 'final.resume.weights.pt'
    spec = {'id': 'compact_smoke', 'method': 'rl', 'label': 'compact',
            'solver': {'id': 'compact', 'params': cfg['network']},
            'weights': str(weights), 'weights_sha256': digest(weights)}
    row = run_timed('harmonies', spec, 21, 1)
    assert not row['failed'], row['error']
    assert row['reason'] in ('natural', 'stopped', 'deadline')
    export('harmonies', spec, tmp_path / 'policies/compact_smoke')
    registry = PolicyRegistry({'id': 'harmonies', 'version': Harmonies.rules_version}, tmp_path / 'policies')
    try:
        restored, _ = registry.get('compact_smoke', fresh=True)
        assert restored.training['updates'] > 0
    finally:
        registry.close()
