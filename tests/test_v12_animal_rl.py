from copy import deepcopy

import pytest

from boardbench.environments.harmonies import Harmonies
from boardbench.solvers.board import ranked_actions
from boardbench.v12.animal_diagnostics import case_split, measure
from boardbench.v12.diagnostics import exact_values
from boardbench.v12.training import Trainer


def final_case(seed=107, horizon=1):
    env = Harmonies(); obs, _ = env.reset(seed)
    while sum(c['stack_id'] == 0 for c in obs['board']) > 2:
        obs = env.step(ranked_actions(obs)[0][1]).observation
        assert not obs['terminated']
    snapshot = env.save_state()
    _, values = exact_values(snapshot, horizon)
    return {'snapshot': snapshot, 'horizon': horizon, 'source_seed': seed,
            'level': 'placement', 'layout_id': str(seed), 'values': values}


def test_snapshot_ppo_is_real_on_policy_and_rejects_holdout_games():
    case = final_case()
    original = deepcopy(case)
    config = {'training_seed': 1, 'algorithm': 'ppo',
              'network': {'hidden': 16, 'action_encoding': 'slots240_v1', 'observation_encoding': 'cards_v1'},
              'splits': {'train': [107], 'validation': [21], 'test': [31]},
              'batch_episodes': 4, 'epochs': 1}
    trainer = Trainer(config)
    result = trainer.ppo_batch([case])
    assert result['initial_ratio_max_error'] < 1e-4 and trainer.updates > 0
    assert trainer.episodes == 4 and trainer.steps <= 4
    assert 'auxiliary' in result['objective'] and case == original
    gaps = measure(trainer.solver, [case])
    assert gaps[0]['rollout_gap'] >= 0
    bad = {**case, 'source_seed': 31}
    with pytest.raises(ValueError, match='training-only'):
        trainer.ppo_batch([bad])


def test_source_game_and_layout_split_and_exact_node_cap():
    cases = [{'source_seed': seed, 'layout_id': str(seed) + suffix}
             for seed in range(8) for suffix in ('a', 'b')]
    train, heldout = case_split(cases)
    assert {c['source_seed'] for c in train}.isdisjoint(c['source_seed'] for c in heldout)
    assert {c['source_seed'] for c in heldout} == {3, 7}
    with pytest.raises(ValueError, match='layout leakage'):
        case_split([{'source_seed': 0, 'layout_id': 'x'}, {'source_seed': 3, 'layout_id': 'x'}])
    case = final_case()
    with pytest.raises(ValueError, match='node cap'):
        exact_values(case['snapshot'], depth=3, max_nodes=1)


def test_short_horizon_targets_terminate_without_full_game_bootstrap():
    case = final_case(107, 2)
    config = {'training_seed': 9, 'algorithm': 'ppo',
              'network': {'hidden': 16, 'action_encoding': 'slots240_v1'},
              'splits': {'train': [107], 'validation': [21], 'test': [31]},
              'batch_episodes': 1, 'epochs': 1}
    trainer = Trainer(config)
    captured = []
    update = trainer._update
    def capture(rows, bc=False):
        captured.extend(rows)
        return update(rows, bc)
    trainer._update = capture
    result = trainer.ppo_batch([case])
    assert 1 <= len(captured) <= 2
    assert captured[0][4] * 150 == pytest.approx(result['mean_score_gain'])


def test_animal_gate_requires_total_animals_and_actual_rl_contribution():
    from boardbench.v12.animal_experiment import animal_gate, nominated_gate
    def rows(score, animals, placements=2, completed=1):
        return [{'seed': seed, 'failed': False, 'score': score,
                 'components': {'animals': animals, 'animal_placements': placements, 'completed_cards': completed}}
                for seed in range(8)]
    base = rows(30, 10, 1, 0)
    stronger = rows(40, 15)
    teacher = rows(35, 12)
    assert animal_gate([stronger]*3, {'old': base}, base, [teacher]*3)['passed']
    assert not animal_gate([rows(40, 8)]*3, {'old': base}, base)['passed']
    assert not animal_gate([stronger]*3, {'old': base}, base, [stronger]*3)['passed']
    bad_teacher = [{**r, 'failed': True} for r in teacher]
    assert not animal_gate([stronger]*3, {'old': base}, base, [bad_teacher]*3)['passed']
    assert nominated_gate(stronger, {'old': base}, base, teacher)['passed']
    assert not nominated_gate(rows(40, 8), {'old': base}, base)['passed']
    assert not nominated_gate(stronger, {'old': base}, base, stronger)['passed']


def test_real_bc_to_ppo_cli_and_fresh_rollout(tmp_path):
    import subprocess
    import sys
    from boardbench.artifacts.store import digest, read_json, write_json
    from boardbench.v12.neural import CompactSolver
    cfg = {'training_seed': 5, 'algorithm': 'bc', 'network': {'hidden': 16},
           'splits': {'train': [10], 'validation': [21], 'test': [31]},
           'batch_episodes': 2, 'epochs': 1}
    obs, _ = Harmonies().reset(10)
    data = CompactSolver(5, hidden=16).encode(obs)
    teacher = tmp_path / 'teacher.json'
    write_json(teacher, [{'features': data['features'], 'mask': data['action_mask'],
                          'action': data['action_mask'].index(True), 'seed': 10}])
    cfg['teacher_sha256'] = digest(teacher)
    config = tmp_path / 'config.json'; write_json(config, cfg)
    bc = tmp_path / 'bc'
    subprocess.run([sys.executable, '-m', 'boardbench.v12.training', '--config', str(config),
                    '--out', str(bc), '--seconds', '.01', '--teacher', str(teacher)],
                   check=True, capture_output=True, timeout=30)
    cfg['algorithm'] = 'ppo'; cfg.pop('teacher_sha256'); write_json(config, cfg)
    weights = bc / 'final.resume.weights.pt'
    ppo = tmp_path / 'ppo'
    subprocess.run([sys.executable, '-m', 'boardbench.v12.training', '--config', str(config),
                    '--out', str(ppo), '--seconds', '.01', '--initialize-bc', str(weights),
                    '--initialize-sha256', digest(weights)], check=True, capture_output=True, timeout=30)
    report = read_json(ppo / 'summary.json')
    assert report['episodes'] == 2 and report['updates'] > 0
    assert report['lineage']['weights_sha256'] == digest(weights)
    assert read_json(ppo / 'initial.resume.json')['updates'] == 0


def test_continuation_preserves_both_original_deadline_and_all_charges(tmp_path):
    from boardbench.artifacts.store import write_json
    from boardbench.v12.launch import continuation_budget
    old = tmp_path / 'outputs/v12/old'
    write_json(old / 'allocation.json', {'budget_seconds': 14400, 'deadline_unix': 20000})
    write_json(old / 'watchdog_result.json', {'elapsed_seconds': 600})
    plan = {'budget_seconds': 14400, 'continuation_from': 'outputs/v12/old',
            'prior_diagnostic_charge_seconds': 350}
    seconds, provenance = continuation_budget(plan, tmp_path, now=10000)
    assert seconds == 10000 and provenance['prior_charged_seconds'] == 950
    seconds, _ = continuation_budget(plan, tmp_path, now=1000)
    assert seconds == 13450
    with pytest.raises(ValueError, match='exhausted'):
        continuation_budget(plan, tmp_path, now=20000)
    with pytest.raises(ValueError, match='project'):
        continuation_budget({**plan, 'continuation_from': '/tmp'}, tmp_path, now=1000)


def test_kl_guard_stops_without_disabling_behavior_consistency():
    config = {'training_seed': 19, 'network': {'hidden': 16},
              'splits': {'train': [107], 'validation': [21], 'test': [31]},
              'batch_episodes': 16, 'epochs': 10, 'minibatch_size': 4,
              'learning_rate': .01, 'target_kl': 1e-7}
    trainer = Trainer(config)
    result = trainer.ppo_batch([final_case(107, 2)])
    assert result['initial_ratio_max_error'] < 1e-4
    assert result['kl_early_stop'] and result['optimizer_steps_this_batch'] > 0
