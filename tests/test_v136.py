from copy import deepcopy
import os

import numpy as np
import pytest
import torch

from test_v127 import config
from test_v129 import artifact
from test_v134 import planning
from test_v135 import advice_config
from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.v12.encoding import encode
from boardbench.v130.pipeline import checkpoint_spec
from boardbench.v134.plans import candidates, encode_plan, refresh_goal
from boardbench.v135.training import Trainer as ParentTrainer
from boardbench.v135.campaign import summarize
from boardbench.v136.data import load, segments
from boardbench.v136.training import Trainer
from boardbench.v136.probe import continuations, execution_gate
from boardbench.v136.campaign import new_manifest, promotion, stage2


@pytest.fixture
def skill_config(advice_config, tmp_path):
    parent_config = {**advice_config, 'method': 'sticky', 'arm': 'persistent'}
    t = ParentTrainer(parent_config)
    t.import_sampling_continuation(parent_config['continuation'])
    path = tmp_path/'sticky.pt'; t.save(path)
    seeds = t.train_seeds[:3]
    origin = t.ppo_actions; t.close()
    data = {k: [] for k in ('x', 'goal', 'ordinary', 'mask', 'y', 'seed', 'validation', 'group')}
    for i, seed in enumerate(seeds):
        obs, _ = Harmonies().reset(seed); enc = encode(obs, allow_stop=False)
        goal = encode_plan(obs, candidates(obs)[1])
        for k, v in dict(x=enc['features'], goal=goal, ordinary=goal, mask=enc['action_mask'],
                         y=np.flatnonzero(enc['action_mask'])[0], seed=seed, validation=i==2, group=i).items():
            data[k].append(v)
    dataset = tmp_path/'skills.npz'
    np.savez_compressed(dataset, **{k: np.asarray(v) for k, v in data.items()})
    probe = tmp_path/'probes.json'; write_json(probe, [])
    return {**parent_config, 'skill_version': '1.3.6', 'skill_method': 'skills',
            'skill_dataset': {'path': str(dataset), 'sha256': digest(dataset), 'probes': str(probe), 'probes_sha256': digest(probe)},
            'skill_origin_actions': origin, 'skill_warmup_steps': 3, 'skill_warmup_lr': .0003,
            'skill_anchor_kl': 1., 'skill_bc_weight': .02, 'skill_bc_decay_actions': 100000,
            'continuation': checkpoint_spec(path)}


def check_resume(skill_config, method, tmp_path, device):
    c = {**skill_config, 'skill_method': method, 'arm': 'manager' if method == 'manager' else 'persistent'}
    t = Trainer(c, device)
    t.import_sampling_continuation(c['continuation'])
    saved = torch.load(c['continuation']['checkpoint'], weights_only=False)
    # Anchored skills cannot silently overwrite the mature terrain backbone.
    for name, value in t.model.state_dict().items():
        if name.startswith(('body.', 'actor.', 'critic.')):
            assert torch.equal(value.cpu(), saved['model'][name].cpu())
    row = t.batch()
    assert row['initial_ratio_max_error'] < 1e-4 and row['actor_change_l2'] > 0
    assert row['expert_bc_weight'] == pytest.approx(0. if method == 'control' else .02)
    if method == 'manager':
        assert row['manager_change'] > 0 and row['manager_ratio_error'] < 1e-4
    path = tmp_path/'checkpoint.pt'; t.save(path)
    expected = t.batch(); weights = deepcopy(t.model.state_dict()); high = deepcopy(t.manager.state_dict())
    t.close()
    restored = Trainer(c, device); restored.load(path)
    actual = restored.batch()
    assert actual['games'] == expected['games']
    for name, value in weights.items():
        torch.testing.assert_close(value, restored.model.state_dict()[name], atol=0, rtol=0)
    for name, value in high.items():
        torch.testing.assert_close(value, restored.manager.state_dict()[name], atol=0, rtol=0)
    rng = torch.get_rng_state().clone(); cuda = torch.cuda.get_rng_state_all() if device == 'cuda' else []
    restored.evaluate(c['curve_seeds'])
    assert torch.equal(rng, torch.get_rng_state())
    if cuda:
        assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    restored.close()


@pytest.mark.parametrize('method', ('control', 'plain', 'skills', 'manager'))
def test_update_resume(skill_config, method, tmp_path):
    check_resume(skill_config, method, tmp_path, 'cpu')


@pytest.mark.skipif(os.environ.get('BOARDBENCH_V136_CUDA') != '1', reason='authorized GPU only')
@pytest.mark.parametrize('method', ('control', 'plain', 'skills', 'manager'))
def test_cuda_resume(skill_config, method, tmp_path):
    check_resume({**skill_config, 'workers': 2, 'batch_episodes': 8}, method, tmp_path, 'cuda')


def test_control_parent_exact_and_stage(skill_config, tmp_path):
    c = {**skill_config, 'skill_method': 'control'}
    t = Trainer(c); t.import_sampling_continuation(c['continuation'])
    saved = torch.load(c['continuation']['checkpoint'], weights_only=False)
    p = ParentTrainer(saved['config']); p.model.load_state_dict(saved['model'])
    assert t.evaluate(c['curve_seeds'])['games'] == p.evaluate(c['curve_seeds'])['games']
    p.close(); t.batch()
    checkpoint = tmp_path/'stage1.pt'; t.save(checkpoint)
    expected = t.batch(); weights = deepcopy(t.model.state_dict()); t.close()
    c2 = {**c, 'continuation': checkpoint_spec(checkpoint), 'target_actions': 10000, 'milestones': [10000], 'max_seconds': 900}
    next_stage = Trainer(c2); next_stage.import_sampling_continuation(c2['continuation'])
    assert expected['games'] == next_stage.batch()['games']
    for name, value in weights.items():
        assert torch.equal(value, next_stage.model.state_dict()[name])
    next_stage.close()
    bad = Trainer({**c2, 'skill_anchor_kl': 2.})
    with pytest.raises(ValueError, match='preserve'):
        bad.import_sampling_continuation(c2['continuation'])
    bad.close()


def test_dataset_exclusion_and_mask(skill_config):
    t = Trainer(skill_config)
    with pytest.raises(ValueError, match='leakage'):
        load(skill_config['skill_dataset'], [])
    with pytest.raises(ValueError, match='hash'):
        load({**skill_config['skill_dataset'], 'sha256': 'wrong'}, t.train_seeds)
    x, mask, y = t.skill_batch(256, np.random.default_rng(1))
    assert mask[torch.arange(256), y].all() and x.shape[1] == 2165
    t.close()


def test_public_paired_continuation(skill_config):
    t = Trainer({**skill_config, 'skill_method': 'control'})
    t.import_sampling_continuation(skill_config['continuation'])
    obs, _ = Harmonies().reset(t.train_seeds[0]); goal = candidates(obs)[1][0]
    specs = [{'state': 0, 'environment_seed': t.train_seeds[0], 'future_seed': 889,
              'branch': branch, 'observation': obs, 'plan': [goal], 'target': goal} for branch in ('a', 'b')]
    original = deepcopy(obs); rng = torch.get_rng_state().clone()
    result = continuations(t, specs)
    a, b = result['rows']
    assert a['score'] == b['score'] and a['animal_score'] == b['animal_score']
    assert a['natural_terminal'] and b['natural_terminal'] and obs == original
    assert torch.equal(rng, torch.get_rng_state())
    t.close()


def test_hindsight_never_labels_invisible_card():
    # Build real legal trajectories using the existing deterministic public heuristic.
    from boardbench.v125.policies import GoalContinuation
    from boardbench.v128.environment import FastHarmonies
    policy = GoalContinuation(compiled=True)
    e = FastHarmonies(); obs, _ = e.reset(140)
    observations, actions = [], []
    while not obs['terminated']:
        action = policy.decide(obs)
        observations.append(obs); actions.append(action); obs = e.step(action).observation
    rows = segments({'episodes': [{'id': 'fixture', 'environment_seed': 140, 'observations': observations, 'actions': actions}]})
    assert rows
    for row in rows:
        assert len(row['actions']) >= 3 and row['actions'][-1]['type'] == 'place_animal'
        assert refresh_goal(row['observations'][0], row['goal'])['missing'] >= 1
        for state in row['observations']:
            assert row['goal']['card_id'] in state['animal_market'] or any(c['card_id'] == row['goal']['card_id'] for c in state['active_cards'])
            assert refresh_goal(state, row['goal']) is not None


def test_execution_gate_and_fresh_panels(config, tmp_path):
    old = read_json(config['split_manifest']); new = new_manifest(old, config['split_sha256'], 8)
    assert new['sets']['train'] == old['sets']['train'] and new['sets']['test'] == old['sets']['test']
    fresh = new['sets']['development'] + new['sets']['confirmation']
    assert len(set(fresh)) == 24 and all(60000000 <= s < 70000000 for s in fresh)
    control = {'rows': [{'environment_seed': 1, 'state': 0, 'future_seed': 1, 'branch': 'target'}],
               'means': {'target': {'score': 100., 'target_completed': .2}, 'free': {'score': 100., 'target_completed': .2}}}
    skills = deepcopy(control); skills['means']['target']['target_completed'] = .3
    assert execution_gate(control, skills)['passed']
    skills['means']['target']['score'] = 98.
    assert not execution_gate(control, skills)['passed']
    write_json(tmp_path/'stage1/jobs.json', [[m+'_'+s, {'skill_method': m}] for m in ('control', 'plain', 'skills') for s in 'ab'])
    for method, score in [('control', 100), ('plain', 101), ('skills', 102)]:
        for s in 'ab':
            directory = tmp_path/f'stage1/trials/{method}_{s}'
            write_json(directory/'evaluations/final.json', {'mean_score': score})
            if method in ('control', 'skills'):
                write_json(directory/'execution_probe.json', control if method == 'control' else skills)
    assert promotion(tmp_path)['selected'] == ['control', 'skills']  # Failed total gate blocks manager.


def test_v135_negative_numpy_bool_summary_serializes(tmp_path):
    seeds = [1, 2]
    write_json(tmp_path/'splits.json', {'sets': {'confirmation': seeds}})
    write_json(tmp_path/'promotion.json', {'selected': ['control', 'sticky']})
    results = {}
    for name in ('parent', 'control_a', 'control_b', 'sticky_a', 'sticky_b'):
        score = 99 if name.startswith('sticky') else 100
        results[name] = {'mean_score': score, 'games': [{'environment_seed': s, 'score': score} for s in seeds]}
    interventions = {n: {'zero': {'mean': 1.}} for n in ('sticky_a', 'sticky_b')}
    summarize(tmp_path, results, interventions)
    assert read_json(tmp_path/'confirmation.json')['provisional_useful_advice_signal'] is False


def test_manager_branch_preserves_trained_worker(skill_config, tmp_path, monkeypatch):
    import boardbench.v136.probe as probe
    t = Trainer(skill_config); t.import_sampling_continuation(skill_config['continuation']); t.batch()
    path = tmp_path/'skills_stage1.pt'; t.save(path)
    expected = deepcopy(t.model.state_dict()); t.close()
    seen = []
    def bootstrap(current):
        assert current.config['arm'] == 'manager'
        assert all(torch.equal(v, current.model.state_dict()[k]) for k, v in expected.items())
        seen.append(True)
        return {'unit_test': 'worker preservation before current-worker Q labeling'}
    monkeypatch.setattr(probe, 'bootstrap_manager', bootstrap)
    c = {**skill_config, 'skill_method': 'manager', 'arm': 'manager',
         'continuation': checkpoint_spec(path), 'target_actions': 10000, 'milestones': [10000], 'max_seconds': 900}
    manager = Trainer(c); manager.import_sampling_continuation(c['continuation'])
    assert seen and manager.lineage['kind'] == 'current_worker_goal_value_initialization'
    row = manager.batch()
    assert row['manager_ratio_error'] < 1e-4 and row['manager_change'] > 0
    manager.close()


def test_manager_bootstrap_targets_current_worker(skill_config, monkeypatch):
    import boardbench.v136.probe as probe
    c = {**skill_config, 'manager_label_states': 2, 'manager_label_updates': 8, 'skill_corpus': {}}
    t = Trainer(c); t.import_sampling_continuation(c['continuation'])
    rows = []
    for s in t.train_seeds[:2]:
        obs, _ = Harmonies().reset(s)
        rows.append({'environment_seed': s, 'observations': [obs]})
    monkeypatch.setattr(probe, 'load_corpus', lambda spec, allowed: {'rows': rows})
    monkeypatch.setattr(probe, 'segments', lambda corpus: corpus['rows'])
    def rollout(current, specs, timeout):
        assert current is t and all(s['environment_seed'] in t.train_seeds for s in specs)
        return {'rows': [{k: s[k] for k in ('state', 'branch', 'environment_seed', 'future_seed')}
                         | {'score': 100 + 2*s['branch']} for s in specs]}
    monkeypatch.setattr(probe, 'continuations', rollout)
    before = deepcopy(t.model.state_dict())
    result = probe.bootstrap_manager(t)
    assert result['states'] == 2 and result['last_loss'] < result['first_loss']
    assert all(torch.equal(v, t.model.state_dict()[k]) for k, v in before.items())
    t.close()


def test_failed_post_training_probe_not_marked_complete(skill_config, tmp_path):
    from boardbench.v136.training import run
    c = {**skill_config, 'skill_method': 'control', 'target_actions': 0, 'milestones': []}
    with pytest.raises(ValueError, match='probe environments'):
        run(c, tmp_path/'failed_probe', 'cpu')
    assert (tmp_path/'failed_probe/final.pt').exists()
    assert not (tmp_path/'failed_probe/summary.json').exists()
    assert read_json(tmp_path/'failed_probe/progress.json')['phase'] == 'failed'
