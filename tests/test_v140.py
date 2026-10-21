from copy import deepcopy
import os
import numpy as np
import pytest
import torch

from boardbench.artifacts.store import digest
from boardbench.environments.harmonies_duo import HarmoniesDuo, player_view
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import build_model, export, Planner
from boardbench.v140.encoding import ACTIONS, FEATURES, LAYOUT, action_map, encode_batch
from boardbench.v140.collector import EnvironmentPool
from boardbench.v140.training import Trainer
from boardbench.v140.policy import export_policy, load_policy, DuoSolver
from test_v129 import artifact


@pytest.fixture
def duo_config(tmp_path, artifact):
    source = tmp_path/'source.pt'
    model = build_model(16)
    torch.save(dict(version=137, source_id='test', model=model.state_dict(), config=dict(
        hidden=16, selection_method='control', arm='persistent', plan_horizon=2,
        method='sticky', advice_strength=.6)), source)
    out = export(source, tmp_path/'export'); parent = str(out/'weights.pt')
    return dict(layout=LAYOUT, parent=parent, parent_sha256=digest(parent), native_artifact=artifact,
                objective='margin', seed=14001, learning_rate=1e-4, workers=0, batch_matches=4,
                gae_lambda=.95, minibatch=128, epochs=1, target_kl=.02, entropy=.005,
                snapshot_interval=1, max_seconds=300, evaluate_interval=2,
                development_seeds=[200000001, 200000002], confirmation_seeds=[300000001, 300000002])


def test_native_all_public_fields(artifact):
    obs, _ = HarmoniesDuo().reset(141)
    plan = Planner().vector(player_view(obs))
    slow = encode_batch([obs], [plan])
    fast = encode_batch([obs], [plan], NativeEncoder.from_artifact(artifact))
    assert slow.shape == (1, FEATURES)
    np.testing.assert_array_equal(slow, fast)
    changed = deepcopy(obs); changed['players'][1]['board'][0]['stack_id'] = 3
    assert not np.array_equal(slow, encode_batch([changed], [plan]))
    changed = deepcopy(obs); changed['animal_market'][4] = 32
    if obs['animal_market'][4] == 32: changed['animal_market'][4] = 31
    assert not np.array_equal(slow, encode_batch([changed], [plan]))
    mapping = action_map(obs)
    assert {0, 1, 2, 240, 241, 141, 142, 143, 242, 243} == set(mapping)
    assert len(mapping) == len(obs['legal_actions'])


@pytest.mark.parametrize('workers', [0, 2])
def test_pool_reference_and_per_seat_plans(artifact, workers):
    seeds = [141, 142]; envs = [HarmoniesDuo(), HarmoniesDuo()]
    obs = [e.reset(s)[0] for e, s in zip(envs, seeds)]
    planners = [[Planner(), Planner()] for _ in seeds]; rng = np.random.default_rng(121)
    with EnvironmentPool(2, workers, native_artifact=artifact) as pool:
        arrays = pool.reset(seeds)
        while not arrays['done'].all():
            slots = np.flatnonzero(~arrays['done']); chosen = []
            for i in slots:
                seat = obs[i]['active_player']; plan = planners[i][seat].vector(player_view(obs[i]))
                np.testing.assert_array_equal(arrays['features'][i], encode_batch([obs[i]], [plan])[0])
                mapping = action_map(obs[i])
                assert set(np.flatnonzero(arrays['masks'][i])) == set(mapping)
                code = int(rng.choice(list(mapping))); chosen.append(code)
                obs[i] = envs[i].step(mapping[code]).observation
            pool.step(slots, chosen)
        for i, o in enumerate(obs): assert list(arrays['stats'][i, :2]) == o['scores']


def resume_check(config, tmp_path, device):
    trainer = Trainer(config, device); first = trainer.batch()
    assert first['initial_ratio_max_error'] < 1e-5 and first['actor_change'] > 0 and first['context_change'] > 0
    assert all(t['decisions'] and t['natural_terminal'] for t in trainer.last_traces)
    from boardbench.v140.replay import replay
    for game in trainer.last_traces:
        assert len(list(replay(game))) == len(game['decisions'])
    path = tmp_path/'saved.pt'; trainer.save(path)
    expected = trainer.batch(); weights = deepcopy(trainer.model.state_dict()); trainer.close()
    restored = Trainer(config, device); restored.load(path); actual = restored.batch()
    assert actual['per_game'] == expected['per_game']
    for k, v in weights.items(): torch.testing.assert_close(v, restored.model.state_dict()[k], atol=0, rtol=0)
    rng = restored.rng.get_state().clone()
    result = restored.evaluate(config['development_seeds'])
    assert result['games'] == 4 and result['environments'] == 2
    assert torch.equal(rng, restored.rng.get_state()); restored.close()


def test_cpu_update_resume(duo_config, tmp_path):
    resume_check(duo_config, tmp_path, 'cpu')


def test_portable_duo(duo_config, tmp_path):
    t = Trainer(duo_config); path = tmp_path/'duo.pt'
    export_policy(t.model, path)
    restored = load_policy(path)
    assert all(torch.equal(v, restored.state_dict()[k]) for k,v in t.model.state_dict().items())
    obs, _ = HarmoniesDuo().reset(765)
    assert DuoSolver(t.model).decide(obs) == DuoSolver(restored).decide(obs)
    initial = t.evaluate(duo_config['development_seeds'])
    assert initial['margin_mean'] == 0 and initial['win_credit_mean'] == .5
    t.close()


@pytest.mark.skipif(os.getenv('BOARDBENCH_V140_CUDA') != '1', reason='authorized actual CUDA only')
def test_cuda_update_resume(duo_config, tmp_path):
    resume_check({**duo_config, 'workers': 2}, tmp_path, 'cuda')
