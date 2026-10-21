from copy import deepcopy
import numpy as np
import torch

from boardbench.environments.harmonies import Harmonies
from boardbench.v12.encoding import feature_vector
from boardbench.v134.collector import EnvironmentPool
from boardbench.v138.baseline import build_model, export, load_model, Planner, BaselineSolver
from test_v129 import artifact


def test_portable_export(tmp_path):
    model = build_model(16)
    path = tmp_path / 'source.pt'
    torch.save(dict(version=137, source_id='test', model=model.state_dict(), config=dict(
        hidden=16, selection_method='control', arm='persistent', plan_horizon=2,
        method='sticky', advice_strength=.6, dataset='/does/not/exist')), path)
    out = export(path, tmp_path / 'export')
    path.unlink()  # Runtime has no ancestral checkpoint or training data dependency.
    loaded, saved = load_model(out / 'weights.pt')
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    assert 'dataset' not in saved
    solver = BaselineSolver(); solver.load_weights(out / 'weights.pt')
    obs, _ = Harmonies().reset(9)
    assert solver.decide(obs) in obs['legal_actions']


def test_planner_matches_existing_collector(artifact):
    pool = EnvironmentPool(2, 0, native_artifact=artifact, arm='persistent', horizon=2)
    envs = [Harmonies(), Harmonies()]; plans = [Planner(), Planner()]
    observations = [env.reset(seed)[0] for env, seed in zip(envs, [912, 913])]
    arrays = pool.reset([912, 913]); rng = np.random.default_rng(19)
    while not arrays['done'].all():
        slots = np.flatnonzero(~arrays['done']); actions = []
        for i in slots:
            obs = observations[i]; extra = pool.extra
            if extra['request'][i]:
                choice = int(extra['plan_mask'][i, 1]); extra['choice'][i] = choice
                expected = extra['plans'][i, choice]
            else:
                expected = extra['plan'][i]
            np.testing.assert_array_equal(plans[i].vector(obs), expected)
            np.testing.assert_array_equal(arrays['features'][i], np.asarray(feature_vector(obs), np.float32))
            code = int(rng.choice(np.flatnonzero(arrays['masks'][i, :240])))
            from boardbench.v12.encoding import decode
            observations[i] = envs[i].step(decode(obs, code, 'slots240_v1')).observation
            actions.append(code)
        pool.step(slots, actions)
    pool.close()
