from copy import deepcopy
import os

import numpy as np
import pytest
import torch

from boardbench.environments.harmonies import Harmonies
from boardbench.training import parameter_hash
from boardbench.v12.encoding import STOP, decode, encode
from boardbench.v12 import training as original
from boardbench.v121.collector import EnvironmentPool, _Shard, collect, REWARD, STOPPED
from boardbench.v121.registry import training_config, make_splits, definitions, THROUGHPUT
from boardbench.v121.training import Trainer


def config(name='E00_T16', workers=0):
    cfg = training_config(name, 911, make_splits(train_count=32), workers)
    cfg.update(batch_episodes=4, epochs=1, minibatch_size=64, purpose='interface_check')
    return cfg


def assert_rows_match(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2] == b[2]
        np.testing.assert_allclose(a[3:], b[3:], atol=2e-6, rtol=2e-6)


def test_new_variants_do_not_change_model_initialization_or_ppo_hyperparameters():
    for new, old in THROUGHPUT.items():
        a, b = Trainer(config(new)), Trainer(config(old))
        assert parameter_hash(a.solver.model) == parameter_hash(b.solver.model)
        for key in ['batch_episodes','epochs','minibatch_size','learning_rate','reward_scale','target_kl','clip_ratio','entropy_coef','splits']:
            assert a.config[key] == b.config[key]
        assert definitions()[new]['parent_id'] == old
    cfg = config(); cfg['collector']['auto_reset'] = True
    with pytest.raises(ValueError, match='collector'):
        Trainer(cfg)
    cfg = config('E00'); cfg['collector'] = config()['collector']
    with pytest.raises(ValueError, match='serial'):
        Trainer(cfg)


@pytest.mark.parametrize('workers', [0, 2])
def test_pool_scripted_games_match_serial_engine_masks_returns_and_no_autoreset(workers):
    if workers and len(os.sched_getaffinity(0)) < 3:
        pytest.skip('three cores needed for two-worker pool')
    seeds = [11, 13, 17, 19]
    games = [Harmonies() for _ in seeds]
    obs = [g.reset(s)[0] for g,s in zip(games,seeds)]
    rng = np.random.default_rng(44)
    done = np.zeros(4, bool)
    rewards = [[] for _ in seeds]
    with EnvironmentPool(4, workers) as pool:
        arrays = pool.reset(seeds)
        with pytest.raises(ValueError, match='unfinished'):
            pool.reset(seeds)
        for turn in range(256):
            slots = np.flatnonzero(~done)
            if not len(slots):
                break
            chosen = []
            for slot in slots:
                data = encode(obs[slot])
                np.testing.assert_allclose(arrays['features'][slot], data['features'], atol=1e-7)
                np.testing.assert_array_equal(arrays['masks'][slot], data['action_mask'])
                legal = np.flatnonzero(arrays['masks'][slot])
                action_id = 240 if slot == 0 or (slot == 1 and turn == 3) else int(rng.choice(legal[legal != 240]))
                chosen.append(action_id)
                action = decode(obs[slot], action_id, 'slots240_v1', True)
                nxt = obs[slot] if action == STOP else games[slot].step(action).observation
                rewards[slot].append((nxt['score_breakdown']['total']-obs[slot]['score_breakdown']['total'])/150.)
                obs[slot] = nxt
                done[slot] = action == STOP or nxt['terminated']
            arrays = pool.step(slots, chosen)
            np.testing.assert_array_equal(arrays['done'], done)
            for slot in slots:
                assert arrays['stats'][slot, REWARD] == rewards[slot][-1]
                if done[slot]:
                    assert not arrays['masks'][slot].any()
        assert done.all() and arrays['stats'][:, STOPPED].sum() == 2
        for slot in range(4):
            assert sum(rewards[slot]) == pytest.approx(obs[slot]['score_breakdown']['total']/150.)
        pool.reset(seeds)
        with pytest.raises(ValueError, match='legal mask'):
            pool.step(np.arange(4), [10000]*4)
        processes = list(pool.processes)
    assert all(not p.is_alive() for p in processes)


@pytest.mark.parametrize('name', list(THROUGHPUT))
def test_batched_rows_equal_serial_with_fixed_greedy_actions(name, monkeypatch):
    distribution = original.policy_distribution
    def greedy(logits):
        dist = distribution(logits)
        dist.sample = lambda: logits.argmax(-1)
        return dist
    monkeypatch.setattr(original, 'policy_distribution', greedy)
    serial, batched = Trainer(config(THROUGHPUT[name])), Trainer(config(name))
    serial_rows, batched_rows = [], []
    def capture(target):
        def update(rows):
            target.extend(rows)
            return {}
        return update
    serial._update = capture(serial_rows)
    batched._update = capture(batched_rows)
    a, b = serial.ppo_batch(), batched.ppo_batch()
    batched.close()
    assert_rows_match(serial_rows, batched_rows)
    for key in ['real_atomic_actions','policy_decisions','stop_decisions','mean_training_score',
                'mean_training_animal_score','natural_fraction','animal_placements_per_episode','completed_cards_per_episode']:
        assert a[key] == b[key]


@pytest.mark.parametrize('name', list(THROUGHPUT))
def test_batched_real_update_cpu_exact_resume_and_artifact(name, tmp_path):
    a = Trainer(config(name))
    initial = parameter_hash(a.solver.model)
    first = a.ppo_batch()
    assert first['initial_ratio_max_error'] <= 1e-4
    assert parameter_hash(a.solver.model) != initial
    assert first['mean_inference_batch'] > 1
    assert first['policy_decisions'] == first['real_atomic_actions'] + first['stop_decisions']
    path = tmp_path/'state.pt'
    a.save(path)
    a.ppo_batch()
    expected = parameter_hash(a.solver.model)
    a.close()
    b = Trainer(config(name)); b.load(path); b.ppo_batch()
    assert parameter_hash(b.solver.model) == expected
    assert (a.steps,a.decision_count,a.stop_decisions,a.updates) == (b.steps,b.decision_count,b.stop_decisions,b.updates)
    for key,state in a.optimizer.state_dict()['state'].items():
        for field,value in state.items():
            assert torch.equal(value,b.optimizer.state_dict()['state'][key][field])
    c = Trainer(config(name)); c.solver.load_weights(path.with_suffix('.weights.pt'))
    assert c.solver.training['experiment_id'] == name
    b.close()


def test_parallel_worker_collection_equals_inprocess_rng_and_update():
    if len(os.sched_getaffinity(0)) < 3:
        pytest.skip('three cores needed')
    a = Trainer(config(workers=0)); a.ppo_batch()
    expected = parameter_hash(a.solver.model)
    a.close()
    b = Trainer(config(workers=2))
    try:
        b.ppo_batch()
        assert parameter_hash(b.solver.model) == expected
    finally:
        b.close()


def test_parallel_worker_resume_is_exact(tmp_path):
    if len(os.sched_getaffinity(0)) < 3:
        pytest.skip('three cores needed')
    cfg = config('E11_T16', workers=2)
    a = Trainer(cfg)
    try:
        a.ppo_batch(); a.save(tmp_path/'state.pt'); a.ppo_batch()
        expected = parameter_hash(a.solver.model)
    finally:
        a.close()
    b = Trainer(cfg)
    try:
        b.load(tmp_path/'state.pt'); b.ppo_batch()
        assert parameter_hash(b.solver.model) == expected
    finally:
        b.close()


def test_stop_accounting_and_failed_batch_cannot_be_saved(tmp_path):
    a = Trainer(config())
    with torch.no_grad():
        a.solver.model.actor.weight.zero_()
        a.solver.model.actor.bias.fill_(-100)
        a.solver.model.actor.bias[-1] = 100
    row = a.ppo_batch()
    assert row['real_atomic_actions'] == 0
    assert row['stop_decisions'] == row['policy_decisions'] == 4
    assert row['natural_fraction'] == row['mean_training_score'] == 0
    def fail(_):
        raise RuntimeError('update failed')
    a._update = fail
    with pytest.raises(RuntimeError, match='update failed'):
        a.ppo_batch()
    with pytest.raises(RuntimeError, match='boundary'):
        a.save(tmp_path/'invalid.pt')
    with pytest.raises(RuntimeError, match='previous collection'):
        a.ppo_batch()
    a.close()


def test_private_future_not_published():
    with EnvironmentPool(1, 0) as pool:
        pool.reset([19])
        x, mask = pool.arrays['features'].copy(), pool.arrays['masks'].copy()
        env = pool.local.games[0]
        env._bag.reverse(); env._deck.reverse()
        pool.local.observations[0] = env.observe()
        pool.local._publish(0)
        np.testing.assert_array_equal(pool.arrays['features'], x)
        np.testing.assert_array_equal(pool.arrays['masks'], mask)


def test_dead_worker_fails_instead_of_silent_missing_trajectories():
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip('two cores needed')
    with EnvironmentPool(2, 1, timeout=5.) as pool:
        pool.reset([3,4])
        pool.processes[0].terminate(); pool.processes[0].join(timeout=2.)
        with pytest.raises((RuntimeError, BrokenPipeError, EOFError, ConnectionResetError)):
            pool.step(np.arange(2), [240,240])
