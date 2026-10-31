from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from boardbench.families.neural import CandidateActorCritic, collate, load_model, save_model
from boardbench.families.opponent_pool import OpponentPool, pool_seed
from boardbench.families.pool_training import (audit_behavior, restore_checkpoint, rollout,
                                               save_checkpoint, signal_diagnostics)
from boardbench.families.ppo import update
from boardbench.families.registry import GAME_IDS, game_class
from boardbench.families.reward_signals import (SCORELESS, calibrate, raw_labels, terminal_signals,
                                               validate_signal)
from boardbench.families.trajectory_io import compact_episode


def result(game='harmonies', scores=(90, 70), status='complete'):
    return dict(game=game, seed=1, status=status, outcome=dict(scores=list(scores)),
                score_breakdown=[dict(total=scores[0], animals=30), dict(total=scores[1], animals=10)],
                rewards=[1., -1.])


def signal(r=None, mode='aux'):
    return dict(mode=mode, schema=calibrate([r or result()]),
                auxiliary_coef=.25 if mode!='win' else 0., mix_weight=.25 if mode=='mixed' else 0.)


def test_terminal_seat_alignment_mixed_utility_and_incomplete_exclusion():
    r = result()
    cfg = signal(r, 'mixed')
    a, b = [terminal_signals(r, p, cfg) for p in (0, 1)]
    assert a['training_reward'] == pytest.approx(.75+.25*np.tanh(1.))
    assert b['training_reward'] == -a['training_reward']
    names = cfg['schema']['names']
    assert a['targets'][names.index('own.animals')] == 1.
    assert b['targets'][names.index('own.animals')] == -1.
    for mode in ('win', 'aux'):
        assert terminal_signals(r, 0, signal(r, mode))['training_reward'] == 1.
    r['status'] = 'truncated'
    assert raw_labels(r, 0) == {}
    with pytest.raises(ValueError): terminal_signals(r, 0, cfg)


@pytest.mark.parametrize('game', [g for g in GAME_IDS if g!='five_tribes'])
def test_all_game_breakdown_schemas_and_scoreless_masks(game):
    env = game_class(game)(); env.reset(seed=1711)
    r = dict(game=game, seed=1, status='complete', outcome=env.outcome(),
             score_breakdown=env.score_breakdown(), rewards=[1., -1.])
    cfg = signal(r)
    validate_signal(cfg, game)
    row = terminal_signals(r, 0, cfg)
    assert any(row['mask']) and np.isfinite(row['targets']).all()
    if game in SCORELESS:
        assert 'score_margin' not in cfg['schema']['names']
        with pytest.raises(ValueError): validate_signal(signal(r, 'mixed'), game)
    else:
        assert 'score_margin' in cfg['schema']['names']


def test_absent_components_are_masked_not_fabricated():
    cfg = signal()
    r = result(); r['score_breakdown'][0].pop('animals')
    row = terminal_signals(r, 0, cfg)
    i = cfg['schema']['names'].index('own.animals')
    assert row['targets'][i] == 0 and not row['mask'][i]
    broken = deepcopy(cfg); broken['schema']['scales'][0] = 0
    with pytest.raises(ValueError): validate_signal(broken, 'harmonies')
    with pytest.raises(ValueError): validate_signal(cfg, 'harmonies', ['wrong'])


def test_multitask_checkpoint_policy_identity_padding_and_encoder_gradient(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(1711)
    cls = game_class('toy_battle')
    parent = CandidateActorCritic(16)
    model = CandidateActorCritic(16, ('score', 'component'))
    model.load_state_dict(parent.state_dict(), strict=False)
    rows = [dict(x=np.ones((2, 24), np.float32), prior=np.array([0., -1.], np.float32)),
            dict(x=np.zeros((7, 24), np.float32), prior=np.zeros(7, np.float32))]
    for a, b in zip(parent(*collate(rows,'cpu')), model(*collate(rows,'cpu'))):
        assert torch.equal(a, b)
    single = model(*collate(rows[:1], 'cpu'), return_auxiliary=True)[2]
    padded = model(*collate(rows, 'cpu'), return_auxiliary=True)[2]
    torch.testing.assert_close(single[0], padded[0])
    (padded-1).square().mean().backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.actor.weight.grad is None and model.critic[0].weight.grad is None
    path = tmp_path/'aux.pt'; save_model(path, model, cls, cls.default_weights)
    restored, metadata = load_model(path, cls)
    assert metadata['auxiliary_names'] == ['score', 'component']
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())


def make_episode(tmp_path, mode='aux', seat=0):
    torch.set_num_threads(1); torch.manual_seed(1711); random.seed(1711)
    cls = game_class('toy_battle')
    cfg = signal(result('toy_battle', (7, 3)), mode)
    model = CandidateActorCritic(16, cfg['schema']['names'])
    path = tmp_path/f'{mode}_{seat}.pt'; save_model(path, model, cls, cls.default_weights)
    job = dict(game='toy_battle', path=str(path), self_play=False, learner_seat=seat,
               seed=pool_seed('smoke',1711,index=seat), opponent=dict(kind='heuristic'),
               opponent_id='frozen', policy_version='initial', max_seconds=20,
               gae_lambda=1., validate=True, reward_signal=cfg)
    episode = rollout(job)
    assert episode['result']['status'] == 'complete'
    return model, cfg, job, episode


@pytest.mark.parametrize('mode', ['win', 'aux', 'mixed'])
@pytest.mark.parametrize('seat', [0, 1])
def test_real_rollout_targets_compact_transport_and_returns(tmp_path, mode, seat):
    model, cfg, job, episode = make_episode(tmp_path, mode, seat)
    rows = episode['records']
    expected = terminal_signals(episode['result'], seat, cfg)
    assert all(r['player_id']==seat for r in rows)
    assert all(r['return']==pytest.approx(expected['training_reward']) for r in rows)
    assert rows[-1]['reward']==expected['training_reward']
    assert all(r['reward']==0 for r in rows[:-1])
    assert audit_behavior(model, rows, 'cpu') < 2e-4
    compact = compact_episode(episode)['records']
    assert all(r['auxiliary_targets']==s['auxiliary_targets'] and r['auxiliary_mask']==s['auxiliary_mask']
               for r,s in zip(rows,compact))
    job['max_seconds'] = 0
    truncated = rollout(job)
    assert not truncated['records']
    assert all('auxiliary_targets' not in r for r in truncated['excluded_records'])


def test_auxiliary_update_exact_recovery_and_masked_loss(tmp_path):
    model,cfg,_,episode = make_episode(tmp_path)
    rows = compact_episode(episode)['records']
    rng = np.random.default_rng(12)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pool = OpponentPool(dict(id='initial', spec={}))
    checkpoint = tmp_path/'checkpoint.pt'
    save_checkpoint(checkpoint,model,opt,rng,pool,{},'cfg',{})
    before = deepcopy(model.state_dict())
    metric = update(model,opt,rows,'cpu',rng,epochs=2,batch_size=8,auxiliary_coef=.25)
    expected = deepcopy(model.state_dict())
    assert np.isfinite(metric['auxiliary_loss']) and metric['auxiliary_loss'] > 0
    assert not torch.equal(before['auxiliary.weight'],expected['auxiliary.weight'])
    assert not torch.equal(before['encoder.0.weight'],expected['encoder.0.weight'])
    restore_checkpoint(checkpoint,model,opt,rng,'cfg')
    update(model,opt,rows,'cpu',rng,epochs=2,batch_size=8,auxiliary_coef=.25)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
    assert signal_diagnostics(model,rows,'cpu')['actor_residual_abs_max'] > 0
    # Masked labels must have no effect on the gradient/update.
    for r in rows: r['auxiliary_mask'][0] = False
    restore_checkpoint(checkpoint,model,opt,rng,'cfg')
    update(model,opt,rows,'cpu',rng,epochs=1,batch_size=8,auxiliary_coef=.25)
    masked = deepcopy(model.state_dict())
    for r in rows: r['auxiliary_targets'][0] = 1000.
    restore_checkpoint(checkpoint,model,opt,rng,'cfg')
    update(model,opt,rows,'cpu',rng,epochs=1,batch_size=8,auxiliary_coef=.25)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in masked.items())


def test_zero_aux_coefficient_preserves_legacy_ppo_update(tmp_path):
    model,_,_,episode = make_episode(tmp_path, 'win')
    legacy = CandidateActorCritic(model.hidden)
    legacy.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('auxiliary.')})
    rows = episode['records']
    initial_aux = model.auxiliary.weight.detach().clone()
    for m in (model, legacy):
        update(m,torch.optim.Adam(m.parameters(),lr=1e-3),rows,'cpu',np.random.default_rng(4),
               epochs=2,batch_size=8,auxiliary_coef=0.)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in legacy.state_dict().items())
    assert torch.equal(initial_aux,model.auxiliary.weight)


def test_paired_score_report_preserves_seats_and_missing_game_denominators():
    from boardbench.families.reward_report import score_difference
    controls, candidates = [], []
    for seed in range(4):
        for seat in (0, 1):
            r=result(); r.update(seed=seed,learner_seat=seat,opponent_id='frozen')
            controls.append(r)
            c=deepcopy(r); c['outcome']['scores'][seat]+=5
            candidates.append(c)
    paired=score_difference(candidates,controls)
    assert paired['complete']
    assert paired['signals']['own_score']['mean']==5.
    assert paired['signals']['score_margin']['interval']==[5.,5.]
    assert not score_difference(candidates[:-1],controls)['complete']
