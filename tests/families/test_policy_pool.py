from copy import deepcopy
import json
import random
import time

import numpy as np
import pytest
import torch

from boardbench.families.neural import CandidateActorCritic, save_model, finish_returns
from boardbench.families.opponent_pool import OpponentPool, pool_seed, sampling_probabilities
from boardbench.families.pool_evaluator import paired_difference, panel_summary, bounded_jobs
from boardbench.families.pool_training import (audit_behavior, rollout, save_checkpoint,
                                             restore_checkpoint, write_compressed)
from boardbench.families.ppo import update
from boardbench.families.registry import game_class
from boardbench.families.seeds import experiment_seed


def entry(key, quality=.5, fingerprint=None):
    return dict(id=key, spec={'kind':'neural', 'path':key}, quality=quality,
                fingerprint=fingerprint or ['a']*20, panel_vector=[quality])


def test_sampler_decays_only_completed_training_outcomes_and_has_uniform_floor():
    pool=OpponentPool(entry('initial'))
    pool.add(entry('new'));pool.select()
    pool.update_training([('initial',1.)]*100+[('new',0.)]*2)
    ids, p=pool.distribution()
    assert p.sum()==pytest.approx(1.)
    assert p[ids.index('new')]>p[ids.index('initial')]
    assert min(p)>=.5/len(ids)
    before=deepcopy(pool.state)
    assert OpponentPool.restore(before).distribution()[1]==pytest.approx(p)
    pool.update_training([])
    assert pool.state['statistics']['initial']['games']==pytest.approx(95.)
    with pytest.raises(ValueError): pool.update_training([('initial',None)])


def test_pool_roles_quality_filter_diversity_and_archive_rotation():
    pool=OpponentPool(entry('initial'),capacity=5)
    pool.add(entry('best',.8),confirmed_best=True)
    pool.add(entry('poor',.1, ['b']*20))
    pool.add(entry('diverse',.75,['c']*20))
    pool.add(entry('latest',.2))
    active=pool.select()
    assert {'initial','best','latest','diverse'}<=set(active)
    assert 'poor' not in active and 'poor' in pool.state['members']
    assert pool.audit_members()==['poor']
    counter=entry('counter',.1,['d']*20);counter['confirmed_counter']=True
    pool.add(counter);pool.select()
    assert 'counter' in pool.state['active']


def test_namespaces_disjoint_and_replicates_really_change_seeds():
    seeds={pool_seed(s,r,i,j) for s in ('train','quick','confirm','fingerprint','audit','smoke')
           for r in (0,1,1701,4095) for i in (0,2**24-1) for j in (0,4095)}
    assert len(seeds)==6*4*2*2 and max(seeds)<2**53
    assert min(seeds)>experiment_seed('ppo_confirmation',2**32-1,2**16-1)
    with pytest.raises(ValueError):pool_seed('train',4096)


def result(seed,seat,reward,opponent='old',status='complete'):
    return dict(seed=seed,learner_seat=seat,opponent_id=opponent,status=status,
                rewards=[reward,-reward] if seat==0 else [-reward,reward],seconds=.01)


def test_paired_statistics_preserve_swapped_seats_and_missing_games():
    a=[result(s,p,1.) for s in range(8) for p in (0,1)]
    b=[result(s,p,-1.) for s in range(8) for p in (0,1)]
    diff=paired_difference(a,b)
    assert diff['mean']==1 and diff['interval']==[1.,1.]
    summary=panel_summary(a,['old'],8)
    assert summary['all_complete'] and summary['seed_groups']==8
    a[0]['status']='truncated'
    assert not paired_difference(a,b)['complete']
    assert not panel_summary(a,['old'],8)['all_complete']
    assert not panel_summary(b[:-1],['old'],8)['all_complete']


def test_scoreless_game_keeps_win_result_without_invented_numeric_scores():
    rows=[dict(result(1,seat,1.),outcome={'scores':[None,None]},
               policy_seconds=[.2,.1],policy_max_seconds=[.01,.01],seat_decisions=[20,10])
          for seat in (0,1)]
    summary=panel_summary(rows,['old'],1)
    assert summary['all_complete'] and summary['quality']==1.
    assert summary['opponents']['old']['own_score'] is None
    assert summary['opponents']['old']['opponent_score'] is None
    assert summary['opponents']['old']['mean_policy_seconds']==pytest.approx(.01)


def test_deadline_never_submits_an_entire_long_queue():
    class Executor:
        calls=0
        def map(self,fn,jobs,chunksize):
            self.calls+=1
            return [fn(j) for j in jobs]
    ex=Executor()
    def fn(job):
        assert job['max_seconds']<=8
        return job
    rows=bounded_jobs(ex,fn,[{}]*20,time.time()+10,3,lambda:ex.calls>=1)
    assert len(rows)==3 and ex.calls==1


def test_real_rollout_update_frozen_opponent_and_exact_resume(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(17);random.seed(17)
    cls=game_class('toy_battle');model=CandidateActorCritic()
    path=tmp_path/'initial.pt';save_model(path,model,cls,cls.default_weights)
    job=dict(game='toy_battle',path=str(path),self_play=False,learner_seat=0,
        seed=pool_seed('smoke',17),opponent={'kind':'neural','path':str(path),'stochastic':True},
        opponent_id='initial',policy_version='frozen0',max_seconds=20,gae_lambda=.95,validate=True)
    initial=path.read_bytes()
    played=rollout(job);rows=played['records']
    assert played['result']['status']=='complete' and rows
    assert all(r['player_id']==0 and r['policy_version']=='frozen0' for r in rows)
    assert sum(r['terminated'] for r in rows)==1
    assert all(r['reward']==0 for r in rows[:-1])
    assert audit_behavior(model,rows,'cpu')<1e-4
    pool=OpponentPool(entry('initial'));rng=np.random.default_rng(17)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    checkpoint=tmp_path/'checkpoint.pt'
    save_checkpoint(checkpoint,model,opt,rng,pool,{'next_update':0},'cfg',{})
    before={k:v.clone() for k,v in model.state_dict().items()}
    update(model,opt,rows,'cpu',rng,epochs=2,batch_size=8)
    expected={k:v.clone() for k,v in model.state_dict().items()}
    assert any(not torch.equal(before[k],expected[k]) for k in before)
    assert path.read_bytes()==initial
    restore_checkpoint(checkpoint,model,opt,rng,'cfg')
    update(model,opt,rows,'cpu',rng,epochs=2,batch_size=8)
    assert all(torch.equal(expected[k],v) for k,v in model.state_dict().items())
    with pytest.raises(ValueError):restore_checkpoint(checkpoint,model,opt,rng,'different')
    broken=deepcopy(rows);broken[0]['old_logp']+=1
    with pytest.raises(ValueError):audit_behavior(model,broken,'cpu')
    job['max_seconds']=0
    truncated=rollout(job)
    assert truncated['result']['status']=='truncated' and not truncated['records']


def test_return_uses_own_chain_and_complete_match_result():
    rows=[dict(value=.4),dict(value=-.7),dict(value=.1)]
    finish_returns(rows,-1,gamma=1,gae_lambda=1)
    assert [r['return'] for r in rows]==pytest.approx([-1,-1,-1])
