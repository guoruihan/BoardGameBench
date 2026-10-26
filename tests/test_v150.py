from copy import deepcopy
import time
import numpy as np
import pytest
import torch
from boardbench.artifacts.store import digest,write_json
from boardbench.v140.encoding import FEATURES
from boardbench.v150.value import Heads,Value
from boardbench.v150.train import Learner,pair_loss
from boardbench.v150.data import components
from boardbench.v150.worker import play,validate
from test_v146 import model,job
from test_v129 import artifact


def test_components_real_both_seats_and_negative_terrain(job):
    for seat in (0,1):
        result,trace=play({**job,'seat':seat,'environment_seed':150000001,'split':'train'})
        targets=components(trace)
        actual=[r['mc_return'] for r in trace['decisions'] if r['seat']==seat]
        assert np.allclose(np.asarray(targets).sum(-1),actual)
        assert result['natural_terminal']
        broken=deepcopy(trace);broken['split']='development'
        with pytest.raises(ValueError,match='training'):components(broken)
        broken=deepcopy(trace);next(r for r in broken['decisions'] if r['seat']==seat)['mc_return']+=1
        with pytest.raises(ValueError,match='conservation'):components(broken)
    # Deliberately negative future terrain is a valid regression target, not clipped.
    trace['decisions'][0]['observation']['score_breakdown'][0]['total']=1000
    trace['decisions'][0]['mc_return']=(trace['scores'][0]-1000)/150
    trace['seat']=0
    assert components(trace)[0][0]<0


def test_head_conservation_and_grouped_ranking_gradient():
    head=Heads();x=torch.randn(8,FEATURES,dtype=torch.float64);base=torch.randn(8,dtype=torch.float64)
    parts,rank=head(x,base)
    assert torch.equal(parts.sum(-1),base)
    assert torch.allclose(rank,x[:,-4]+base)
    p=torch.tensor([.1,.2,.3,.6],dtype=torch.float64,requires_grad=True)
    q=p.detach().flip(0);groups=torch.tensor([0,0,1,1])
    assert pair_loss(q,q,groups)==0
    loss=pair_loss(p,q,groups);loss.backward();assert p.grad.abs().sum()>0


@pytest.fixture
def config(job,tmp_path):
    # Synthetic tensors test optimization/splits only; experiment data always
    # comes through replay-audited complete trajectories and public branches.
    rng=np.random.default_rng(150);seeds=np.repeat(np.arange(150000000,150000006),4)
    x=rng.normal(0,.1,(24,FEATURES)).astype(np.float32);y=np.stack((.1+x[:,0]*.03,.2+x[:,1]*.04),-1)
    data=tmp_path/'data.npz';np.savez_compressed(data,x=x,y=y,seed=seeds)
    dm=tmp_path/'data.json';write_json(dm,dict(format='v150_components',split='train',path=str(data),sha256=digest(data),games=[dict(opponent_sha256=job['opponent_sha256'])]))
    pairs=tmp_path/'pairs.npz';np.savez_compressed(pairs,x=x,q=.7+x[:,0]*.3,group=np.repeat(np.arange(6),4),seed=seeds)
    pm=tmp_path/'pairs.json';write_json(pm,dict(format='v150_habitat_pairs',split='train',path=str(pairs),sha256=digest(pairs)))
    return dict(base_model=job['student'],base_sha256=job['student_sha256'],dataset=str(dm),dataset_sha256=digest(dm),
        pairs=str(pm),pairs_sha256=digest(pm),seed=150,steps=2,batch=8,learning_rate=.0001,
        component_loss=1.,rank_loss=1.,deadline_unix=time.time()+100)


def test_learning_gradients_exact_resume_and_no_base_change(config):
    t=Learner(config,'cpu');base=deepcopy(t.base.state_dict());head=deepcopy(t.head.state_dict())
    row=t.step();assert row['gradient_norm']>0
    assert any(not torch.equal(v,t.head.state_dict()[k]) for k,v in head.items())
    saved=deepcopy(t.state());first=t.step();expected=deepcopy(t.head.state_dict())
    t.restore(saved);second=t.step()
    assert first==second and all(torch.equal(v,t.head.state_dict()[k]) for k,v in expected.items())
    assert all(torch.equal(v,t.base.state_dict()[k]) for k,v in base.items())
    assert all(s%5==0 for s in t.validation_seeds)
    assert all(t.pair_groups[g][0]//4 not in (0,5) for g in t.train_groups)
    assert np.isfinite(t.validate()['objective'])


def test_train_cli_export_and_value_perspective(config,job,tmp_path):
    from boardbench.v150.train import run
    from boardbench.v138.baseline import Planner
    from boardbench.environments.harmonies_duo import FastHarmoniesDuo
    out=tmp_path/'fit';run(config,out,'cpu')
    assert (out/'done.json').exists() and (out/'device_gate.json').exists()
    value=Value(out/'value_manifest.json',digest(out/'value_manifest.json'),None,job['opponent_sha256'],rank_mix=.5)
    obs,_=FastHarmoniesDuo().reset(150900001);plans=[Planner(),Planner()]
    assert np.isfinite(value.batch([obs],[plans],0)[0])
    with pytest.raises(ValueError,match='focal'):value.batch([obs],[plans],1)
    assert value.batch([{**obs,'terminated':True,'scores':[108,71]}],[plans],1)==[71.]
    with pytest.raises(ValueError):validate({**job,'environment_seed':250000000,'split':'train'})


def test_adaptation_preserves_score_objective_and_expands_after_stalls():
    from boardbench.v150.campaign import adapt
    s={'incumbent':dict(terrain=56,animal=28,fallback_rate=0), 'rollout_1.0':dict(terrain=64,animal=25)}
    a=adapt(s,'incumbent',2)
    assert a['terrain_weight']==2 and a['animal_weight']==1 and a['expanded_environments']==32
    assert a['width']==8 and a['branch_samples']==4
    assert a['continuation']=='actor'


def test_gpu_dispatch_never_uses_an_unapproved_idle_card(monkeypatch,tmp_path):
    from boardbench.v149 import campaign
    run=campaign.Campaign.__new__(campaign.Campaign)
    run.c=dict(gpu=0,allowed_gpus=[0],deadline_unix=time.time()+60);run.stop=False
    monkeypatch.setattr(campaign.subprocess,'check_output',lambda *a,**k:'0, 30000, 90\n1, 14, 0\n')
    monkeypatch.setattr(campaign.time,'sleep',lambda *_:None)
    with pytest.raises(RuntimeError,match='no idle GPU'):
        run.execute([],tmp_path/'out',tmp_path/'out/done.json',0,gpu=True)
