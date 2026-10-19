from copy import deepcopy
import os
from pathlib import Path

import pytest
np=pytest.importorskip('numpy')
torch=pytest.importorskip('torch')

from boardbench.artifacts.store import digest,read_json,write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.v12.encoding import encode
from boardbench.v121.collector import EnvironmentPool
from boardbench.v121.registry import make_splits
from boardbench.v126.training import export_model
from boardbench.v127.training import Trainer,advantages,run
from boardbench.v127.launch import validate_allocation,build_jobs


@pytest.fixture
def config(tmp_path):
    manifest=make_splits(train_count=32);p=tmp_path/'splits.json';write_json(p,manifest)
    states=[];masks=[];labels=[]
    for seed in manifest['sets']['train'][:3]:
        e=Harmonies();obs,_=e.reset(seed);d=encode(obs,allow_stop=False)
        states.append(d['features']);masks.append(d['action_mask']);labels.append(np.flatnonzero(d['action_mask'])[0])
    ds=tmp_path/'data.npz'
    np.savez(ds,x=np.asarray(states,np.float32),mask=np.asarray(masks,bool),y=np.array(labels),
             seed=np.array(manifest['sets']['train'][:3]),validation=np.array([False,False,True]))
    return {'research_version':'1.2.7','calibration':True,'hidden':32,'seed':12701,'reward_scale':150.,
            'dataset':str(ds),'dataset_sha256':digest(ds),'split_manifest':str(p),'split_sha256':digest(p),
            'curve_seeds':manifest['sets']['development'][:2],'final_seeds':manifest['sets']['development'][2:4],
            'learning_rate':.0001,'batch_episodes':2,'workers':0,'critic_batches':1,'critic_epochs':4,
            'minibatch':128,'epochs':2,'target_kl':.02,'gae_lambda':.95,'entropy_coef':.005,'bc_weight':0.,
            'target_actions':100,'max_seconds':60,'milestones':[100]}


def test_gae_mc_identity_and_terminal():
    r=[.2,-.1,.7];v=[.5,.3,.1]
    adv,target=advantages(r,v,1.)
    np.testing.assert_allclose(target,[.8,.6,.7]);np.testing.assert_allclose(adv,np.array(target)-v)
    a,t=advantages(r,v,0.);np.testing.assert_allclose(a,[0.,-.3,.6])
    assert t[-1]==pytest.approx(.7)


def test_split_and_mask_admission(config):
    t=Trainer(config);assert len(t.train_seeds)==31
    bad={**config,'curve_seeds':[t.manifest['sets']['test'][0]]}
    with pytest.raises(ValueError,match='development'):Trainer(bad)
    bad={**config,'dataset_sha256':'bad'}
    with pytest.raises(ValueError,match='hash'):Trainer(bad)
    t.close()


def test_critic_warmup_keeps_actor_and_body_exact(config):
    t=Trainer(config);before={k:v.clone() for k,v in t.model.state_dict().items()}
    row=t.batch();assert row['phase']=='critic_warmup' and row['critic_mse_after']<row['critic_mse_before']
    assert t.ppo_actions==0 and t.actions>0
    assert all(g['natural_terminal'] for g in row['games'])
    for k,v in t.model.state_dict().items():
        if not k.startswith('critic.'):assert torch.equal(v,before[k])
    assert not torch.equal(t.model.critic.weight,before['critic.weight']);t.close()


def test_real_reward_ppo_updates_actor_critic_and_no_stop(config):
    t=Trainer(config);t.batch();r=t.batch()
    assert r['phase']=='ppo' and r['actor_change_l2']>0 and r['critic_change_l2']>0
    assert r['initial_ratio_max_error']<1e-4 and r['optimizer_steps']>0
    assert t.ppo_actions>0 and all(g['natural_terminal'] for g in r['games'])
    assert np.isfinite([r[k] for k in ('policy_loss','value_loss','entropy','approx_kl','gradient_norm')]).all()
    assert r['bc_weight']==0;t.close()


def test_probability_gate_and_illegal_action(config):
    t=Trainer(config)
    with EnvironmentPool(2) as p:rows,_=t.collect(p,t.train_seeds[:2])
    corrupted=[tuple(list(r[:3])+[r[3]+.1]+list(r[4:])) for r in rows]
    with pytest.raises(RuntimeError,match='probability'):t.update(corrupted)
    broken=list(rows);r=list(broken[0]);r[1]=r[1].copy();r[1][r[2]]=False;broken[0]=tuple(r)
    with pytest.raises(ValueError,match='mask'):t.update(broken)
    t.close()


def test_resume_matches_next_full_update(config,tmp_path):
    t=Trainer(config);t.batch();t.batch();path=tmp_path/'resume.pt';t.save(path)
    expected=t.batch();weights={k:v.clone() for k,v in t.model.state_dict().items()};t.close()
    other=Trainer(config);other.load(path);actual=other.batch()
    assert expected['games']==actual['games'] and expected['ppo_actions']==actual['ppo_actions']
    for k,v in other.model.state_dict().items():assert torch.equal(v,weights[k])
    other.close()


def test_evaluation_preserves_rng_weights_and_actions(config):
    t=Trainer(config);rng=torch.get_rng_state().clone();before={k:v.clone() for k,v in t.model.state_dict().items()}
    a=t.evaluate(config['curve_seeds']);b=t.evaluate(config['curve_seeds'])
    assert a['games']==b['games'] and torch.equal(rng,torch.get_rng_state())
    assert t.actions==t.updates==0
    assert all(torch.equal(v,before[k]) for k,v in t.model.state_dict().items());t.close()


def test_decay_auxiliary_then_fully_reward_driven(config):
    c={**config,'bc_weight':.05,'bc_decay_actions':1}
    t=Trainer(c);t.batch();r=t.batch();assert r['bc_weight']==.05 and r['bc_loss']>0
    r=t.batch();assert r['bc_weight']==0 and r['bc_loss']==0;t.close()


def test_imitation_import_exact_actor_and_fresh_critic(config,tmp_path):
    t=Trainer(config);t.batch();t.batch();p=tmp_path/'parent'/'best.npz';p.parent.mkdir()
    export_model(t.model,p);weights={k:v.clone() for k,v in t.model.state_dict().items()};t.close()
    write_json(p.parent/'summary.json',{'checkpoint_sha256':digest(p),'hidden':32,'parameter_changed':True,'source_id':'fixture'})
    write_json(p.parent/'config.json',{'dataset_sha256':config['dataset_sha256']})
    child=Trainer({**config,'initial_checkpoint':str(p),'initial_sha256':digest(p)})
    for k,v in child.model.state_dict().items():
        if not k.startswith('critic.'):assert torch.equal(v,weights[k])
    assert child.model.critic.weight.count_nonzero()==0 and not child.optimizer.state
    child.close()


def test_run_completion_noop_and_artifacts(config,tmp_path):
    p=tmp_path/'run';s=run(config,p,device='cpu')
    assert s['status']=='completed' and s['updates']>0
    assert run(config,p,device='cpu')==s
    assert read_json(p/'evaluations'/'initial.json')['ppo_actions']==0
    assert (p/'evaluations'/'final.json').exists() and (p/'best.pt').exists()
    final=read_json(p/'final_comparison.json')
    assert set(final)=={'latest','initial','best','paired_best_minus_initial'}
    assert final['paired_best_minus_initial']['n']==2


def test_cpu_cap_and_eight_paired_configs():
    validate_allocation(list(range(128)),[str(i) for i in range(8)])
    with pytest.raises(ValueError,match='CPU'):validate_allocation(list(range(129)),[str(i) for i in range(8)])
    repo=Path(__file__).resolve().parents[1];c=read_json(repo/'configs/v127_a218_online_ppo.json')
    if not (repo/c['dataset']).exists():pytest.skip('local research artifacts unavailable')
    jobs=dict(build_jobs(c,repo));assert len(jobs)==8
    for suffix in 'ab':
        assert jobs['bc_'+suffix]['initial_sha256']==jobs['anchor_'+suffix]['initial_sha256']
        assert jobs['bc_'+suffix]['seed']==jobs['random_'+suffix]['seed']
    assert all(v['workers']==14 and v['target_actions']==2000000 for v in jobs.values())


@pytest.mark.skipif(os.environ.get('BOARDBENCH_V127_CUDA')!='1',reason='explicit authorized CUDA probe only')
def test_cuda_collection_update_and_resume(config,tmp_path):
    assert torch.cuda.is_available()
    c={**config,'hidden':256,'batch_episodes':8,'workers':2,'minibatch':256}
    t=Trainer(c,'cuda');t.batch();r=t.batch();assert r['actor_change_l2']>0 and r['initial_ratio_max_error']<1e-4
    p=tmp_path/'cuda.pt';t.save(p);expected=t.batch();weights={k:v.clone() for k,v in t.model.state_dict().items()};t.close()
    other=Trainer(c,'cuda');other.load(p);actual=other.batch()
    assert expected['games']==actual['games']
    for k,v in other.model.state_dict().items():torch.testing.assert_close(v,weights[k],rtol=1e-10,atol=1e-10)
    other.close()
