from copy import deepcopy
import os

import numpy as np
import pytest
import torch

from test_v127 import config
from test_v129 import artifact
from test_v134 import planning
from test_v135 import advice_config
from test_v136 import skill_config
from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.v12.encoding import encode
from boardbench.v130.pipeline import checkpoint_spec
from boardbench.v134.plans import candidates, encode_plan, K, PLAN_DIM
from boardbench.v136.training import Trainer as SkillTrainer
from boardbench.v136.probe import continuations
from boardbench.v137.training import Trainer, METHODS
from boardbench.v137.data import load, diagnostic, own_states, branch_batch, build
from boardbench.v137.campaign import new_manifest, summarize


@pytest.fixture
def selection_config(skill_config, tmp_path):
    t=SkillTrainer(skill_config); t.import_sampling_continuation(skill_config['continuation']); t.batch()
    parent=tmp_path/'skills_parent.pt';t.save(parent)
    seeds=t.train_seeds[:4]; origin=t.ppo_actions;t.close()
    x=[];plans=np.zeros((4,K,PLAN_DIM),np.float32);mask=np.zeros((4,K),bool)
    for i,seed in enumerate(seeds):
        obs,_=Harmonies().reset(seed);x.append(encode(obs,allow_stop=False)['features'])
        for j,plan in enumerate(candidates(obs)):plans[i,j]=encode_plan(obs,plan);mask[i,j]=True
    q=np.broadcast_to((100+np.arange(K)*2)[None,:,None],(4,K,2)).copy().astype(np.float32)
    data=tmp_path/'goals.npz'
    np.savez_compressed(data,x=np.asarray(x,np.float32),plans=plans,mask=mask,q_fit=q,q_verify=q,
                        seed=np.asarray(seeds),validation=np.array([False,False,False,True]),start_score=np.zeros(4,np.float32))
    return {**skill_config,'selection_version':'1.3.7','selection_method':'control','skill_method':'skills','arm':'persistent',
            'goal_dataset':{'path':str(data),'sha256':digest(data)},'selection_origin_actions':origin,
            'manager_warmup_actions':0,'goal_fit_updates':4,'goal_temperature':5.,'joint_skill_bc':.005,
            'continuation':checkpoint_spec(parent)}


def configured(c,method):
    return {**c,'selection_method':method,'arm':'persistent' if method=='control' else 'manager',
            'skill_method':'skills' if method=='control' else 'manager'}


def check_resume(config,method,tmp_path,device):
    c=configured(config,method);t=Trainer(c,device);t.import_sampling_continuation(c['continuation'])
    saved=torch.load(c['continuation']['checkpoint'],weights_only=False)
    assert all(torch.equal(v.cpu(),saved['model'][k].cpu()) for k,v in t.model.state_dict().items())
    before=deepcopy(t.model.state_dict());row=t.batch()
    assert row['initial_ratio_max_error']<1e-4
    if method=='selector':
        assert row['worker_frozen'] and row['actor_change_l2']==0
        assert all(torch.equal(v,t.model.state_dict()[k]) for k,v in before.items())
    else: assert row['actor_change_l2']>0
    if method!='control':assert row['manager_change']>0 and row['manager_ratio_error']<1e-4
    p=tmp_path/'resume.pt';t.save(p)
    expected=t.batch();weights=deepcopy(t.model.state_dict());high=deepcopy(t.manager.state_dict());t.close()
    restored=Trainer(c,device);restored.load(p);actual=restored.batch()
    assert actual['games']==expected['games']
    for obj,state in ((restored.model,weights),(restored.manager,high)):
        for k,v in state.items():torch.testing.assert_close(v,obj.state_dict()[k],atol=0,rtol=0)
    rng=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if device=='cuda' else []
    restored.evaluate(c['curve_seeds'])
    assert torch.equal(rng,torch.get_rng_state())
    if cuda:assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
    restored.close()


@pytest.mark.parametrize('method',METHODS)
def test_update_resume(selection_config,method,tmp_path):
    check_resume(selection_config,method,tmp_path,'cpu')


@pytest.mark.skipif(os.environ.get('BOARDBENCH_V137_CUDA')!='1',reason='authorized CUDA only')
@pytest.mark.parametrize('method',METHODS)
def test_cuda_resume(selection_config,method,tmp_path):
    check_resume({**selection_config,'workers':2,'batch_episodes':8},method,tmp_path,'cuda')


def test_joint_warmup_and_parent_alignment(selection_config):
    t=Trainer(selection_config);t.import_sampling_continuation(selection_config['continuation'])
    saved=torch.load(selection_config['continuation']['checkpoint'],weights_only=False)
    baseline=SkillTrainer(saved['config']);baseline.model.load_state_dict(saved['model'])
    assert t.evaluate(t.train_seeds[:2])['games']==baseline.evaluate(t.train_seeds[:2])['games']
    t.close();baseline.close()
    c={**configured(selection_config,'joint'),'manager_warmup_actions':1000000}
    t=Trainer(c);t.import_sampling_continuation(c['continuation']);before=deepcopy(t.model.state_dict())
    row=t.batch();assert row['worker_frozen'] and row['manager_change']>0
    assert all(torch.equal(v,t.model.state_dict()[k]) for k,v in before.items())
    t.config['manager_warmup_actions']=0
    assert t.batch()['actor_change_l2']>0;t.close()


def test_independent_future_values_not_used_in_training(selection_config,tmp_path):
    c=configured(selection_config,'selector')
    a=Trainer(c);a.import_sampling_continuation(c['continuation']);weights=deepcopy(a.manager.state_dict());a.close()
    with np.load(c['goal_dataset']['path']) as f:data={k:f[k].copy() for k in f.files}
    data['q_verify'][:]=7777;data['q_fit'][data['validation']]=8888
    path=tmp_path/'changed_audit.npz';np.savez_compressed(path,**data)
    b=Trainer({**c,'goal_dataset':{'path':str(path),'sha256':digest(path)}})
    b.import_sampling_continuation(c['continuation'])
    assert all(torch.equal(v,b.manager.state_dict()[k]) for k,v in weights.items());b.close()
    with pytest.raises(ValueError,match='leakage'):load(c['goal_dataset'],[])


def test_public_counterfactual_parity_and_planning_states(selection_config):
    t=Trainer({**selection_config,'record_games_per_batch':128});t.import_sampling_continuation(selection_config['continuation'])
    states,traces,_=own_states(t,4,0)
    assert states and all(r['environment_seed'] in t.train_seeds for r in states)
    assert all(trace['planning_steps'][0]==0 for trace in traces)
    obs=states[0]['observation'];bank=candidates(obs);old=deepcopy(obs)
    specs=[{'state':0,'candidate':i,'repeat':0,'future_seed':159} for i in range(min(3,len(bank)))]
    rows,_=branch_batch(t,states,specs)
    reference=[{'state':0,'environment_seed':states[0]['environment_seed'],'future_seed':159,'branch':s['candidate'],
                'observation':obs,'plan':bank[s['candidate']]} for s in specs]
    other=continuations(t,reference)['rows']
    assert [r['score'] for r in rows]==[r['score'] for r in other] and obs==old
    t.close()


def test_continuation_refresh_preserves_worker(selection_config,tmp_path):
    c=configured(selection_config,'joint');t=Trainer(c);t.import_sampling_continuation(c['continuation']);t.batch()
    path=tmp_path/'stage1.pt';t.save(path);worker=deepcopy(t.model.state_dict())
    expected=t.batch();weights=deepcopy(t.model.state_dict());t.close()
    next_config={**c,'continuation':checkpoint_spec(path),'target_actions':9999,'max_seconds':900,'milestones':[9999]}
    same=Trainer(next_config);same.import_sampling_continuation(next_config['continuation'])
    assert same.batch()['games']==expected['games']
    assert all(torch.equal(v,same.model.state_dict()[k]) for k,v in weights.items());same.close()
    with np.load(c['goal_dataset']['path']) as f:data={k:f[k].copy() for k in f.files}
    data['q_fit'][:,2]+=10
    new=tmp_path/'refresh.npz';np.savez_compressed(new,**data)
    refreshed=Trainer({**next_config,'selection_method':'refresh','goal_dataset':{'path':str(new),'sha256':digest(new)}})
    refreshed.import_sampling_continuation(next_config['continuation'])
    assert 'goal_refresh' in refreshed.lineage
    assert all(torch.equal(v,refreshed.model.state_dict()[k]) for k,v in worker.items());refreshed.close()


def test_fresh_manifest_and_noise_diagnostic(selection_config):
    old=read_json(selection_config['split_manifest']);new=new_manifest(old,selection_config['split_sha256'],8)
    assert new['sets']['train']==old['sets']['train'] and new['sets']['test']==old['sets']['test']
    assert all(70000000<=s<80000000 for s in new['sets']['development']+new['sets']['confirmation'])
    data=load(selection_config['goal_dataset'],old['sets']['train'])
    data['q_verify'][:]=100
    result=diagnostic(data)
    assert result['fit_selected_gain']>0 and result['independent_future_gain']['mean']==0


def test_final_summary_pairing():
    seeds=[1,2];results={}
    for name in ['parent','parent_b','strong_reference']+[m+'_'+s for m in METHODS for s in 'ab']:
        results[name]={'mean_score':100.,'games':[{'environment_seed':s,'score':100.,'animal_score':25.} for s in seeds]}
    report=summarize(results,seeds);assert report['methods']['joint']['vs_continued_ppo']['mean']==0.
    results['joint_a']['games'].reverse()
    with pytest.raises(ValueError,match='alignment'):summarize(results,seeds)


def test_shared_stage_forks_only_when_labels_change(tmp_path, monkeypatch):
    from boardbench.v137 import campaign
    root = tmp_path/'outputs/v137/test'
    write_json(root/'config.json', {'stage2_actions':4000000, 'stage2_seconds':4200})
    jobs = []
    for method in METHODS[:-1]:
        for suffix in 'ab':
            name = method+'_'+suffix
            jobs.append([name, {'selection_method':method, 'goal_dataset':{'path':'initial_'+suffix}}])
            trial = root/'stage1/trials'/name
            write_json(trial/'evaluations/final.json', {'mean_score':100.})
            write_json(trial/'final.json', {'ppo_actions':1234})
    write_json(root/'stage1/jobs.json', jobs)
    monkeypatch.setattr(campaign, 'checkpoint_spec', lambda p: {'checkpoint':str(p)})
    monkeypatch.setattr(campaign, 'labels', lambda *a: {s:{'path':'refreshed_'+s} for s in 'ab'})
    children = dict(read_json(campaign.stage2(root)/'jobs.json'))
    assert len(children)==8
    for suffix in 'ab':
        joint, refresh = children['joint_'+suffix], children['refresh_'+suffix]
        assert joint['continuation']==refresh['continuation']
        assert joint['goal_dataset']['path']=='initial_'+suffix
        assert refresh['goal_dataset']['path']=='refreshed_'+suffix
        assert joint['target_actions']==refresh['target_actions']==4001234
