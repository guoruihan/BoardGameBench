from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
np=pytest.importorskip('numpy')
torch=pytest.importorskip('torch')

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.environments.adapters import SearchAccess
from boardbench.solvers.neural import PolicyNetwork
from boardbench.v12.encoding import encode, index
from boardbench.v121.registry import make_splits
from boardbench.v125.policies import GoalContinuation, mixed_candidates
from boardbench.v126.data import targets, build
from boardbench.v126.policy import Student, LearnedSearch
from boardbench.v126.training import objective, train, export_model
from boardbench.v126.campaign import prepare_plan
from boardbench.v126.episode import run_episode
from boardbench.v126.pipeline import arm, Pipeline


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(1)
    model=PolicyNetwork(1346,240,32)
    p=tmp_path/'model.npz';export_model(model,p)
    return p,model


def test_numpy_matches_torch_public_encoding_and_legality(checkpoint):
    p,model=checkpoint;student=Student(p,digest(p))
    engine=Harmonies();obs,_=engine.reset(113);heuristic=GoalContinuation()
    while not obs['terminated']:
        data=encode(obs,allow_stop=False)
        with torch.inference_mode():
            logits,_=model(torch.tensor([data['features']]),torch.tensor([data['action_mask']]))
        public_logits=student.logits(obs);valid=np.array(data['action_mask'])
        np.testing.assert_allclose(public_logits[valid],logits.numpy()[0,valid],rtol=1e-4,atol=1e-6)
        assert student.decide(obs) in obs['legal_actions']
        assert index(obs,student.decide(obs),'slots240_v1')==int(logits.argmax(-1))
        perturbed={**obs,'environment_seed':9999,'search_seed':9999}
        np.testing.assert_array_equal(student.logits(perturbed),public_logits)
        obs=engine.step(heuristic.decide(obs)).observation
    assert student.hits>0


def test_targets_null_not_zero_and_conditional_mask():
    e=Harmonies();obs,_=e.reset(7);a,b,c=obs['legal_actions'][:3]
    candidates=[{'action':a,'samples':4,'mean':100.},{'action':b,'samples':4,'mean':110.},
                {'action':c,'samples':0,'mean':None}]
    x,mask,y,q,qmask=targets(obs,b,candidates)
    assert x.shape==(1346,) and mask.shape==(240,) and qmask.sum()==2
    assert q.sum()==pytest.approx(1) and not qmask[index(obs,c,'slots240_v1')]
    assert q[index(obs,b,'slots240_v1')]>q[index(obs,a,'slots240_v1')]
    logits=torch.zeros((1,240),requires_grad=True)
    loss,bc,aux=objective(logits,torch.tensor([y]),torch.tensor(q[None]),torch.tensor(qmask[None]),.5)
    loss.backward();assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    loss,_,aux=objective(logits,torch.tensor([y]),torch.zeros((1,240)),torch.zeros((1,240),dtype=torch.bool),.5)
    assert torch.isfinite(loss) and aux.item()==0


def test_learned_variants_are_legal_and_dagger_labels_separate(checkpoint):
    p,_=checkpoint;e=Harmonies();obs,_=e.reset(113)
    context=SimpleNamespace(reference_simulator=SearchAccess(lambda seed:Harmonies.from_observation(obs,seed),lambda:None))
    for variant in ('student_roots','student_rollout','student','dagger'):
        solver=LearnedSearch(11,variant,.05,str(p),digest(p))
        before=deepcopy(obs);action=solver.decide(obs,context)
        assert action in obs['legal_actions'] and obs==before
        assert solver.spent>=0 and solver.last_search['cumulative_solver_seconds']==solver.spent
        if variant=='dagger':
            assert solver.last_search['expert_action'] in obs['legal_actions']
            assert action==solver.student.decide(obs)
        if variant=='student_roots':
            roots=solver.rank_candidates(obs)
            assert len(roots)==min(6,len(obs['legal_actions']))
            assert roots[:5]==mixed_candidates(obs)[:5]


def test_student_rollout_evaluates_real_terminal_candidates(checkpoint):
    p,_=checkpoint;e=Harmonies();obs,_=e.reset(7);states=[];heuristic=GoalContinuation()
    while not obs['terminated']:
        states.append(obs);obs=e.step(heuristic.decide(obs)).observation
    obs=states[-12];calls=[]
    class Clock:
        now=0.
        def __call__(self):
            self.now+=.0001
            return self.now
    solver=LearnedSearch(11,'student_rollout',10,str(p),digest(p),clock=Clock())
    context=SimpleNamespace(reference_simulator=SearchAccess(
        lambda seed:Harmonies.from_observation(obs,seed),lambda:calls.append(1)))
    action=solver.decide(obs,context);s=solver.last_search
    assert action in obs['legal_actions'] and solver.student.calls>0
    assert s['completed_rounds']>0 and s['terminal_rollouts']>0
    assert s['simulation_steps']==len(calls)
    assert {c['samples'] for c in s['candidates']}=={s['completed_rounds']}
    assert all(c['mean'] is not None for c in s['candidates'])


@pytest.fixture
def dataset(tmp_path):
    run=tmp_path/'teacher';manifest=make_splits(train_count=8)
    jobs=[]
    for i,seed in enumerate(manifest['sets']['train'][:3]):
        job={'id':f'e{i}','method':'terminal','multiplier':1,'arm':'teacher','environment_seed':seed,
             'search_seed':11,'params':{'mode':'timed','continuation':'mixed_roots','seconds':.03}}
        jobs.append(job);result=run_episode(job,run/'episodes'/job['id'])
        assert result['status']=='completed'
        write_json(run/'resolved'/f'{job["id"]}.json',result)
    write_json(run/'plan.json',{'split':'train_calibration'})
    write_json(run/'status.json',{'phase':'completed','active_count':0})
    write_json(run/'splits.json',manifest);write_json(run/'allocation.json',{'source_id':'test'})
    write_json(run/'jobs.json',jobs)
    p=tmp_path/'dataset.npz';build([run],p,[manifest['sets']['train'][2]])
    return p,run,manifest


def test_data_full_replay_split_and_hash(dataset,tmp_path):
    p,run,manifest=dataset
    with np.load(p) as a:
        assert set(a['seed'][a['validation']]).isdisjoint(a['seed'][~a['validation']])
        assert a['mask'][np.arange(len(a['y'])),a['y']].all()
    write_json(run/'plan.json',{'split':'development'})
    with pytest.raises(ValueError,match='train-split'):
        build([run],tmp_path/'illegal.npz',[manifest['sets']['train'][2]])


def test_real_training_export_and_completed_resume(dataset,tmp_path):
    p,_,_=dataset
    config={'dataset':str(p),'dataset_sha256':digest(p),'seed':11,'hidden':32,'q_weight':.5,
            'epochs':2,'threads':1,'max_seconds':60,'device':'cpu'}
    result=train(config,tmp_path/'train')
    assert result['parameter_changed'] and result['epoch']==2 and result['gpu_seconds']==0
    assert train(config,tmp_path/'train')==result
    Student(result['checkpoint'],result['checkpoint_sha256'])
    with pytest.raises(ValueError,match='config mismatch'):
        train({**config,'hidden':64},tmp_path/'train')


def test_training_recovers_completed_epoch_before_summary(dataset,tmp_path):
    p,_,_=dataset
    config={'dataset':str(p),'dataset_sha256':digest(p),'seed':11,'hidden':32,'q_weight':0.,
            'epochs':2,'threads':1,'max_seconds':60,'device':'cpu'}
    directory=tmp_path/'train';result=train(config,directory)
    # Simulate crash between atomic epoch checkpoint and final summary.
    (directory/'summary.json').rename(directory/'preserved_summary.json')
    recovered=train(config,directory)
    assert recovered['checkpoint_sha256']==result['checkpoint_sha256']
    assert recovered['epoch']==result['epoch'] and recovered['updates']==result['updates']


def test_plan_enforces_split_and_checkpoint(checkpoint):
    p,_=checkpoint;model={'checkpoint':str(p),'checkpoint_sha256':digest(p)}
    raw={'split':'development','environment_indices':[108,109,110,111],'panels':[4],
         'search_seeds':[11,12],'arms':[arm('mixed_roots'),arm('student','student_roots',model)]}
    manifest=make_splits(train_count=8)
    result=prepare_plan(raw,manifest)
    assert result['environment_seeds']==manifest['sets']['development'][108:112]
    raw['arms']=[arm('dagger','dagger',model)]
    with pytest.raises(ValueError,match='train split'): prepare_plan(raw,manifest)
    raw['split']='test'
    with pytest.raises(ValueError,match='confirmation/test'): prepare_plan(raw,manifest)


def test_production_design_seed_isolation():
    config=read_json(Path(__file__).resolve().parents[1]/'configs/v126_search_learning.json')
    groups=[config[k] for k in ('train_indices','validation_indices','student_selection_indices','dagger_indices')]
    assert len(set(sum(groups,[])))==sum(map(len,groups))
    assert set(config['pilot_indices']).isdisjoint(config['fresh_indices'])
    assert config['game_seconds']==900 and config['campaign_cores']<=101
    assert set(config['pilot_indices']+config['fresh_indices']).isdisjoint(range(64,108))


def test_gpu_probe_timeout_is_unavailable_not_pipeline_failure(tmp_path,monkeypatch):
    import subprocess
    pipeline=Pipeline.__new__(Pipeline)
    pipeline.root=tmp_path;pipeline.env={};pipeline.config={'gpu_uuids':['GPU-test']}
    def timeout(*args,**kwargs):
        raise subprocess.TimeoutExpired('nvidia-smi',15)
    monkeypatch.setattr('boardbench.v126.pipeline.subprocess.run',timeout)
    assert pipeline.probe_gpus()==[]
    report=read_json(tmp_path/'gpu_probe.json')
    assert not report['devices'][0]['cuda_execution_ok']
    assert 'unavailable' in report['devices'][0]['reason']


def test_child_campaign_copies_frozen_pipeline_not_mutable_repo(tmp_path,monkeypatch):
    from boardbench.v126 import campaign
    frozen=tmp_path/'outer/source';live=tmp_path/'mutable_worktree'
    monkeypatch.setattr(campaign,'source_root',lambda:frozen)
    assert campaign.snapshot_repository(live,{})==frozen
    parent=tmp_path/'previous_attempt'
    monkeypatch.setattr(campaign,'resume_parent',lambda plan:parent)
    assert campaign.snapshot_repository(live,{'resume_from':str(parent)})==parent/'source'


@pytest.mark.skipif(os.environ.get('BOARDBENCH_V126_CUDA_CHECK')!='1',
                    reason='requires a specifically reserved V126 CUDA device')
def test_real_cuda_training_export_and_epoch_recovery(dataset,tmp_path):
    p,_,_=dataset
    config={'dataset':str(p),'dataset_sha256':digest(p),'seed':11,'hidden':32,'q_weight':.5,
            'epochs':2,'threads':1,'max_seconds':60,'device':'cuda'}
    directory=tmp_path/'cuda_train';result=train(config,directory)
    assert result['device']=='cuda' and result['gpu_seconds']>0 and result['parameter_changed']
    (directory/'summary.json').rename(directory/'preserved_summary.json')
    recovered=train(config,directory)
    assert recovered['updates']==result['updates'] and recovered['checkpoint_sha256']==result['checkpoint_sha256']
    model=PolicyNetwork(1346,240,32).cuda()
    with np.load(result['checkpoint'],allow_pickle=False) as weights:
        model.load_state_dict({k:torch.from_numpy(weights[k]).cuda() for k in weights.files})
    student=Student(result['checkpoint'],result['checkpoint_sha256'])
    e=Harmonies();obs,_=e.reset(113);data=encode(obs,allow_stop=False)
    with torch.inference_mode():
        logits,_=model(torch.tensor([data['features']],device='cuda'),
                       torch.tensor([data['action_mask']],device='cuda'))
    valid=np.array(data['action_mask'])
    np.testing.assert_allclose(student.logits(obs)[valid],logits.cpu().numpy()[0,valid],rtol=1e-4,atol=1e-6)
    assert student.decide(obs) in obs['legal_actions']
