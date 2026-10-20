import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from test_v127 import config
from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.v127.training import Trainer as ParentTrainer
from boardbench.v130.training import export_checkpoint
from boardbench.v130.teacher import Student64
from boardbench.v130.pipeline import report
from boardbench.v131.counterfactual import paired_stats, fingerprint, evaluate_state, summarize
from boardbench.v131.campaign import prepare_plan, make_jobs
from boardbench.v131.data import weights
from boardbench.v131.training import Trainer, run
from boardbench.v130.pipeline import checkpoint_spec
from boardbench.v131.launch import build_jobs,report as learning_report
from boardbench.v127.launch import validate_allocation


@pytest.fixture
def guided(config,tmp_path):
    with np.load(config['dataset']) as z:d={k:z[k].copy() for k in z.files}
    d['student_y']=np.array([next(i for i in np.flatnonzero(mask) if i!=y) for mask,y in zip(d['mask'],d['y'])])
    d.update(habitat_mean=np.full(3,8.),habitat_se=np.ones(3),student_mean=np.full(3,6.),
             student_se=np.ones(3),samples=np.full(3,32))
    p=tmp_path/'paired.npz';np.savez(p,**d)
    return {**config,'guidance_version':'1.2.11','arm':'student','bc_weight':.02,
            'paired_dataset':{'path':str(p),'sha256':digest(p)},'minimum_guidance_rows':1,
            'guidance_z':1.96,'guidance_minimum_gain':1.,'guidance_gain_scale':5.}


def test_report_tolerates_comparison_before_summary(tmp_path):
    trial = tmp_path/'round_01/ppo/trials/control_a'; trial.mkdir(parents=True)
    metric = {'games': [{'environment_seed': 3, 'score': 5}], 'mean_score': 5, 'animal_mean': 1}
    write_json(trial/'final_comparison.json', {k: metric for k in ('initial','latest','best')})
    state = {'phase': 'running'}
    report(tmp_path, state)
    assert read_json(tmp_path/'state.json')['phase'] == 'running'
    assert len(read_json(tmp_path/'results.json')) == 1
    assert not (tmp_path/'retained_best.json').exists()


def test_paired_variance_and_environment_aggregation():
    assert paired_stats([3.,3.,3.])['se'] == 0
    assert paired_stats([1.,3.])['mean'] == 2
    rows = [{'status':'completed', 'natural_terminal':True, 'environment_seed':s,
             'statistics':{k:paired_stats([v,v]) for k in ('habitat','student','interaction')}}
            for s,v in [(1,2),(1,2),(2,8)]]
    summary = summarize(rows)
    assert summary['student']['mean'] == 5  # not pooled-state mean 4
    with pytest.raises(ValueError): paired_stats([float('nan'),2])


def test_four_branches_fresh_public_futures(config, tmp_path):
    t = ParentTrainer(config); t.save(tmp_path/'parent.pt'); t.close()
    meta = export_checkpoint(tmp_path/'parent.pt', tmp_path/'student.npz')
    student = Student64(tmp_path/'student.npz',meta['sha256'])
    e = Harmonies(); obs,_ = e.reset(123)
    # Late-game state makes the genuine four-way test small.
    history=[]
    while not obs['terminated']:
        history.append(obs)
        obs = e.step(student.decide(obs)).observation
    obs=history[-5]
    action = student.decide(obs)
    job = {'id':'fixture', 'checkpoint':str(tmp_path/'student.npz'), 'checkpoint_sha256':meta['sha256'],
           'observation':obs,'observation_sha256':fingerprint(obs),'student_action':action,'teacher_action':action,
           'timeout_seconds':60,'future_seed':131,'repeats':2,'environment_seed':123}
    r = evaluate_state(job,tmp_path/'result')
    assert r['status']=='completed' and r['natural_terminal']
    assert len({s['future_seed'] for s in r['samples']}) == 2
    for s in r['samples']:
        for policy in ('habitat','student'):
            assert s['values'][policy+'_teacher'] == s['values'][policy+'_student']
    assert r['statistics']['student']['mean'] == r['statistics']['habitat']['mean'] == 0
    assert 'observation' not in r


def test_state_plan_rejects_test_leakage_and_tampering(config,tmp_path):
    manifest=read_json(config['split_manifest'])
    t=ParentTrainer(config);t.save(tmp_path/'parent.pt');t.close()
    meta=export_checkpoint(tmp_path/'parent.pt',tmp_path/'student.npz')
    rows={'states':[{'id':'s1','environment_seed':manifest['sets']['train'][0],
        'student_checkpoint_sha256':meta['sha256'],'observation':{'public':1}}]}
    p=tmp_path/'states.json';write_json(p,rows)
    raw={'split':'train_calibration','states':str(p),'states_sha256':digest(p),
         'checkpoint':str(tmp_path/'student.npz'),'checkpoint_sha256':meta['sha256'],
         'repeats':2,'job_seconds':30,'sample_seed':8}
    plan=prepare_plan(raw,manifest);a=make_jobs(plan);b=make_jobs({**plan,'sample_seed':9})
    assert a[0]['future_seed']!=b[0]['future_seed']
    rows['states'][0]['environment_seed']=manifest['sets']['test'][0];write_json(p,rows)
    with pytest.raises(ValueError,match='changed'):make_jobs(plan)
    with pytest.raises(ValueError,match='leakage'):prepare_plan({**raw,'states_sha256':digest(p)},manifest)


def test_advantage_filter_ignores_ties_noise_and_negative():
    np.testing.assert_allclose(weights([0,1,-4,5,8,100],[0,0,0,3,1,0]),[0,0,0,0,1,1])
    with pytest.raises(ValueError):weights([3],[float('nan')])


def test_guidance_real_update_gradients_and_exact_resume(guided,tmp_path):
    t=Trainer(guided);t.batch();row=t.batch()
    assert row['bc_loss']>0 and row['bc_weight']==.02
    assert row['guidance_gradient_norm_probe']>0 and -1.000001<=row['guidance_policy_gradient_cosine_probe']<=1.000001
    assert row['initial_ratio_max_error']<1e-4 and row['actor_change_l2']>0
    p=tmp_path/'resume.pt';t.save(p);expected=t.batch();model={k:v.clone() for k,v in t.model.state_dict().items()};t.close()
    b=Trainer(guided);b.load(p);actual=b.batch()
    assert expected['games']==actual['games'] and expected['bc_loss']==actual['bc_loss']
    assert all(torch.equal(v,model[k]) for k,v in b.model.state_dict().items());b.close()


def test_parent_import_preserves_both_optimizers_and_initial_weights(config,guided,tmp_path):
    a=ParentTrainer(config);a.batch();a.batch();a.save(tmp_path/'parent.pt')
    model={k:v.clone() for k,v in a.model.state_dict().items()};a.close()
    b=Trainer(guided);b.import_sampling_continuation(checkpoint_spec(tmp_path/'parent.pt'))
    assert b.optimizer.state and b.critic_optimizer.state
    assert all(torch.equal(v,model[k]) for k,v in b.model.state_dict().items())
    b.close()


def test_guidance_split_mask_and_empty_signal_gates(guided,tmp_path):
    with np.load(guided['paired_dataset']['path']) as z:d={k:z[k].copy() for k in z.files}
    def changed(**values):
        p=tmp_path/'bad.npz';np.savez(p,**{**d,**values})
        return {**guided,'paired_dataset':{'path':str(p),'sha256':digest(p)}}
    with pytest.raises(ValueError,match='insufficient'):Trainer(changed(student_mean=np.zeros(3)))
    with pytest.raises(ValueError,match='leakage'):
        Trainer(changed(seed=np.full(3,read_json(guided['split_manifest'])['sets']['test'][0])))
    with pytest.raises(ValueError,match='invalid paired'):Trainer(changed(student_y=d['y']))
    with pytest.raises(ValueError,match='checksum'):
        Trainer({**guided,'paired_dataset':{**guided['paired_dataset'],'sha256':'wrong'}})


def test_guidance_run_validation_metrics_and_noop(guided,tmp_path):
    result=run(guided,tmp_path/'complete','cpu')
    assert result['status']=='completed' and run(guided,tmp_path/'complete','cpu')==result
    metric=read_json(tmp_path/'complete/evaluations/final.json')
    assert metric['policy']=='greedy_model_only'
    assert metric['guidance_diagnostic']['validation_student']['n']==1


def test_six_job_cpu_cap_and_paired_configuration():
    validate_allocation(list(range(49)),['a','b'],8,3,6)
    with pytest.raises(ValueError):validate_allocation(list(range(48)),['a','b'],8,3,6)
    repo=Path(__file__).resolve().parents[1]
    data=repo/'outputs/v131/continuation_guidance_20261020_a/paired.npz'
    if not data.exists():pytest.skip('real paired research artifacts not installed')
    jobs=dict(build_jobs(read_json(repo/'configs/v131_continuation_guidance.json'),repo,data))
    assert len(jobs)==6 and len({v['continuation']['sha256'] for v in jobs.values()})==1
    for suffix in 'ab':
        variants=[v for k,v in jobs.items() if k.endswith(suffix)]
        for key in ('seed','curve_seeds','final_seeds','target_actions','excluded_train_seeds','paired_dataset'):
            assert all(v[key]==variants[0][key] for v in variants)


def test_learning_report_does_not_read_incomplete_publication(tmp_path):
    write_json(tmp_path/'config.json',{'training':{'target_actions':20}})
    trial=tmp_path/'trials/student_a';trial.mkdir(parents=True)
    write_json(trial/'final_comparison.json',{'latest':{'mean_score':100}})
    state={'phase':'running','jobs':{'student_a':{'status':'running'}}}
    learning_report(tmp_path,state)
    assert read_json(tmp_path/'paired_results.json')==[]
    assert not (tmp_path/'retained_best.json').exists()


@pytest.mark.skipif(os.environ.get('BOARDBENCH_V131_CUDA')!='1',reason='explicit reserved GPU preflight only')
def test_cuda_guidance_exact_update_resume(guided,tmp_path):
    assert torch.cuda.is_available()
    c={**guided,'hidden':256,'batch_episodes':8,'workers':2,'minibatch':256}
    t=Trainer(c,'cuda');t.batch();row=t.batch();assert row['guidance_gradient_norm_probe']>0
    p=tmp_path/'cuda.pt';t.save(p);expected=t.batch();model={k:v.clone() for k,v in t.model.state_dict().items()};t.close()
    b=Trainer(c,'cuda');b.load(p);actual=b.batch()
    assert expected['games']==actual['games'] and expected['bc_loss']==actual['bc_loss']
    assert all(torch.equal(v,model[k]) for k,v in b.model.state_dict().items());b.close()
