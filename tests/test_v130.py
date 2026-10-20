from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from test_v127 import config
from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.v12.encoding import encode, index
from boardbench.v126.data import build as build_data
from boardbench.v127.training import Trainer as ParentTrainer
from boardbench.v130.training import Trainer, export_checkpoint, run
from boardbench.v130.teacher import Student64, run_episode
from boardbench.v130.campaign import prepare_plan
from boardbench.v130.pipeline import build_jobs, checkpoint_spec, label_indices, validation_indices, paired


@pytest.fixture
def guided(config):
    return {**config,'guidance_version':'1.2.10','correction_fraction':.75,'bc_weight':.2,
            'correction_datasets':[{'path':config['dataset'],'sha256':config['dataset_sha256']}],
            'arm':'correct_high','replicate':0}


def test_persistent_loss_and_full_update_resume(guided,tmp_path):
    t=Trainer(guided);t.batch();t.ppo_actions=10000000
    row=t.batch();assert row['bc_weight']==.2 and row['bc_loss']>0
    assert row['actor_change_l2']>0 and row['critic_change_l2']>0 and row['initial_ratio_max_error']<1e-4
    path=tmp_path/'resume.pt';t.save(path)
    expected=t.batch();weights={k:v.clone() for k,v in t.model.state_dict().items()};t.close()
    b=Trainer(guided);b.load(path);actual=b.batch()
    assert actual['games']==expected['games'] and actual['bc_loss']==expected['bc_loss']
    assert all(torch.equal(v,weights[k]) for k,v in b.model.state_dict().items());b.close()


def test_parent_actor_critic_adam_preserved_not_reset(config,guided,tmp_path):
    a=ParentTrainer(config);a.batch();a.batch();p=tmp_path/'parent.pt';a.save(p)
    weights={k:v.clone() for k,v in a.model.state_dict().items()};a.close()
    t=Trainer(guided);t.import_sampling_continuation(checkpoint_spec(p))
    assert all(torch.equal(v,weights[k]) for k,v in t.model.state_dict().items())
    assert t.optimizer.state and t.critic_optimizer.state and t.warm_batches==1
    assert t.lineage['rng']=='new paired seed'
    t.save(tmp_path/'first_round.pt');old_rng=t.rng.bit_generator.state
    other=Trainer(guided);other.import_sampling_continuation(checkpoint_spec(tmp_path/'first_round.pt'))
    assert other.rng.bit_generator.state==old_rng and other.lineage['rng']=='preserved'
    bad=Trainer({**guided,'reward_scale':150.,'learning_rate':.01})
    with pytest.raises(ValueError,match='protected'):bad.import_sampling_continuation(checkpoint_spec(p))
    t.close();other.close();bad.close()


def test_correction_split_and_hash_gates(guided,tmp_path):
    with pytest.raises(ValueError,match='hash'):
        Trainer({**guided,'correction_datasets':[{'path':guided['dataset'],'sha256':'bad'}]})
    with pytest.raises(ValueError,match='without admitted'):
        Trainer({**guided,'correction_datasets':[]})
    with np.load(guided['dataset']) as z:data={k:z[k].copy() for k in z.files}
    data['seed'][0]=read_json(guided['split_manifest'])['sets']['development'][0]
    p=tmp_path/'leak.npz';np.savez(p,**data)
    with pytest.raises(ValueError,match='leakage'):
        Trainer({**guided,'correction_datasets':[{'path':str(p),'sha256':digest(p)}]})
    with np.load(guided['dataset']) as z:data={k:z[k].copy() for k in z.files}
    data['mask'][:]=False;data['mask'][np.arange(len(data['y'])),data['y']]=True
    forced=tmp_path/'forced.npz';np.savez(forced,**data)
    with pytest.raises(ValueError,match='only forced actions'):
        Trainer({**guided,'correction_datasets':[{'path':str(forced),'sha256':digest(forced)}]})


def test_fp64_export_and_complete_public_rollin(config,tmp_path):
    t=ParentTrainer(config);p=tmp_path/'parent.pt';t.save(p)
    export=tmp_path/'rollin.npz';meta=export_checkpoint(p,export)
    assert export_checkpoint(p,export)==meta
    student=Student64(export,meta['sha256'])
    e=Harmonies();obs,_=e.reset(221)
    while not obs['terminated']:
        encoded=encode(obs,allow_stop=False)
        with torch.no_grad():
            logits,_=t.model(torch.as_tensor(np.asarray(encoded['features'],np.float32),dtype=torch.float64)[None],
                             torch.tensor(encoded['action_mask'])[None])
        mask=np.asarray(encoded['action_mask'])
        np.testing.assert_allclose(student.logits(obs)[mask],logits[0].numpy()[mask],atol=1e-12,rtol=1e-12)
        action=student.decide(obs)
        assert index(obs,action,'slots240_v1',allow_stop=False)==int(logits.argmax(-1))
        obs=e.step(action).observation
    t.close()


def test_teacher_labels_never_control_rollin_and_replay_data(config,tmp_path):
    t=ParentTrainer(config);p=tmp_path/'parent.pt';t.save(p);t.close()
    export=tmp_path/'rollin.npz';meta=export_checkpoint(p,export)
    manifest=read_json(config['split_manifest']);seeds=manifest['sets']['train'][:2]
    root=tmp_path/'campaign';(root/'resolved').mkdir(parents=True)
    jobs=[]
    for i,seed in enumerate(seeds):
        job={'id':f'e{i}','arm':'correct_low_a','environment_seed':seed,'search_seed':130,
             'method':'terminal','multiplier':1,'params':{'mode':'timed','continuation':'dagger',
                 'seconds':.03,'checkpoint':str(export),'checkpoint_sha256':meta['sha256'],'stochastic':False}}
        jobs.append(job);result=run_episode(job,root/'episodes'/job['id'])
        assert result['status']=='completed' and result['natural_terminal']
        write_json(root/'resolved'/f'{job["id"]}.json',result)
        s=Student64(export,meta['sha256'])
        for line in (root/'episodes'/job['id']/'decisions.jsonl').read_text().splitlines():
            row=json.loads(line)
            assert row['action']==s.decide(row['observation_before'])
            assert row['search']['labels_only'] and row['search']['expert_action'] in row['observation_before']['legal_actions']
            assert any(c['action']==row['action'] for c in row['search']['candidates'])
    write_json(root/'plan.json',{'split':'train_calibration'});write_json(root/'splits.json',manifest)
    write_json(root/'jobs.json',jobs);write_json(root/'status.json',{'phase':'completed','active_count':0})
    write_json(root/'allocation.json',{'source_id':'test'})
    info=build_data([root],tmp_path/'admitted.npz',[seeds[-1]],minimum_teacher_rounds=2)
    assert info['games']==2 and info['discarded_unsampled_labels']>0
    # Tampering with recorded state must be rejected, not silently trained on.
    log=root/'episodes/e0/decisions.jsonl';rows=log.read_text().splitlines();bad=json.loads(rows[0])
    bad['observation_before']['turn_index']+=1;rows[0]=json.dumps(bad);log.write_text('\n'.join(rows)+'\n')
    with pytest.raises(ValueError,match='state mismatch'):build_data([root],tmp_path/'bad.npz',[seeds[-1]])


def test_public_train_only_plan(config,tmp_path):
    t=ParentTrainer(config);p=tmp_path/'parent.pt';t.save(p);t.close()
    ex=tmp_path/'rollin.npz';m=export_checkpoint(p,ex)
    raw={'split':'train_calibration','panels':[2],'environment_indices':[0,1],'search_seeds':[1],
         'arms':[{'id':'a','method':'terminal','multiplier':1,'timeout_seconds':901,
            'params':{'mode':'timed','continuation':'dagger','seconds':900,'checkpoint':str(ex),
                      'checkpoint_sha256':m['sha256'],'stochastic':False}}]}
    manifest=read_json(config['split_manifest']);plan=prepare_plan(raw,manifest)
    assert plan['environment_seeds']==manifest['sets']['train'][:2]
    with pytest.raises(ValueError,match='training split'):prepare_plan({**raw,'split':'development'},manifest)
    bad=deepcopy(raw);bad['arms'][0]['params']['seconds']=901
    with pytest.raises(ValueError,match='invalid'):prepare_plan(bad,manifest)


def test_job_pairing_and_evaluation_separation():
    repo=Path(__file__).resolve().parents[1];c=read_json(repo/'configs/v130_teacher_guided_ppo.json')
    if not (repo/c['parent']).exists():pytest.skip('research parent unavailable')
    correction=[{'path':str(repo/c['dataset']),'sha256':c['dataset_sha256']}]
    jobs=dict(build_jobs(c,repo,1,None,correction));assert len(jobs)==8
    assert len({v['continuation']['sha256'] for v in jobs.values()})==1
    for suffix in 'ab':
        variants=[v for k,v in jobs.items() if k.endswith(suffix)]
        for key in ('seed','excluded_train_seeds','curve_seeds','final_seeds','target_actions'):
            assert all(v[key]==variants[0][key] for v in variants)
    assert jobs['control_a']['bc_weight']==0 and not jobs['static_a']['correction_datasets']
    assert sum(len(label_indices(r))*(2 if r==1 else 4) for r in range(1,4))==192
    assert len({i for r in range(1,4) for i in validation_indices(r)})==8
    with pytest.raises(ValueError,match='seeds differ'):
        paired({'games':[{'environment_seed':1,'score':2}]},{'games':[{'environment_seed':2,'score':2}]})


def test_guided_full_run_artifacts(guided,tmp_path):
    s=run(guided,tmp_path/'run',device='cpu')
    assert s['status']=='completed' and s['updates']>0
    m=read_json(tmp_path/'run/evaluations/final.json')
    assert 'correction_fit' in m['teacher_fit_diagnostic'] and m['policy']=='greedy_model_only'
    assert run(guided,tmp_path/'run',device='cpu')==s


@pytest.mark.skipif(os.environ.get('BOARDBENCH_V130_CUDA')!='1',reason='explicit reserved V130 CUDA preflight only')
def test_cuda_guidance_resume(guided,tmp_path):
    assert torch.cuda.is_available()
    c={**guided,'hidden':256,'batch_episodes':8,'workers':2,'minibatch':256}
    t=Trainer(c,'cuda');t.batch();r=t.batch();assert r['bc_loss']>0 and r['bc_weight']==.2
    p=tmp_path/'cuda.pt';t.save(p);expected=t.batch();weights={k:v.clone() for k,v in t.model.state_dict().items()};t.close()
    b=Trainer(c,'cuda');b.load(p);actual=b.batch()
    assert expected['games']==actual['games'] and expected['bc_loss']==actual['bc_loss']
    assert all(torch.equal(v,weights[k]) for k,v in b.model.state_dict().items());b.close()
