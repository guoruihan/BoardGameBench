from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from boardbench.artifacts.store import digest, read_json, write_json, source_id
from boardbench.v121.registry import make_splits
from boardbench.v126 import campaign
from boardbench.v126.pipeline import validate_config, validate_host, verify_external_teacher, arm, Pipeline


def config():
    return read_json(Path(__file__).resolve().parents[1]/'configs/v126_a218_search_learning.json')


def test_a218_design_has_eight_balanced_students_and_unchanged_cohorts():
    c=config();validate_config(c)
    old=read_json(Path(__file__).resolve().parents[1]/'configs/v126_search_learning.json')
    for k in ('train_indices','validation_indices','student_selection_indices','pilot_indices','dagger_indices','fresh_indices'):
        assert c[k]==old[k]
    assert {(h,q,s) for _,h,q,s in c['model_specs']} == {
        (h,q,s) for h in (128,256) for q in (0.,.5) for s in (12601,12602)}
    assert len(c['gpu_uuids'])==8 and c['require_cuda']
    assert c['campaign_cores']+1<=128
    with pytest.raises(ValueError,match='duplicate GPU'):
        validate_config({**c,'gpu_uuids':c['gpu_uuids'][:7]+[c['gpu_uuids'][0]]})
    with pytest.raises(ValueError,match='one authorized GPU'):
        validate_config({**c,'gpu_uuids':c['gpu_uuids'][:7]})


def test_host_authorization_and_campaign_reserve_outer(monkeypatch):
    c=config();validate_host(c,'a218.cluster.helixon.com')
    with pytest.raises(ValueError,match='resource policy'):
        validate_host(c,'a219.cluster.helixon.com')
    with pytest.raises(ValueError,match='resource policy'):
        validate_host({**c,'resource_profile':'b35_reserve16_v1'},'b35.cluster.helixon.com')
    validate_host({**c,'resource_profile':'b35_reserve16_v1','gpu_uuids':[]},'b35.cluster.helixon.com')
    monkeypatch.setattr(campaign.os,'uname',lambda:SimpleNamespace(nodename='a218.cluster.helixon.com'))
    assert campaign.cpu_core_limit(list(range(128)),c)==127
    assert campaign.cpu_core_limit(list(range(256)),c)==127
    monkeypatch.setattr(campaign.os,'uname',lambda:SimpleNamespace(nodename='b35.cluster.helixon.com'))
    with pytest.raises(ValueError,match='a218 profile'):
        campaign.cpu_core_limit(list(range(128)),c)


@pytest.fixture
def external(tmp_path):
    root=tmp_path/'teacher';manifest=make_splits(train_count=8)
    raw={'split':'train_calibration','environment_indices':[0,1,2], 'panels':[3],
         'search_seeds':[12620261020],'arms':[arm('teacher')]}
    plan=campaign.prepare_plan(raw,manifest);jobs=campaign.make_jobs(plan)
    write_json(root/'plan.json',plan);write_json(root/'splits.json',manifest);write_json(root/'jobs.json',jobs)
    write_json(root/'status.json',{'phase':'completed','active_count':0})
    write_json(root/'watchdog_result.json',{'allocated_cpu_core_seconds':100})
    source=root/'source';source.mkdir();(source/'pyproject.toml').write_text('[project]\nname="fixture"\n')
    allocation={'source_id':source_id(source),'plan_sha256':digest(root/'plan.json'),
                'splits_sha256':digest(root/'splits.json')}
    write_json(root/'allocation.json',allocation)
    for j in jobs: write_json(root/'resolved'/f'{j["id"]}.json',{'status':'completed','natural_terminal':True})
    c={**config(),'train_indices':[0,1],'validation_indices':[2],
       'external_teacher_source_id':allocation['source_id'],
       'external_teacher_plan_sha256':allocation['plan_sha256']}
    return root,c


def test_external_teacher_reuses_exact_completed_provenance(external):
    root,c=external
    assert verify_external_teacher(root,c,root/'splits.json')==root
    with pytest.raises(ValueError,match='cohort'):
        verify_external_teacher(root,{**c,'train_indices':[0,3]},root/'splits.json')
    with pytest.raises(ValueError,match='hash mismatch'):
        verify_external_teacher(root,{**c,'external_teacher_source_id':'changed'},root/'splits.json')
    write_json(root/'status.json',{'phase':'running','active_count':1})
    with pytest.raises(ValueError,match='finalized'):
        verify_external_teacher(root,c,root/'splits.json')


def test_external_teacher_rejects_modified_job_or_censored_result(external):
    root,c=external;jobs=read_json(root/'jobs.json');bad=deepcopy(jobs)
    bad[0]['search_seed']+=1;write_json(root/'jobs.json',bad)
    with pytest.raises(ValueError,match='cohort'):
        verify_external_teacher(root,c,root/'splits.json')
    write_json(root/'jobs.json',jobs)
    write_json(root/'resolved'/f'{jobs[0]["id"]}.json',{'status':'completed','natural_terminal':False})
    with pytest.raises(ValueError,match='censored'):
        verify_external_teacher(root,c,root/'splits.json')


def test_required_cuda_does_not_silently_fall_back(tmp_path,monkeypatch):
    p=Pipeline.__new__(Pipeline);p.root=tmp_path/'run';p.repo=tmp_path
    p.config={'require_cuda':True};p.state={'stages':{}};p.children=[]
    (tmp_path/'outputs/v122').mkdir(parents=True)
    monkeypatch.setattr(p,'wait_ready',lambda:None)
    monkeypatch.setattr(p,'probe_gpus',lambda:[])
    monkeypatch.setattr(p,'publish',lambda phase:None)
    with pytest.raises(RuntimeError,match='no CPU fallback'):
        p.training('initial',tmp_path/'unused.npz',[('bc',32,0.,11)])
    assert not p.children and 'active_training' not in p.state
    assert len(list((p.root/'receipts').glob('*.json')))==1
