import shutil
from pathlib import Path

import pytest
np=pytest.importorskip('numpy')
torch=pytest.importorskip('torch')

from boardbench.artifacts.store import digest,read_json,write_json
from boardbench.v128.collector import EnvironmentPool as FastPool
from boardbench.v128.environment import FastHarmonies
from boardbench.v129.collector import EnvironmentPool as NativePool,_SamplingHarmonies
from boardbench.v129.encoding import build,NativeEncoder
from boardbench.v127.training import Trainer
from boardbench.v127.launch import attach_continuations,build_jobs


@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    if not shutil.which('g++'):pytest.skip('native backend requires explicit C++ build')
    return build(tmp_path_factory.mktemp('v129')/'build')


@pytest.mark.parametrize('workers',[0,2])
def test_native_pool_exact_arrays_and_repeated_reset(artifact,workers):
    rng=np.random.default_rng(129)
    with FastPool(4,workers) as a,NativePool(4,workers,native_artifact=artifact) as b:
        for seeds in ([1,12,55,300],[13,18,219,528]):
            x=a.reset(seeds);y=b.reset(seeds)
            while True:
                for key in ('features','masks','done','stats'):np.testing.assert_array_equal(x[key],y[key])
                if x['done'].all():break
                slots=np.flatnonzero(~x['done'])
                actions=[int(rng.choice(np.flatnonzero(x['masks'][s,:240]))) for s in slots]
                x=a.step(slots,actions);y=b.step(slots,actions)
        # Explicit STOP still yields identical public statistics and zero masks.
        x=a.reset([1,2,3,4]);y=b.reset([1,2,3,4])
        x=a.step(np.arange(4),[240]*4);y=b.step(np.arange(4),[240]*4)
        for key in ('features','masks','done','stats'):np.testing.assert_array_equal(x[key],y[key])


def test_internal_reset_metadata_only_and_native_provenance(artifact):
    a,b=FastHarmonies(),_SamplingHarmonies()
    oa,ia=a.reset(45);ob,ib=b.reset(45)
    assert oa==ob and a.save_state()==b.save_state() and ia and ib=={}
    for field in ('sha256','cpp_sha256','layout'):
        with pytest.raises(ValueError,match='mismatch'):
            NativeEncoder.from_artifact({**artifact,field:'wrong'})


def config():
    repo=Path(__file__).resolve().parents[1];raw=read_json(repo/'configs/v128_a218_packed_ppo.json')
    if not (repo/raw['dataset']).exists():pytest.skip('local research parents unavailable')
    return {**build_jobs(raw,repo)[0][1],'batch_episodes':4,'workers':0,'epochs':1,
            'critic_batches':1,'critic_epochs':2,'minibatch':128}


def test_native_ppo_next_update_and_resume_exact(artifact,tmp_path):
    c=config();a=Trainer(c);a.batch();a.batch();p=tmp_path/'parent.pt';a.save(p)
    spec={'checkpoint':str(p),'sha256':digest(p),'source_id':a.source}
    expected=a.batch();weights={k:v.clone() for k,v in a.model.state_dict().items()};a.close()
    child={**c,'environment_backend':'native_v129','native_artifact':artifact,'continuation':spec}
    b=Trainer(child);b.import_sampling_continuation(spec);actual=b.batch()
    assert actual['games']==expected['games'] and actual['ppo_actions']==expected['ppo_actions']
    for k,v in b.model.state_dict().items():assert torch.equal(v,weights[k])
    path=tmp_path/'native.pt';b.save(path);expected=b.batch()
    weights={k:v.clone() for k,v in b.model.state_dict().items()};b.close()
    b=Trainer(child);b.load(path);actual=b.batch()
    assert expected['games']==actual['games']
    for k,v in b.model.state_dict().items():assert torch.equal(v,weights[k])
    assert actual['initial_ratio_max_error']<1e-4;b.close()


def test_interrupted_parent_requires_explicit_optin_and_preserves_budget(tmp_path):
    root=tmp_path/'old';p=root/'trials/a';p.mkdir(parents=True)
    (p/'final.pt').write_bytes(b'parent fixture')
    write_json(p/'final.json',{'sha256':digest(p/'final.pt'),'source_id':'parent'})
    write_json(p/'summary.json',{'status':'interrupted','source_id':'parent','wall_seconds':600.})
    write_json(root/'state.json',{'phase':'interrupted'})
    raw={'continuation_root':'old','carry_stage_budget':True}
    with pytest.raises(ValueError,match='quiescent'):
        attach_continuations(raw,tmp_path,[('a',{'max_seconds':14400})])
    jobs=[('a',{'max_seconds':14400})]
    attach_continuations({**raw,'allow_interrupted_parent':True},tmp_path,jobs)
    assert jobs[0][1]['max_seconds']==13800
    assert jobs[0][1]['continuation']['sha256']==digest(p/'final.pt')
    with pytest.raises(ValueError,match='exhausted'):
        attach_continuations({**raw,'allow_interrupted_parent':True},tmp_path,[('a',{'max_seconds':600})])
    write_json(root/'state.json',{'phase':'running'})
    with pytest.raises(ValueError,match='quiescent'):
        attach_continuations({**raw,'allow_interrupted_parent':True},tmp_path,jobs)
