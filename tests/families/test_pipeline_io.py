from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import torch

from boardbench.families.neural import CandidateActorCritic, save_model
from boardbench.families.opponent_pool import OpponentPool
from boardbench.families.pool_evaluator import bounded_jobs
from boardbench.families.pool_training import rollout, compact_rollout, audit_behavior, save_checkpoint, restore_checkpoint
from boardbench.families.ppo import update
from boardbench.families.registry import game_class
from boardbench.families.trajectory_io import (encode_json, compact_episode, write_compressed,
                                               write_episode_batch, read_episode_batch)


def test_lossless_legacy_and_multi_member_archives(tmp_path):
    row=dict(x=np.array([[1.25,-.5]],np.float32),prior=np.zeros(1,np.float32),
             action=0,old_logp=0.,value=.25,advantage=.75,return_=1.,legal_mask=[True],
             player_id=0,policy_version='v',opponent_id='o',legal_actions=[{'take':1}],
             observation={'public':['é',4]},terminated=True,truncated=False,reward=1.)
    row['return']=row.pop('return_')
    episode=dict(result={'status':'complete'},records=[row],excluded_records=[])
    original=dict(policy_version='v',games=[episode,episode])
    old=tmp_path/'old.json.gz';write_compressed(old,original)
    compact=[compact_episode(episode),compact_episode(episode)]
    new=tmp_path/'new.jsonl.gz';write_episode_batch(new,{'policy_version':'v'},compact)
    assert read_episode_batch(old)==read_episode_batch(new)
    assert read_episode_batch(new)['games'][0]['records'][0]['observation']==row['observation']
    assert 'observation' not in compact[0]['records'][0]
    assert compact[0]['records'][0]['legal_action_count']==1
    assert gzip.decompress(old.read_bytes())==encode_json(original)
    with pytest.raises(ValueError):write_compressed(tmp_path/'bad.json.gz',{'x':float('nan')})
    assert not (tmp_path/'bad.json.gz').exists()


def test_rolling_dispatch_avoids_wave_barrier_and_preserves_order():
    reached=threading.Event()
    def job(spec):
        if spec['i']==0:
            assert reached.wait(3), 'fast worker did not receive next job while slow worker was active'
        if spec['i']==2:reached.set()
        return spec['i']
    with ThreadPoolExecutor(max_workers=2) as pool:
        values=bounded_jobs(pool,job,[{'i':i} for i in range(6)],time.time()+10,2)
    assert values==list(range(6))


def test_pause_drains_inflight_jobs_without_starting_new_ones():
    paused=threading.Event()
    def job(spec):paused.set();return spec['i']
    with ThreadPoolExecutor(max_workers=2) as pool:
        values=bounded_jobs(pool,job,[{'i':i} for i in range(20)],time.time()+10,2,paused.is_set)
    assert 1<=len(values)<=2 and values==list(range(len(values)))


def test_compact_rollout_has_identical_training_and_archived_data(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(23)
    cls=game_class('toy_battle');model=CandidateActorCritic()
    path=tmp_path/'initial.pt';save_model(path,model,cls,cls.default_weights)
    job=dict(game='toy_battle',path=str(path),self_play=False,learner_seat=0,seed=903,
             opponent={'kind':'neural','path':str(path),'stochastic':True},opponent_id='initial',
             policy_version='initial',max_seconds=30,gae_lambda=.95)
    original=rollout(job);compact=compact_rollout(job)
    assert original['result']['status']==compact['result']['status']=='complete'
    assert original['result']['outcome']==compact['result']['outcome']
    archived=json.loads(gzip.decompress(compact['archive_member']))
    assert archived['records']==json.loads(encode_json(original['records']))
    assert audit_behavior(model,compact['records'],'cpu')<1e-4
    a,b=deepcopy(model),deepcopy(model)
    ao,bo=torch.optim.Adam(a.parameters(),lr=1e-4),torch.optim.Adam(b.parameters(),lr=1e-4)
    update(a,ao,original['records'],'cpu',np.random.default_rng(4),epochs=2,batch_size=8)
    update(b,bo,compact['records'],'cpu',np.random.default_rng(4),epochs=2,batch_size=8)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


def test_static_checkpoint_data_once_and_checksum_verified(tmp_path):
    torch.manual_seed(3);model=CandidateActorCritic();optimizer=torch.optim.Adam(model.parameters())
    rng=np.random.default_rng(3);pool=OpponentPool({'id':'initial','spec':{'kind':'neural'}})
    state=dict(next_update=0,baseline_rows=[{'id':2}],fingerprint_states=[{'observation':{'large':'x'*10000}}])
    path=tmp_path/'checkpoint.pt'
    save_checkpoint(path,model,optimizer,rng,pool,state,'hash',{})
    payload=torch.load(path,weights_only=True)
    assert 'fingerprint_states' not in payload['state']
    _, restored=restore_checkpoint(path,model,optimizer,rng,'hash')
    assert restored['fingerprint_states']==state['fingerprint_states']
    static=Path(state['static_data_ref']['path']);before=static.stat().st_mtime_ns
    save_checkpoint(path,model,optimizer,rng,pool,state,'hash',{})
    assert static.stat().st_mtime_ns==before
    static.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):restore_checkpoint(path,model,optimizer,rng,'hash')


def test_paused_resume_preserves_model_optimizer_rng_and_remaining_budget(tmp_path):
    import hashlib
    from boardbench.families.pool_resume import prepare_resume
    source=tmp_path/'old';source.mkdir();destination=tmp_path/'resumed'
    torch.manual_seed(9);model=CandidateActorCritic();opt=torch.optim.Adam(model.parameters())
    cls=game_class('toy_battle');initial=source/'initial.pt';save_model(initial,model,cls,cls.default_weights)
    sha=hashlib.sha256(initial.read_bytes()).hexdigest()
    cfg=dict(game='toy_battle',arm='B',output=str(source),source_commit='old',hours=1.,
             initial_path=str(initial),initial_sha256=sha)
    (source/'config.json').write_text(json.dumps(cfg))
    pool=OpponentPool({'id':'initial','spec':{'kind':'neural','path':str(initial),'sha256':sha}})
    state=dict(phase='failed',error='Interrupted worker',deadline=time.time()+50,began_unix=time.time()-100,
               next_update=3,learner_decisions=1234)
    rng=np.random.default_rng(9)
    save_checkpoint(source/'checkpoint.pt',model,opt,rng,pool,state,
                    hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),cfg)
    original=(source/'checkpoint.pt').read_bytes()
    record=tmp_path/'pause.json';record.write_text(json.dumps({'jobs':[dict(game='toy_battle',arm='B',
        output=str(source),exit_code='1',finite_weights=True,remaining_seconds=2400.)]}))
    resumed=prepare_resume(record,'toy_battle','B',destination,'new')
    payload=torch.load(destination/'checkpoint.pt',weights_only=True)
    assert all(torch.equal(v,payload['model'][k]) for k,v in model.state_dict().items())
    assert payload['state']['phase']=='prepared_resume'
    assert payload['state']['learner_decisions']==1234 and payload['state']['elapsed_before_resume']==1200
    assert resumed['hours']*3600==pytest.approx(2400)
    assert 2390<resumed['deadline_unix']-time.time()<=2400
    assert (source/'checkpoint.pt').read_bytes()==original
    assert payload['numpy_rng']==rng.bit_generator.state
    with pytest.raises(ValueError):prepare_resume(record,'toy_battle','B',destination,'new')
