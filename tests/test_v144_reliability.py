from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time
import pytest
import torch

from boardbench.artifacts.store import read_json,write_json
from boardbench.v121 import collector
from boardbench.v121.collector import WorkerTimeoutError
from boardbench.v140.training import Trainer as OldEvaluator
from boardbench.v144.training import Trainer,run
from boardbench.v144 import campaign
from test_v129 import artifact
from test_v140 import duo_config
from test_v142 import config
from test_v143 import selfplay_config
from test_v144 import score_config


@pytest.mark.parametrize('workers',[0,2])
def test_persistent_pool_matches_reference_and_next_update(score_config,tmp_path,workers):
    c={**score_config,'workers':workers};t=Trainer(c);t.batch();pool=t.pool
    path=tmp_path/'before.pt';t.save(path);rng=t.rng.get_state().clone();seed_rng=deepcopy(t.seed_rng.bit_generator.state)
    old=OldEvaluator.evaluate(t,c['development_seeds'])
    pids=[p.pid for p in pool.processes]
    for _ in range(4):
        new=t.evaluate_panel(c['development_seeds'])
        assert new['per_game']==[dict(g,evaluation_opponent='transfer') for g in old['per_game']]
        assert t.pool is pool and [p.pid for p in pool.processes]==pids
        assert torch.equal(rng,t.rng.get_state()) and seed_rng==t.seed_rng.bit_generator.state
    expected=t.batch();weights=deepcopy(t.model.state_dict());t.close()
    restored=Trainer(c);restored.load(path);actual=restored.batch()
    assert actual['per_game']==expected['per_game']
    assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in weights.items());restored.close()


def test_evaluation_worker_timeout_rebuilds_once(score_config,monkeypatch):
    t=Trainer(score_config);t.batch();expected=t.evaluate_panel(score_config['development_seeds']);old=t.pool
    rng=t.rng.get_state().clone()
    def fail(_):raise WorkerTimeoutError(dict(expected='ok',command='reset',pid=999999999))
    monkeypatch.setattr(old,'reset',fail)
    actual=t.evaluate_panel(score_config['development_seeds'])
    assert expected==actual and old.closed and t.pool is not old
    assert torch.equal(rng,t.rng.get_state()) and len(t.pool_recovery_events)==1;t.close()


@pytest.mark.parametrize('failure_point',['evaluation','after_curve'])
def test_completed_update_survives_evaluation_and_append_crash(score_config,tmp_path,monkeypatch,failure_point):
    c={**score_config,'max_batches':2,'evaluate_interval':1,'finish_reserve_seconds':0}
    original_eval=Trainer.evaluate_panel;original_save=Trainer.save;failed=[False]
    def evaluate(self,seeds):
        if failure_point=='evaluation' and self.batches==1 and not failed[0]:
            failed[0]=True;raise TimeoutError('injected post-update evaluation failure')
        return original_eval(self,seeds)
    def save(self,path):
        if (failure_point=='after_curve' and self.batches==1 and self.pending_curve is None
                and Path(path).name=='latest.pt' and not failed[0]):
            failed[0]=True;raise OSError('injected crash after curve append')
        return original_save(self,path)
    monkeypatch.setattr(Trainer,'evaluate_panel',evaluate);monkeypatch.setattr(Trainer,'save',save)
    out=tmp_path/'interrupted'
    with pytest.raises((TimeoutError,OSError)):run(c,out,'cpu')
    checkpoint=torch.load(out/'latest.pt',weights_only=False)
    assert checkpoint['batches']==1 and checkpoint['extra_state']['pending_curve']['batches']==1
    assert (out/'best_policy.pt').exists()
    run(c,out,'cpu',resume=True)
    control=tmp_path/'uninterrupted';run(c,control,'cpu')
    a=torch.load(out/'final.pt',weights_only=False);b=torch.load(control/'final.pt',weights_only=False)
    assert a['batches']==b['batches']==2
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    assert torch.equal(a['rng'],b['rng']) and a['seed_rng']==b['seed_rng']
    assert [json.loads(line)['batches'] for line in (out/'curve.jsonl').read_text().splitlines()]==[1,2]
    assert a['extra_state']['pending_curve'] is None


def test_worker_startup_diagnostic_and_group_cleanup(monkeypatch):
    pool=collector.EnvironmentPool.__new__(collector.EnvironmentPool)
    pool.connections=[SimpleNamespace(poll=lambda _:False)];pool.processes=[SimpleNamespace(pid=999999999,exitcode=None)]
    pool.timeout=.001;pool.last_command='startup'
    with pytest.raises(WorkerTimeoutError) as failure:pool._receive('ready')
    assert failure.value.diagnostic['pid']==999999999 and failure.value.diagnostic['expected']=='ready'
    now=[0.];calls=[]
    class Process:
        alive=True
        def join(self,timeout):
            if self.alive:now[0]+=timeout
        def is_alive(self):return self.alive
        def terminate(self):calls.append('terminate')
        def kill(self):calls.append('kill');self.alive=False
    pool.processes=[Process() for _ in range(13)];pool.connections=[];pool.closed=False
    monkeypatch.setattr(collector,'time',SimpleNamespace(monotonic=lambda:now[0]))
    pool.close();assert now[0]<=4.01 and calls.count('terminate')==13 and calls.count('kill')==13


def runner_fixture(tmp_path,monkeypatch,durations):
    now=[1000.];root=tmp_path/'controller'
    for name in ('logs','rounds','decisions'):(root/name).mkdir(parents=True,exist_ok=True)
    write_json(root/'resolved_config.json',dict(started_unix=1000,deadline_unix=1500,gpu_uuids=['g0','g1'],retry_min_remaining_seconds=180))
    write_json(root/'launch.json',dict(cpus=list(range(28))))
    monkeypatch.setattr(campaign,'time',SimpleNamespace(time=lambda:now[0],time_ns=lambda:int(now[0]*1e9),sleep=lambda sec:now.__setitem__(0,now[0]+sec)))
    runner=campaign.Runner(tmp_path,root);starts=[]
    class Process:
        def __init__(self,end,rc,complete,pid):self.end,self.rc,self.complete,self.pid=end,rc,complete,pid;self.returncode=None
        def poll(self):
            if now[0]>=self.end:
                self.returncode=self.rc
                if self.rc==0:write_json(self.complete,dict(completed=now[0]))
            return self.returncode
        def wait(self,timeout):assert self.poll() is not None
        def send_signal(self,_):self.rc=-15;self.end=now[0]
    def start(spec,slot):
        n=sum(s['name']==spec['name'] for s in starts);duration,rc=durations[spec['name']][min(n,len(durations[spec['name']])-1)]
        starts.append(dict(name=spec['name'],slot=slot,time=now[0]))
        p=Process(now[0]+duration,rc,Path(spec['complete']),len(starts))
        receipt=dict(stage=spec['name'],pid=p.pid,slot=slot,gpu=f'g{slot}',cpus=list(range(slot*14,(slot+1)*14)),started_unix=now[0])
        runner.receipts.append(receipt);runner.active.append((p,SimpleNamespace(close=lambda:None),receipt,spec))
    monkeypatch.setattr(runner,'start',start)
    return runner,starts,now


def test_failed_slot_retries_while_healthy_peer_runs(tmp_path,monkeypatch):
    runner,starts,now=runner_fixture(tmp_path,monkeypatch,dict(healthy=[(30,0)],bad=[(1,1),(2,0)]))
    specs=[dict(name=n,out=str(tmp_path/n),complete=str(tmp_path/n/'done.json')) for n in ('healthy','bad')]
    runner.run(specs,1400,'injected_wave')
    assert [(s['name'],s['slot']) for s in starts]==[('healthy',0),('bad',1),('bad',1)]
    assert starts[2]['time']<1030 and now[0]==1030
    retry=read_json(next((runner.root/'decisions').glob('retry_*.json')))
    assert retry['slot']==1 and retry['gpu']=='g1' and retry['until']==1400
    assert not runner.active


def test_failed_slot_retry_bounded_and_no_deadline_renewal(tmp_path,monkeypatch):
    runner,starts,now=runner_fixture(tmp_path,monkeypatch,dict(bad=[(1,1)]))
    specs=[dict(name='bad',out=str(tmp_path/'bad'),complete=str(tmp_path/'bad/done.json'))]
    with pytest.raises(RuntimeError):runner.run(specs,1400,'injected_wave')
    assert len(starts)==2 and runner.deadline==1500 and not runner.active


def test_final_panel_uses_evaluation_deadline(tmp_path,monkeypatch):
    from boardbench.v144 import evaluate
    from boardbench.artifacts.store import digest
    config_path=tmp_path/'config.json';write_json(config_path,dict(deadline_unix=1))
    policy=tmp_path/'policy.json';write_json(policy,{})
    seen=[]
    class FakeTrainer:
        model=SimpleNamespace(load_state_dict=lambda _:None)
        def __init__(self,c,device):seen.append(c)
        def evaluate_panel(self,seeds):return dict(per_game=[])
        def close(self):pass
    monkeypatch.setattr(evaluate,'Trainer',FakeTrainer)
    monkeypatch.setattr(evaluate,'load_policy',lambda *args:SimpleNamespace(state_dict=lambda:{}))
    evaluate.run(dict(config=str(config_path),model=str(policy),model_sha256=digest(policy),
        stage='development',seeds=[244100000],deadline_unix=999),tmp_path/'evaluation','cpu')
    assert seen==[dict(deadline_unix=1,evaluation_deadline_unix=999)]
    assert read_json(config_path)==dict(deadline_unix=1)


def test_matched_prefix_excludes_early_termination():
    from boardbench.v144.gap_audit import matched_prefix
    def game(seed,setting,reached,score):
        return dict(seed=seed,setting=setting,players=[dict(prefix_reached=reached,
            prefix11=dict(score=score,animal=1,visible_unique_cards=2,cards_taken=3))])
    result=matched_prefix([game(1,'curriculum',True,20),game(1,'official_selfplay',True,10),
        game(2,'curriculum',True,30),game(2,'official_selfplay',False,40)])
    assert result['environments']==1 and result['settings']['official_selfplay']['score']==10
