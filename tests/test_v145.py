from copy import deepcopy
import json
import time
import numpy as np
import pytest
import torch
from boardbench.artifacts.store import digest,read_json
from boardbench.v144.training import Trainer as OldTrainer
from boardbench.v144.policy import export_policy as old_export
from boardbench.v145.training import Trainer,run
from boardbench.v145.policy import LAYOUT,export_policy,load_policy
from boardbench.v145.credit import plan_potential,credit,visible_cards
from boardbench.v145.replay import replay
from boardbench.v138.baseline import Planner
from boardbench.environments.harmonies_duo import HarmoniesDuo,player_view
from boardbench.v140.encoding import encode_batch
from test_v129 import artifact
from test_v140 import duo_config
from test_v142 import config
from test_v143 import selfplay_config
from test_v144 import score_config


@pytest.fixture
def animal_config(score_config,tmp_path):
    t=OldTrainer(score_config);parent=tmp_path/'v144.pt';old_export(t.model,parent,score_config);t.close()
    return {**score_config,'layout':LAYOUT,'mode':'fixed','progress_scale':0.,'sil_weight':0.,
        'warm_start':str(parent),'warm_start_sha256':digest(parent),'training_anchor':str(parent),
        'training_anchor_sha256':digest(parent),'replay_directory':str(tmp_path/'replay'),
        'replay_batches':2,'sil_steps':2,'sil_minibatch':64,
        'development_seeds':[245000001,245000002],'decision_seeds':[245010001,245010002],
        'confirmation_seeds':[345000001,345000002],'deadline_unix':time.time()+600}


def test_public_potential_and_return_conservation():
    obs,_=HarmoniesDuo().reset(1);vector=Planner().vector(player_view(obs));x=encode_batch([obs],[vector])
    assert plan_potential(x)[0]==0
    assert visible_cards(x[0])==set(obs['animal_market'])
    hidden=deepcopy(obs);hidden['private_order']=[99]
    np.testing.assert_array_equal(x,encode_batch([hidden],[vector]))
    raw=[.1,0.,.2,-.05];phis=[0.,3.,5.,2.]
    shaped,adv,targets,mc=credit(raw,[0.]*4,phis,1.,1.)
    assert abs(shaped.sum()-sum(raw))<1e-12
    np.testing.assert_allclose(targets,mc,atol=1e-12)
    with pytest.raises(ValueError):credit(raw,[0.]*4,[1.,2.,3.,0.],1.,.95)


@pytest.mark.parametrize('mode,scale,sil',[('current',0.,0.),('fixed',0.,0.),('fixed',1.,0.),('fixed',0.,.05),('mixed',1.,.05)])
def test_roles_credit_replay_and_exact_resume(animal_config,tmp_path,mode,scale,sil):
    c={**animal_config,'mode':mode,'progress_scale':scale,'sil_weight':sil}
    t=Trainer(c);anchor=deepcopy(t.training_anchor.state_dict());first=t.batch()
    assert first['learner_trajectories']=={'current':8,'fixed':4,'mixed':6}[mode]
    assert first['actor_change']>0 and first['initial_ratio_max_error']<1e-5
    for game in t.last_traces:list(replay(game))
    assert not sil or first['replay_rows']>0
    path=tmp_path/'resume.pt';t.save(path);expected=t.batch();weights=deepcopy(t.model.state_dict());traces=deepcopy(t.last_traces);t.close()
    restored=Trainer(c);restored.load(path);actual=restored.batch()
    assert actual['per_game']==expected['per_game'] and traces==restored.last_traces
    assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in weights.items())
    assert all(torch.equal(v,restored.training_anchor.state_dict()[k]) for k,v in anchor.items())
    state=deepcopy(restored.replay.state()) if sil else None;rng=restored.rng.get_state().clone();pool=restored.pool
    ev=restored.evaluate_panel(c['development_seeds'])
    assert restored.pool is pool and torch.equal(rng,restored.rng.get_state())
    assert not sil or restored.replay.state()==state
    assert ev['games']==4 and 'prefix10_animal_mean' in ev
    portable=tmp_path/'policy.pt';export_policy(restored.model,portable,c);loaded=load_policy(portable)
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in weights.items())
    for game in restored.last_traces:list(replay(game))
    restored.close()


def test_unchanged_control_evaluation(animal_config,score_config):
    old=OldTrainer(score_config);new=Trainer(animal_config)
    seeds=animal_config['development_seeds']
    a=old.evaluate(seeds);b=new.evaluate(seeds)
    assert [{k:g[k] for k in a['per_game'][0]} for g in b['per_game']]==a['per_game']
    old.close();new.close()


def test_versioned_lifecycle_and_pending_resume(animal_config,tmp_path,monkeypatch):
    c={**animal_config,'max_batches':1,'finish_reserve_seconds':0,'evaluate_interval':1,'sil_weight':.05,'progress_scale':1.}
    original=Trainer.evaluate_panel;failed=[False]
    def fail(self,seeds):
        if self.batches==1 and not failed[0]:failed[0]=True;raise TimeoutError('after PPO/SIL checkpoint')
        return original(self,seeds)
    monkeypatch.setattr(Trainer,'evaluate_panel',fail);out=tmp_path/'trial'
    with pytest.raises(TimeoutError):run(c,out,'cpu')
    s=torch.load(out/'latest.pt',weights_only=False)
    assert s['version']==145 and s['extra_state']['pending_curve'] and s['extra_state']['replay']['receipts']
    run(c,out,'cpu',resume=True)
    assert read_json(out/'done.json')['batches']==1
    assert len((out/'curve.jsonl').read_text().splitlines())==1
    assert not (out/'confirmation.json').exists()
    assert load_policy(out/'best_policy.pt') is not None


def test_guards(animal_config):
    for patch in ({'objective':'margin'},{'mode':'league'},{'progress_scale':3},{'sil_weight':-1},
        {'decision_seeds':[345000001]},{'training_anchor_sha256':'bad'}):
        with pytest.raises(ValueError):Trainer({**animal_config,**patch})


def test_adaptive_choices_and_namespace():
    from boardbench.v145.decisions import first_round,next_round
    arms=first_round();records=[]
    for i,arm in enumerate(arms):
        games=[dict(seed=s,seat=seat,evaluation_opponent='transfer',score=80+i,animal=10+i,margin=0,win_credit=.5)
            for s in (245010001,245010002) for seat in (0,1)]
        records.append(dict(config={**arm,'learning_rate':1e-5,'entropy':.005},policy=f'p{i}',sha256=f'h{i}',
            evaluation=dict(score_mean=80+i,animal_mean=10+i,per_game=games)))
    new,decision=next_round(records,1)
    assert new[0]['warm_start']=='p0' and new[1]['warm_start']=='p3'
    assert new[2]['progress_scale']==1 and new[2]['sil_weight']==.05 and new[3]['mode']=='mixed'
    assert not decision['uses_confirmation']
    new,_=next_round(records,2);assert new[2]['gae_lambda']==1. and new[3]['learning_rate']==5e-6
    bad=deepcopy(records);bad[0]['evaluation']['per_game'][0]['seed']=345000001
    with pytest.raises(ValueError):next_round(bad,1)


def test_real_round_and_confirmation(animal_config,tmp_path):
    from boardbench.v145.evaluate import run as evaluate
    c={**animal_config,'name':'fixed','max_batches':1,'finish_reserve_seconds':0,'evaluate_interval':1}
    out=tmp_path/'round';run(c,out,'cpu')
    path=out/'best_policy.pt';spec=dict(config=str(out/'config.json'),model=str(path),model_sha256=digest(path),
        stage='confirmation',seeds=c['confirmation_seeds'],deadline_unix=c['deadline_unix'])
    confirmed=tmp_path/'confirmed';evaluate(spec,confirmed,'cpu')
    summary=read_json(confirmed/'summary.json')
    assert summary['result']['games']==4 and not summary['selection_uses_confirmation']
    assert summary['paired_vs_parent']['environments']==2 and summary['paired_vs_transfer']['environments']==2
    with pytest.raises(ValueError):evaluate({**spec,'stage':'development'},tmp_path/'leak','cpu')


def test_multiprocess_and_corrupt_credit(animal_config):
    t=Trainer({**animal_config,'workers':2,'progress_scale':1.});t.batch()
    for game in t.last_traces:list(replay(game))
    bad=deepcopy(t.last_traces[0]);row=next(r for r in bad['decisions'] if r['trainable']);row['training_reward']+=1
    with pytest.raises(ValueError):list(replay(bad))
    t.close()


def test_resolved_trial_config_with_round_warm_start(animal_config,tmp_path):
    from boardbench.artifacts.store import write_json
    from boardbench.v145.preflight import trial_config
    native=tmp_path/'native.json';write_json(native,animal_config['native_artifact'])
    campaign=dict(training={'learning_rate':1e-5},parent=animal_config['parent'],parent_sha256=animal_config['parent_sha256'],
        warm_start=animal_config['warm_start'],warm_start_sha256=animal_config['warm_start_sha256'],native_manifest=str(native),
        evaluation_opponents=[],workers=2,batch_matches=4,development_environments=2,decision_environments=2,confirmation_environments=2)
    c=trial_config('.',campaign,dict(warm_start='round2.pt',warm_start_sha256='round2'),1)
    assert c['warm_start']=='round2.pt' and c['training_anchor']==animal_config['warm_start']
    assert c['development_seeds']==[245000000,245000001] and c['decision_seeds']==[245011000,245011001]


def test_shared_device_cost_counts_union_not_slots():
    from boardbench.v145.campaign import device_occupancy
    receipts=[dict(gpu='a',started_unix=0,elapsed_seconds=10),dict(gpu='a',started_unix=5,elapsed_seconds=10),
        dict(gpu='b',started_unix=3,elapsed_seconds=6),dict(gpu='a',started_unix=20)]
    assert device_occupancy(receipts,24)==25
