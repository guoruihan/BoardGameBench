from copy import deepcopy
import json
import time
import numpy as np
import pytest
import torch
from boardbench.artifacts.store import digest
from boardbench.v127.training import advantages
from boardbench.v143.replay import replay
from boardbench.v144.training import Trainer,run
from boardbench.v144.policy import LAYOUT,export_policy,load_policy
from boardbench.v144.evaluation import paired_delta
from boardbench.v144.decisions import first_round,next_round
from test_v129 import artifact
from test_v140 import duo_config
from test_v142 import config
from test_v143 import selfplay_config


@pytest.fixture
def score_config(selfplay_config):
    return {**selfplay_config,'layout':LAYOUT,'objective':'score','evaluation_opponents':[],
        'development_seeds':[244000001,244000002],'decision_seeds':[244010001,244010002],
        'confirmation_seeds':[344000001,344000002],'deadline_unix':time.time()+300}


@pytest.mark.parametrize('objective,mode,lam',[('score','current',.95),('score','league',1.),('score','diverse',.95),('margin','current',.95)])
def test_rewards_roles_full_return_resume(score_config,tmp_path,objective,mode,lam):
    c={**score_config,'objective':objective,'mode':mode,'gae_lambda':lam}
    t=Trainer(c);first=t.batch();first_traces=deepcopy(t.last_traces)
    assert first['actor_change']>0 and first['context_change']>0 and first['initial_ratio_max_error']<1e-5
    assert first['learner_trajectories']==(7 if mode=='diverse' else 8)
    for game in first_traces:
        steps=list(replay(game,objective));assert len(steps)==len(game['decisions'])
        if objective=='score':
            for seat in game['learner_seats']:
                reward=sum(r['reward_to_next_own_decision'] for r in game['decisions'] if r['seat']==seat)
                assert abs(reward*150-(game['score'] if seat==game['seat'] else game['opponent_score']))<1e-8
            if game['score'] and game['opponent_score']:
                with pytest.raises(ValueError):list(replay(game,'margin'))
    path=tmp_path/'saved.pt';t.save(path);expected=t.batch();weights=deepcopy(t.model.state_dict());traces=deepcopy(t.last_traces);t.close()
    restored=Trainer(c);restored.load(path);actual=restored.batch()
    assert actual['per_game']==expected['per_game'] and traces==restored.last_traces
    assert actual['learner_trajectories']=={'current':8,'league':7,'diverse':6}[mode]
    assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in weights.items())
    for game in restored.last_traces:list(replay(game,objective))
    rng=restored.rng.get_state().clone();ev=restored.evaluate_panel(c['development_seeds'])
    assert ev['games']==4 and ev['environments']==2 and torch.equal(rng,restored.rng.get_state())
    assert restored.metric(ev)==ev['score_mean' if objective=='score' else 'margin_mean']
    restored.close()


def test_mc_targets_and_pairing():
    rewards=[.1,.2,-.05];adv,targets=advantages(rewards,[.7,.4,.2],1.)
    np.testing.assert_allclose(targets,[.25,.15,-.05],rtol=0,atol=1e-12)
    games=[dict(seed=s,seat=seat,evaluation_opponent=o,score=70+s,animal=5,margin=0,win_credit=.5)
           for s in (1,2) for seat in (0,1) for o in ('a','b')]
    delta=paired_delta(dict(per_game=games),dict(per_game=games))
    assert delta['games']==8 and delta['environments']==2 and delta['metrics']['score']['ci95']==[0,0]
    with pytest.raises(ValueError):paired_delta(dict(per_game=games),dict(per_game=games[:-1]))


def test_portable_panel_reference_and_warm_start(score_config,tmp_path):
    t=Trainer(score_config);t.batch();path=tmp_path/'policy.pt';export_policy(t.model,path,score_config)
    model=load_policy(path);assert all(torch.equal(v,model.state_dict()[k]) for k,v in t.model.state_dict().items())
    c={**score_config,'warm_start':str(path),'warm_start_sha256':digest(path),
        'evaluation_opponents':[dict(name='old',path=str(path),sha256=digest(path),format='v144')]}
    new=Trainer(c);rng=new.rng.get_state().clone();anchor=deepcopy(new.frozen.state_dict())
    ev=new.evaluate_panel(c['development_seeds']);assert ev['games']==8 and ev['environments']==2
    assert new.optimizer.state_dict()['state']=={} and torch.equal(rng,new.rng.get_state())
    assert all(torch.equal(v,new.frozen.state_dict()[k]) for k,v in anchor.items())
    assert ev['opponents']['old']['margin_mean']==0 and ev['opponents']['old']['win_credit_mean']==.5
    for change in ({'warm_start_sha256':'wrong'},{'decision_seeds':[344000001]},
                   {'decision_seeds':c['development_seeds']},{'objective':'wins'},{'gae_lambda':1.1}):
        with pytest.raises(ValueError):Trainer({**c,**change})
    t.close();new.close()


def test_no_training_initial_retention_and_no_confirmation(score_config,tmp_path):
    out=tmp_path/'trial';run({**score_config,'finish_reserve_seconds':9999},out,'cpu')
    done=json.loads((out/'done.json').read_text());assert done['selected_batch']==0
    assert done['best_development_metric']==json.loads((out/'initial.json').read_text())['score_mean']
    assert not (out/'confirmation.json').exists()
    with pytest.raises(ValueError):run(score_config,out,'cpu',resume=True)


def test_decision_driven_variants():
    arms=first_round();records=[]
    for i,arm in enumerate(arms):
        games=[dict(seed=s,seat=seat,evaluation_opponent='transfer',score=80+i,animal=10,
                    margin=i,win_credit=.5) for s in (244010001,244010002) for seat in (0,1)]
        records.append(dict(config={**arm,'learning_rate':1e-5,'entropy':.005},policy=f'p{i}',sha256=f'h{i}',
            evaluation=dict(score_mean=80+i,animal_mean=10,per_game=games)))
    new,decision=next_round(records,1)
    assert new[0]['warm_start']=='p2' and new[2]['gae_lambda']==.95 and new[3]['warm_start']=='p3'
    assert decision['uses_confirmation'] is False and new[1]['mode']=='diverse'
    new,decision=next_round(records,2)
    assert new[2]['entropy']==.01 and new[2]['replica']==1


def test_real_small_round_handoff_and_confirmation(score_config,tmp_path):
    from boardbench.v144.evaluate import run as evaluate
    from boardbench.artifacts.store import write_json
    c={**score_config,'max_batches':1,'finish_reserve_seconds':0,'evaluate_interval':1}
    first=tmp_path/'r1';run(c,first,'cpu')
    assert json.loads((first/'done.json').read_text())['batches']==1
    parent=first/'best_policy.pt'
    second=tmp_path/'r2';next_c={**c,'warm_start':str(parent),'warm_start_sha256':digest(parent),'mode':'diverse','gae_lambda':1.}
    run(next_c,second,'cpu');assert json.loads((second/'done.json').read_text())['batches']==1
    spec=dict(stage='confirmation',config=str(second/'config.json'),model=str(second/'best_policy.pt'),
              model_sha256=digest(second/'best_policy.pt'),seeds=c['confirmation_seeds'])
    result=tmp_path/'confirmed';evaluate(spec,result,'cpu')
    summary=json.loads((result/'summary.json').read_text())
    assert summary['selection_uses_confirmation'] is False
    assert summary['result']['games']==4 and summary['paired_vs_transfer']['environments']==2
    with pytest.raises(ValueError):evaluate({**spec,'stage':'development'},tmp_path/'leak','cpu')


def test_runner_completed_skip_and_deadline(score_config,tmp_path):
    import os
    from boardbench.artifacts.store import write_json
    from boardbench.v144.campaign import Runner
    root=tmp_path/'controller';root.mkdir();(root/'logs').mkdir();(root/'rounds').mkdir();(root/'decisions').mkdir()
    write_json(root/'resolved_config.json',dict(started_unix=time.time()-10,deadline_unix=time.time()+30,gpu_uuids=['unused']))
    write_json(root/'launch.json',dict(cpus=sorted(os.sched_getaffinity(0))))
    runner=Runner(tmp_path,root);marker=root/'completed.json';write_json(marker,dict(passed=True))
    runner.run([dict(complete=str(marker))],time.time()+20,'skip')
    assert not runner.receipts
    with pytest.raises(TimeoutError):runner.run([dict(complete=str(root/'missing'))],time.time()-1,'expired')
    runner.deadline=time.time()-1
    with pytest.raises(InterruptedError):runner.start({},0)
    runner.close()
