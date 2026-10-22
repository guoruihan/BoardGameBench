from copy import deepcopy
import json
import numpy as np
import pytest
import torch

from boardbench.artifacts.store import digest
from boardbench.environments.harmonies_duo import HarmoniesDuo,player_view
from boardbench.v138.baseline import Planner
from boardbench.v140.encoding import encode_batch,action_map,BASE
from boardbench.v142.training import Trainer as MarketTrainer
from boardbench.v142.policy import export_policy as export_market
from boardbench.v143.policy import LAYOUT,own_projection,export_policy,load_policy,SelfplaySolver
from boardbench.v143.training import Trainer,run
from boardbench.v143.preflight import transfer_check
from boardbench.v143.replay import replay
from test_v129 import artifact
from test_v140 import duo_config
from test_v142 import config


@pytest.fixture
def selfplay_config(config,tmp_path):
    t=MarketTrainer(config);path=tmp_path/'market.pt';export_market(t.model,path,config);t.close()
    return {**config,'layout':LAYOUT,'rules_version':HarmoniesDuo.rules_version,'mode':'current','objective':'margin',
        'parent':str(path),'parent_sha256':digest(path),'snapshot_interval':1,'history_size':2,
        'development_seeds':[243000001,243000002],'confirmation_seeds':[343000001,343000002]}


def test_transfer_and_public_opponent_branch(selfplay_config):
    t=Trainer(selfplay_config);proof=transfer_check(t.model)
    assert proof['actions']>50 and proof['max_error']==0
    obs,_=HarmoniesDuo().reset(44);plan=Planner().vector(player_view(obs))
    x=torch.tensor(encode_batch([obs],[plan]),dtype=torch.float64)
    mapping=action_map(obs);mask=torch.tensor([[i in mapping for i in range(244)]])
    altered=x.clone();altered[:,2*BASE:3*BASE]+=.2
    torch.testing.assert_close(own_projection(x),own_projection(altered),atol=0,rtol=0)
    with torch.no_grad():
        assert torch.equal(t.model(x,mask)[0],t.model(altered,mask)[0])
        t.model.opponent[-1].weight.fill_(.001)
        assert not torch.equal(t.model(x,mask)[0],t.model(altered,mask)[0])
        hidden=deepcopy(obs);hidden['private_future_seed']=123
        np.testing.assert_array_equal(encode_batch([obs],[plan]),encode_batch([hidden],[plan]))
    t.close()


@pytest.mark.parametrize('mode',['current','league'])
def test_both_seats_history_reward_resume(selfplay_config,tmp_path,mode):
    c={**selfplay_config,'mode':mode};t=Trainer(c);frozen=deepcopy(t.frozen.state_dict());first=t.batch()
    assert first['learner_trajectories']==2*c['batch_matches']
    assert all(n>0 for n in first['learner_actions_by_seat'])
    assert first['actor_change']>0 and first['context_change']>0 and first['initial_ratio_max_error']<1e-5
    assert sum(len(g['decisions']) for g in t.last_traces)==first['actions']
    for g in t.last_traces: assert len(list(replay(g)))==len(g['decisions'])
    path=tmp_path/'saved.pt';t.save(path);expected=t.batch();weights=deepcopy(t.model.state_dict())
    traces=deepcopy(t.last_traces);snapshot_weights=[deepcopy(m.state_dict()) for m in t.snapshots];t.close()
    restored=Trainer(c);restored.load(path);actual=restored.batch()
    assert expected['per_game']==actual['per_game'] and traces==restored.last_traces
    assert actual['learner_trajectories']==(7 if mode=='league' else 8)
    assert actual['history_matches']==(1 if mode=='league' else 0)
    assert actual['learner_actions_by_seat'][0]>0 and actual['learner_actions_by_seat'][1]>0
    assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in weights.items())
    assert all(torch.equal(v,restored.frozen.state_dict()[k]) for k,v in frozen.items())
    assert all(torch.equal(v,restored.snapshots[i].state_dict()[k]) for i,s in enumerate(snapshot_weights) for k,v in s.items())
    for g in restored.last_traces:
        assert len(list(replay(g)))==len(g['decisions'])
        assert all(('reward_to_next_own_decision' in r)==r['trainable'] for r in g['decisions'])
    rng=restored.rng.get_state().clone();result=restored.evaluate(c['development_seeds'])
    assert result['games']==4 and result['environments']==2 and torch.equal(rng,restored.rng.get_state())
    restored.close()


def test_initial_self_battle_and_portable_policy(selfplay_config,tmp_path):
    t=Trainer(selfplay_config);ev=t.evaluate(selfplay_config['development_seeds'])
    assert ev['margin_mean']==0 and ev['win_credit_mean']==.5
    path=tmp_path/'policy.pt';export_policy(t.model,path,selfplay_config);m=load_policy(path)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in t.model.state_dict().items())
    obs,_=HarmoniesDuo().reset(17)
    assert SelfplaySolver(m).decide(obs)==SelfplaySolver(t.model).decide(obs)
    t.close()


def test_initial_fallback_confirmation_and_guards(selfplay_config,tmp_path):
    out=tmp_path/'completed';c={**selfplay_config,'finish_reserve_seconds':9999}
    run(c,out,'cpu');done=json.loads((out/'done.json').read_text())
    assert done['selected_batch']==0 and done['confirmation']['win_credit_mean']==.5
    assert done['confirmation']['margin_mean']==0 and done['confirmation']['games']==4
    with pytest.raises(ValueError):run(c,out,'cpu',resume=True)
    for delta in ({'layout':'wrong'},{'objective':'score'},{'mode':'weak_opponents'},
                  {'development_seeds':[242000001]},{'confirmation_seeds':[243000001]},{'parent_sha256':'wrong'}):
        with pytest.raises(ValueError):Trainer({**selfplay_config,**delta})


def test_postselection_crossplay(selfplay_config,tmp_path):
    from boardbench.v143.crossplay import run as crossplay
    t=Trainer(selfplay_config);path=tmp_path/'same.pt';export_policy(t.model,path,selfplay_config);t.close()
    out=tmp_path/'cross';crossplay(selfplay_config,path,path,out,device='cpu')
    result=json.loads((out/'result.json').read_text())
    assert result['games']==4 and result['environments']==2
    assert result['margin_mean']==0 and result['win_credit_mean']==.5
    assert result['selection_uses_crossplay'] is False and result['model_a_sha256']==result['model_b_sha256']
