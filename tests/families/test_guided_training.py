import hashlib
import numpy as np
import pytest
import torch

from boardbench.families.guided_data import reward, targets, teacher_distribution, dump_packet, load_packet
from boardbench.families.guided_neural import (GuidedActorCritic, GuidedPolicy, base_policy,
    encode_record, save_model, load_model)
from boardbench.families.guided_training import supervised_loss, behavior_error, ppo_update, rollout
from boardbench.families.registry import game_class
from boardbench.families.rich_observation import PublicTreeEncoder


@pytest.fixture
def sample():
    torch.set_num_threads(1);torch.manual_seed(173)
    game='splendor';env=game_class(game)();obs=env.reset(1731);actions=env.legal_actions()
    encoder=PublicTreeEncoder(game);base=base_policy(game,{'kind':'heuristic'},17)
    row=encode_record(game,encoder,base,obs,actions)
    row.update(win_target=1.,margin_target=.3,own_target=.2,
               auxiliary_targets=np.array([.5,.2],np.float32),auxiliary_mask=np.array([True,False]),
               teacher=np.eye(len(actions),dtype=np.float32)[(int(row['prior'].argmax())+1)%len(actions)])
    model=GuidedActorCritic(encoder.metadata['vocabulary_size'],16,2)
    schema=dict(game=game,names=['own_score','score_margin'],means=[0.,0.],scales=[10.,10.],margin_scale=10.)
    return game,env,obs,actions,encoder,row,model,dict(base_spec={'kind':'heuristic'},schema=schema,anchor_scale=1.)


def test_initial_anchor_and_model_roundtrip(sample,tmp_path):
    game,env,obs,actions,encoder,row,model,metadata=sample
    logits,_,_=model([row]);assert int(logits.argmax())==int(row['prior'].argmax())
    path=tmp_path/'model.pt';save_model(path,model,encoder,metadata)
    restored,e,m=load_model(path,game)
    torch.testing.assert_close(model([row])[0],restored([row])[0])
    policy=GuidedPolicy(game,{'path':str(path)})
    assert policy.decide(obs,actions)==actions[int(row['prior'].argmax())]
    with pytest.raises(ValueError):load_model(path,'clever')


def test_supervision_updates_context_and_action_branches(sample):
    game,_,_,_,_,row,model,_=sample
    loss,_=supervised_loss(model,[row],'cpu','mixed',game);loss.backward()
    assert torch.isfinite(loss)
    assert model.core.state.node[0].weight.grad.abs().sum()>0
    assert model.core.actor[-1].weight.grad.abs().sum()>0
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.step();optimizer.zero_grad()
    loss,_=supervised_loss(model,[row],'cpu','mixed',game);loss.backward()
    assert model.core.action.node[0].weight.grad.abs().sum()>0


def test_teacher_probability_alignment_and_insufficient_rounds():
    actions=[{'x':0},{'x':1},{'x':2}]
    detail={'candidates':[dict(action=actions[2],samples=8,mean=.9),dict(action=actions[0],samples=8,mean=.1)]}
    p=teacher_distribution(actions,actions[2],detail,'win')
    assert p.sum()==pytest.approx(1.) and p[2]>.9 and p[1]==0
    q=teacher_distribution(actions[::-1],actions[2],detail,'win');np.testing.assert_allclose(p[::-1],q)
    detail['candidates'][0]['samples']=1;detail['candidates'][1]['samples']=1
    np.testing.assert_array_equal(teacher_distribution(actions,actions[2],detail,'win'),[0,0,1])


def test_terminal_targets_are_seat_aligned_and_toy_reward_is_win():
    result=dict(game='splendor',status='complete',rewards=[-1.,1.],outcome={'scores':[10,15]},score_breakdown=[{},{}])
    schema=dict(names=['own_score','score_margin'],means=[10,0],scales=[5,5],margin_scale=5)
    a,b=[targets(result,p,schema) for p in (0,1)]
    assert a['win_target']==-1 and b['win_target']==1
    assert a['margin_target']==pytest.approx(-b['margin_target'])
    assert reward(b,'mixed','splendor')==pytest.approx(.5+.5*np.tanh(1))
    assert reward(a,'mixed','toy_battle')==-1


def test_behavior_alignment_and_real_ppo_update(sample):
    game,_,_,_,_,row,model,_=sample
    with torch.no_grad():logits,value,_=model([row])
    a=int(logits.argmax());r=dict(row,action=a,old_logp=float(torch.log_softmax(logits[0],-1)[a]),
                                value=float(value[0]),advantage=1.,return_=1.)
    r['return']=1.
    assert behavior_error(model,[r],'cpu')<1e-6
    bad=dict(r,old_logp=r['old_logp']+.01)
    with pytest.raises(ValueError):behavior_error(model,[bad],'cpu')
    before=model.core.actor[-1].weight.detach().clone()
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    cfg=dict(game=game,epochs=1,batch_size=2,deadline_unix=1e12,reward_mode='win')
    metrics=ppo_update(model,opt,[r,dict(r,advantage=-1.)],[row],cfg,np.random.default_rng(1),'cpu')
    assert metrics['minibatches']==1 and np.isfinite(metrics['loss'])
    assert not torch.equal(before,model.core.actor[-1].weight)


def test_incomplete_episode_excludes_all_training_records(sample,tmp_path):
    game,_,_,_,encoder,_,model,metadata=sample
    model_path=tmp_path/'model.pt';save_model(model_path,model,encoder,metadata)
    output=tmp_path/'episode.pkl.z'
    item=rollout(dict(game=game,path=str(model_path),seed=1731,seat=0,opponent={'kind':'heuristic'},
                     max_seconds=0.,model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                     output=str(output),reward_mode='win'))
    packet=load_packet(output,item['sha256'])
    assert packet['result']['status']=='truncated' and packet['records']==[]
    with pytest.raises(ValueError):load_packet(output,'bad_hash')


def test_evaluation_denominator_and_score_components():
    from boardbench.families.guided_evaluation import summarize_rows
    row=dict(status='complete',seed=1,learner_seat=1,rewards=[-1,1],seconds=1.,
        outcome={'scores':[10,20]},score_breakdown=[{'animals':2},{'animals':8}],policy_seconds=[.1,.2])
    result=summarize_rows([row],2)
    assert result['all_scheduled_win_rate']==.5 and not result['all_complete']
    assert result['own_score']==20 and result['opponent_score']==10
    assert result['own_components']['animals']==8


def test_selective_teacher_keeps_prior_unless_supported_override(sample):
    from boardbench.families.guided_training import teacher_mode
    row=sample[5];base=int(row['prior'].argmax())
    r=dict(row,action=base,teacher_rounds=100)
    target=teacher_mode([r],'selective')[0]['teacher']
    expected=np.exp(row['prior']-row['prior'].max());expected/=expected.sum()
    np.testing.assert_allclose(target,expected,rtol=1e-6)
    changed=dict(r,action=(base+1)%len(target))
    assert teacher_mode([changed],'selective')[0]['teacher'][changed['action']]==1.
    changed['teacher_rounds']=0
    np.testing.assert_allclose(teacher_mode([changed],'selective')[0]['teacher'],expected,rtol=1e-6)
    np.testing.assert_array_equal(r['teacher'],row['teacher'])


def test_frozen_vocabulary_checkpoint_and_legacy_sidecar(sample,tmp_path):
    import json
    game,_,obs,actions,encoder,_,_,metadata=sample
    symbols=sorted(list(encoder.symbols)+['/old/source/location/game.py'])
    old=PublicTreeEncoder(game,symbols=symbols)
    model=GuidedActorCritic(old.metadata['vocabulary_size'],16,2)
    path=tmp_path/'old.pt';save_model(path,model,old,metadata)
    restored,encoding,_=load_model(path,game)
    np.testing.assert_array_equal(old.encode(obs),encoding.encode(obs))
    payload=torch.load(path,weights_only=True);payload.pop('symbols');torch.save(payload,path)
    with pytest.raises(ValueError):load_model(path,game)
    (tmp_path/'vocabulary.json').write_text(json.dumps(dict(game=game,symbols=symbols)))
    restored,encoding,_=load_model(path,game)
    assert encoding.metadata==old.metadata


def test_student_labeling_preserves_executed_trajectory(monkeypatch,tmp_path):
    import gzip,json
    from boardbench.families import guided_search_evaluation as module
    from boardbench.families.play import run_match
    class FakeTeacher:
        def __init__(self,*args,**kwargs):self.spent=0.;self.total_steps=0;self.total_rounds=0;self.last={}
        def decide(self,env):
            actions=env.legal_actions();self.total_rounds+=1
            self.last=dict(completed_rounds=4,candidates=[dict(action=a,samples=4,mean=float(i==len(actions)-1)) for i,a in enumerate(actions)])
            return actions[-1]
    monkeypatch.setattr(module,'TerminalReference',FakeTeacher)
    reference=dict(rl_spec={'kind':'heuristic'},objective='win',continuation='portfolio')
    cfg=dict(game='splendor',seed=17391,search_seat=0,mode='label',reference=reference,
             candidate={'kind':'heuristic'},budget=1,output=str(tmp_path/'label.json'),source_commit='test')
    result=module.run(cfg)
    expected=run_match('splendor',[{'kind':'heuristic'}]*2,seed=cfg['seed'],record=True,record_legal=True)
    assert result['status']=='complete' and result['outcome']==expected['outcome']
    with gzip.open(tmp_path/'label.trajectory.json.gz','rt') as f:trace=json.load(f)['trajectory']
    assert [r['action'] for r in trace]==[r['action'] for r in expected['trajectory']]
    assert any(r['teacher_action']!=r['action'] for r in trace if r['player']==0)
    assert all(r['teacher_action'] in r['legal_actions'] for r in trace if r['player']==0)


def test_anchor_temperature_is_aligned_and_does_not_mutate_dataset(sample):
    from boardbench.families.guided_training import prepare_demos
    row=sample[5];row=dict(row,action=int(row['prior'].argmax()),teacher_rounds=4)
    before=row['prior'].copy()
    data=dict(rows=[row],anchor_scale=1.)
    new=prepare_demos(data,dict(anchor_scale=.25,teacher_mode='hard'))[0]
    np.testing.assert_array_equal(new['prior'],before*.25)
    np.testing.assert_array_equal(row['prior'],before)
    assert new['prior'].argmax()==row['prior'].argmax()
    with pytest.raises(ValueError):prepare_demos(data,dict(anchor_scale=0))


def test_wide_selection_is_separate_and_retries_incomplete(tmp_path):
    import json
    from boardbench.families.guided_evaluation import best_candidates,collect
    root=tmp_path;directory=root/'runs'/'splendor'/'win';directory.mkdir(parents=True)
    (directory/'config.json').write_text(json.dumps(dict(game='splendor',arm='win')))
    candidate=dict(name='bc_final',path='/model',sha256='abc')
    (directory/'candidates.json').write_text(json.dumps([candidate]))
    for split,complete,rate in [('quick',True,1.),('wide',False,.9)]:
        p=root/'evaluations'/split/'splendor'/'win'/'bc_final.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(game='splendor',arm='win',candidate=candidate,
          summary=dict(all_complete=complete,all_scheduled_win_rate=rate,own_score=15,opponent_score=10))))
    assert best_candidates(root)['splendor']['summary']['all_scheduled_win_rate']==1.
    assert best_candidates(root,'wide')=={}
    assert collect(root)==[] and len(collect(root,'wide'))==1


def test_mixed_teacher_execution_keeps_actual_returns(monkeypatch,tmp_path):
    import gzip,json
    from boardbench.families import guided_search_evaluation as module
    from boardbench.families.play import make_policy,run_match
    class Teacher:
        def __init__(self,cls,*args,**kwargs):
            self.policy=make_policy(cls,{'kind':'heuristic'},0)
            self.spent=0.;self.total_steps=0;self.total_rounds=0;self.last={}
        def decide(self,env):
            actions=env.legal_actions();self.total_rounds+=1
            self.last=dict(completed_rounds=4,candidates=[])
            return self.policy.decide(env.observe(env.current_player),actions)
    monkeypatch.setattr(module,'TerminalReference',Teacher)
    reference=dict(rl_spec={'kind':'heuristic'},objective='win',continuation='portfolio')
    cfg=dict(game='toy_battle',seed=173143,search_seat=0,mode='label',reference=reference,
             candidate={'kind':'random'},budget=1,output=str(tmp_path/'mixed.json'),source_commit='test',
             teacher_execute_probability=1.)
    result=module.run(cfg)
    expected=run_match('toy_battle',[{'kind':'heuristic'}]*2,seed=cfg['seed'],record=True)
    assert result['status']=='complete' and result['outcome']==expected['outcome']
    with gzip.open(tmp_path/'mixed.trajectory.json.gz','rt') as f:trace=json.load(f)['trajectory']
    assert [r['action'] for r in trace]==[r['action'] for r in expected['trajectory']]
    assert all(r['behavior_policy']=='teacher' and r['action']==r['teacher_action'] for r in trace if r['player']==0)
    with pytest.raises(ValueError):module.run(dict(cfg,mode='match'))
    with pytest.raises(ValueError):module.run(dict(cfg,teacher_execute_probability=1.1))


def test_residual_shrink_preserves_prior_values_and_source(sample):
    from boardbench.families.guided_variants import shrink_residual
    row,model=sample[5],sample[6]
    with torch.no_grad():
        model.core.actor[-1].weight.fill_(.13)
        model.feature_actor[-1].bias.fill_(.27)
    before=model.core.actor[-1].weight.detach().clone()
    logits,value,aux=model([row]);scaled=shrink_residual(model,.25)
    after,v,a=scaled([row]);prior=torch.as_tensor(row['prior'])[None]
    torch.testing.assert_close(after-prior,.25*(logits-prior),atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(v,value);torch.testing.assert_close(a,aux)
    torch.testing.assert_close(model.core.actor[-1].weight,before)
    torch.testing.assert_close(shrink_residual(model,0)([row])[0],prior)
    with pytest.raises(ValueError):shrink_residual(model,1.1)
