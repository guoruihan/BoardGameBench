from copy import deepcopy
import gzip
import json
from pathlib import Path
import time
import numpy as np
import pytest
import torch

from boardbench.artifacts.store import digest,read_json,write_json,source_id
from boardbench.environments.harmonies_duo import HarmoniesDuo,player_view
from boardbench.v138.baseline import Planner
from boardbench.v129.encoding import NativeEncoder
from boardbench.v140.encoding import encode_batch,action_map,ACTIONS
from boardbench.v146.policy import Habitat,ModelPlayer
from boardbench.v146.worker import audit_trace
from boardbench.v147.search import Search,StrategicHabitat,VARIANTS,public_horizon
from boardbench.v147.worker import play,validate_job
from boardbench.v147.data import reconstruct,build
from boardbench.v147.training import Trainer,LAYOUT,run
from boardbench.v147.imitation import run as imitate
from boardbench.v147.evaluate import run as evaluate
from boardbench.v147.search_campaign import refine
from boardbench.v147.campaign import choose_arms,trial_config,initial_arms
from test_v129 import artifact
from test_v140 import duo_config
from test_v142 import config
from test_v143 import selfplay_config
from test_v144 import score_config
from test_v145 import animal_config
from test_v146 import model,job


@pytest.mark.parametrize('variant',VARIANTS)
def test_rank_legal_public_and_no_mutation(variant):
    env=HarmoniesDuo();obs,_=env.reset(147);old=deepcopy(obs)
    rank=StrategicHabitat(variant).rank(obs)
    assert len(rank)==len(obs['legal_actions']) and obs==old
    assert all(a in obs['legal_actions'] and np.isfinite(v) for v,a in rank)
    if variant=='habitat':assert rank==Habitat().rank(obs)
    assert public_horizon(obs)>=0


def test_terminal_round_complete_and_planner_isolation(model,artifact):
    obs,_=HarmoniesDuo().reset(147);p=ModelPlayer(model,NativeEncoder.from_artifact(artifact))
    search=Search(p,p,0,'balanced',seconds=30);plans=[Planner(),Planner()];old=deepcopy(plans)
    roots=[dict(type='choose_bundle',bundle_id=i) for i in (0,4)]
    values,steps,counts=search.terminal_round(obs,roots,147,plans,time.monotonic()+30)
    assert len(values)==2 and steps>50 and len(counts)==2
    assert [p.__dict__ for p in plans]==[p.__dict__ for p in old]
    assert obs['players'][0]['completed_turns']==0


@pytest.fixture
def expert(job,tmp_path,artifact):
    j={**job,'method':'search_balanced','split':'train','environment_seed':147000001,'seconds':.3,'source_id':source_id()}
    stats,trace=play(j)
    path=tmp_path/'trace.json.gz'
    with gzip.open(path,'wt') as f:json.dump(trace,f)
    stats.update(trajectory=str(path),trajectory_sha256=digest(path),source_id=j['source_id'],
        student_sha256=j['student_sha256'],opponent_sha256=j['opponent_sha256'])
    manifest=tmp_path/'experts.json';write_json(manifest,dict(split='train',games=[stats]))
    built=build([manifest],tmp_path/'data',artifact,time.time()+60)
    return trace,built,tmp_path/'data/manifest.json'


def test_expert_replay_native_plans_and_train_only(expert,artifact):
    trace,meta,path=expert;native=NativeEncoder.from_artifact(artifact)
    fast=reconstruct(trace,native);slow=reconstruct(trace)
    assert len(fast)==sum(r['seat']==trace['seat'] for r in trace['decisions'])
    for a,b in zip(fast,slow):
        for i in (0,1,4,5):np.testing.assert_array_equal(a[i],b[i])
        assert a[1][a[2]] and a[3]==b[3]
    for split,seed in [('development',247000001),('train',347000001),('train',147900001)]:
        with pytest.raises(ValueError):reconstruct({**trace,'split':split,'environment_seed':seed})
    broken=deepcopy(trace);broken['decisions'][0]['mc_return']+=1
    with pytest.raises(ValueError):reconstruct(broken)


@pytest.fixture
def learning_config(animal_config,expert):
    _,_,path=expert
    return {**animal_config,'layout':LAYOUT,'development_seeds':[247100001,247100002],
        'decision_seeds':[247110001,247110002],'confirmation_seeds':[347000001,347000002],
        'expert_manifest':str(path),'expert_manifest_sha256':digest(path),'expert_weight':.1,
        'expert_decay_actions':10000,'expert_steps':1,'expert_minibatch':16,
        'expert_q_weight':.5,'expert_value_weight':.1,'expert_temperature':4.,
        'imitation_steps':2,'imitation_evaluate_interval':1,'finish_reserve_seconds':0}


def test_bc_ppo_exact_resume_and_frozen_opponent(learning_config,tmp_path):
    t=Trainer(learning_config);anchor=deepcopy(t.training_anchor.state_dict())
    bc=t.expert_step();first=t.batch()
    assert bc['expert_ce']>=0 and first['initial_ratio_max_error']<1e-5 and first['actor_change']>0
    assert all(147000000<=r['seed']<148000000 for r in first['per_game'])
    path=tmp_path/'resume.pt';t.save(path);expected=t.batch();weights=deepcopy(t.model.state_dict());traces=deepcopy(t.last_traces);t.close()
    other=Trainer(learning_config);other.load(path);actual=other.batch()
    assert actual['per_game']==expected['per_game'] and traces==other.last_traces
    assert all(torch.equal(v,other.model.state_dict()[k]) for k,v in weights.items())
    assert all(torch.equal(v,other.training_anchor.state_dict()[k]) for k,v in anchor.items())
    assert list(other.references)==['teacher_opponent']
    rng=other.rng.get_state().clone();other.evaluate_panel(learning_config['development_seeds'])
    assert torch.equal(rng,other.rng.get_state());other.close()


def test_initialization_ppo_and_confirmation(learning_config,tmp_path):
    init=tmp_path/'init';imitate(learning_config,init,'cpu')
    assert read_json(init/'done.json')['bc_steps']==2
    cfg={**learning_config,'warm_start':str(init/'best_policy.pt'),'warm_start_sha256':digest(init/'best_policy.pt'),
         'max_batches':1,'evaluate_interval':1}
    out=tmp_path/'rl';run(cfg,out,'cpu')
    assert read_json(out/'done.json')['batches']==1
    spec=dict(config=str(out/'config.json'),model=str(out/'best_policy.pt'),model_sha256=digest(out/'best_policy.pt'),
        stage='confirmation',seeds=cfg['confirmation_seeds'],deadline_unix=cfg['deadline_unix'],
        baseline=learning_config['warm_start'],baseline_sha256=learning_config['warm_start_sha256'])
    evaluate(spec,tmp_path/'confirm','cpu')
    assert read_json(tmp_path/'confirm/summary.json')['paired_vs_parent']['environments']==2


def test_adaptive_decisions_and_leakage():
    table={m:dict(score=90+i,animal=35+i) for i,m in enumerate(['model','search_habitat','search_model','search_space','search_balanced','search_deadline'])}
    methods=refine(table)
    assert 'search_habitat' in methods and 'search_deadline' in methods and 'search_balanced_space' in methods
    rows=[]
    for i,name in enumerate(['control','best']):
        rows.append(dict(name=name,policy=name,sha256=name,config=dict(name=name),evaluation=dict(score_mean=80+i,
            per_game=[dict(seed=s,seat=seat,evaluation_opponent='teacher_opponent',score=80+i,animal=12+i,margin=0,win_credit=.5)
                for s in (247000001,247000002) for seat in (0,1)])))
    arms,decision=choose_arms(rows,1)
    assert arms[2]['expert_weight']==.12 and arms[3]['gae_lambda']==1 and not decision['uses_confirmation']
    rows[0]['evaluation']['per_game'][0]['seed']=347000001
    with pytest.raises(ValueError):choose_arms(rows,1)


def test_job_guards(job):
    j={**job,'method':'search_space','environment_seed':147900001}
    validate_job(j)
    for patch in ({'split':'confirmation','environment_seed':347000001},{'seconds':901},{'method':'bad'}):
        with pytest.raises(ValueError):validate_job({**j,**patch})


def test_candidate_loss_and_raw_return_guards(learning_config):
    t=Trainer(learning_config)
    # Synthetic candidate targets exercise the dense masked loss only; they are
    # not persisted or represented as actual search-generated training evidence.
    t.expert['qmask']=t.expert['mask'].clone()
    t.expert['q']=torch.arange(ACTIONS,dtype=torch.float64).expand_as(t.expert['q']).clone()
    metric=t.expert_step()
    assert np.isfinite(metric['expert_loss']) and metric['expert_q_loss']>0
    t.close()
    for patch in ({'expert_temperature':0.},{'expert_value_weight':-1.},{'progress_scale':1.}):
        with pytest.raises(ValueError):Trainer({**learning_config,**patch})


def test_keep_complementary_bc_skill_and_avoid_duplicate_fallback():
    old=dict(name='old',policy='old.pt',sha256='old',evaluation=dict(score_mean=85,animal_mean=13))
    animal=dict(name='skill',policy='skill.pt',sha256='skill',evaluation=dict(score_mean=73,animal_mean=27))
    arms,decision=initial_arms([old],[animal])
    assert arms[1]['warm_start']=='skill.pt' and arms[2]['warm_start']=='old.pt'
    assert decision['selected_skill']=='skill' and not decision['uses_confirmation']
    arms,decision=initial_arms([old],[])
    assert arms[1]['name']=='explore' and arms[1]['entropy']==.015
    assert decision['selected_skill'] is None


def test_skill_checkpoint_is_separate_from_total_champion(learning_config,tmp_path,monkeypatch):
    original=Trainer.evaluate_panel
    def measured(self,seeds):
        result=original(self,seeds)
        # Synthetic summaries force the complementary-selection branch while
        # using real BC updates, serialization and legal complete-game evaluation.
        result.update(score_mean=80 if self.bc_steps==0 else 75,animal_mean=10 if self.bc_steps==0 else 20+self.bc_steps)
        return result
    monkeypatch.setattr(Trainer,'evaluate_panel',measured)
    out=tmp_path/'skill_init';imitate(learning_config,out,'cpu')
    done=read_json(out/'done.json')
    assert done['selected_bc_steps']==0 and done['complementary_skill']['bc_steps']==2
    assert (out/'skill_decision.json').exists() and (out/'skill_policy.pt').exists()
    best=torch.load(out/'best.pt',weights_only=False);skill=torch.load(out/'skill.pt',weights_only=False)
    assert best['extra_state']['bc_steps']==0 and skill['extra_state']['bc_steps']==2
