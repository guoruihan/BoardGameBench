from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from boardbench.artifacts.store import read_json, write_json
from boardbench.v121.registry import make_splits
from boardbench.v125.campaign import cpu_core_limit, prepare_plan, make_jobs
from boardbench.v125.historical import validate_reference
from boardbench.v125.report import publish


def plan_and_reference():
    raw={'split':'development','panels':[2,4],'environment_indices':[64,65,66,67],
         'search_seeds':[11,12],'references':['original'],
         'arms':[{'id':a,'method':'terminal','multiplier':1,'timeout_seconds':1020,
                  'params':{'mode':'timed','continuation':a,'seconds':900}}
                 for a in ('original','mixed_roots')]}
    plan=prepare_plan(raw,make_splits(train_count=8))
    rows=[{'arm':'original_900s','environment_seed':e,'search_seed':s,
           'params':{'mode':'timed','continuation':'original','seconds':900},
           'status':'completed','natural_terminal':True,'score':40+i,
           'components':{'total':40+i,'animals':10},'decision_seconds':900.01}
          for i,e in enumerate(plan['environment_seeds']) for s in plan['search_seeds']]
    ref={'schema':1,'split':'development','rules_version':plan['rules_version'],
         'split_sha256':plan['split_sha256'],'environment_seeds':plan['environment_seeds'],
         'search_seeds':plan['search_seeds'],'arms':['original_900s'],'rows':rows,
         'source_id':'frozen','caveat':'mixed historical CPU loads'}
    return plan,ref


def test_expanded_cpu_profile_is_host_scoped_and_reserves_physical_cores(monkeypatch):
    cpus=list(range(128))
    assert cpu_core_limit(cpus,{})==64
    monkeypatch.setattr('boardbench.v125.campaign.os.uname',lambda:SimpleNamespace(nodename='b35.cluster.helixon.com'))
    plan={'resource_profile':'b35_reserve16_v1'}
    assert cpu_core_limit(cpus,plan)==112
    assert cpu_core_limit(list(range(96)),plan)==80
    assert cpu_core_limit(list(range(256)),plan)==112
    assert cpu_core_limit(list(range(16)),plan)==0
    with pytest.raises(ValueError,match='only on b35'):
        cpu_core_limit(cpus,{'resource_profile':'unlimited'})
    monkeypatch.setattr('boardbench.v125.campaign.os.uname',lambda:SimpleNamespace(nodename='other-node'))
    with pytest.raises(ValueError,match='only on b35'):
        cpu_core_limit(cpus,plan)
    assert cpu_core_limit(cpus,{})==64


@pytest.mark.parametrize('available,count,profile',[(128,113,'b35_reserve16_v1'),
    (96,81,'b35_reserve16_v1'),(128,65,'default_half')])
def test_launcher_rejects_excess_cores_before_creating_run(tmp_path,monkeypatch,available,count,profile):
    from boardbench.v122 import campaign as base
    from boardbench.v125 import campaign as backend
    monkeypatch.setattr(base,'physical_cpus',lambda:list(range(available)))
    monkeypatch.setattr(backend.os,'uname',lambda:SimpleNamespace(nodename='b35.cluster.helixon.com'))
    raw={'cpu_cores':count,'resource_profile':profile,'deadline_unix':time.time()+90}
    config=tmp_path/'config.json';write_json(config,raw)
    root=tmp_path/'outputs/v125/rejected'
    with pytest.raises(ValueError,match='CPU limit'):
        base.launch(config,root,tmp_path,backend=backend)
    assert not root.exists()


@pytest.mark.parametrize('corruption',['seed','split','rules','duplicate','missing','score','timeout','budget','policy'])
def test_historical_reference_rejects_mismatches(corruption):
    plan,reference=plan_and_reference();validate_reference(reference,plan)
    reference=deepcopy(reference)
    if corruption=='seed':reference['search_seeds']=[11,13]
    elif corruption=='split':reference['split']='test'
    elif corruption=='rules':reference['rules_version']='other'
    elif corruption=='duplicate':reference['rows'].append(deepcopy(reference['rows'][0]))
    elif corruption=='missing':reference['rows'].pop()
    elif corruption=='score':reference['rows'][0]['score']=float('nan')
    elif corruption=='timeout':reference['rows'][0]['status']='job_timeout'
    elif corruption=='budget':reference['rows'][0]['params']['seconds']=300
    elif corruption=='policy':reference['rows'][0]['params']['continuation']='habitat'
    with pytest.raises(ValueError):validate_reference(reference,plan)


def test_history_uses_only_complete_paired_environment_prefix(tmp_path):
    plan,reference=plan_and_reference();plan['historical_reference']=reference
    jobs=make_jobs(plan);rows=[]
    for j in jobs:
        if j['environment_index']>=2:continue
        score=40+j['environment_index']+(5 if j['arm']=='mixed_roots' else 0)
        rows.append({**j,'status':'completed','natural_terminal':True,'score':score,
          'components':{'total':score,'animals':10},'decision_seconds':900.,'simulation_steps':100,
          'solver_budget_utilization':1.,'terminal_metrics':dict(terrain=score-10,animals_placed=2,completed_cards=1,turns=12,tokens=36)})
    summary=publish(tmp_path,plan,jobs,rows,{'phase':'running'})
    c=summary['historical_comparison']['comparisons']['mixed_roots-vs-historical-original_900s']
    assert c['n']==2 and c['mean_gap']==5 and c['normal_95_ci']==[5,5]
    assert not c['same_batch_comparison'] and c['actual_mean_times_within_10_percent']
    assert summary['historical_comparison']['search_repetitions']==2
    assert '历史900秒对照' in (tmp_path/'REPORT.md').read_text()
    assert publish(tmp_path,plan,jobs,rows[:-1],{'phase':'running'})['historical_comparison']['balanced_environments']==0


def test_expanded_config_exact_legacy_matrix_and_no_stage1_overlap():
    root=Path(__file__).resolve().parents[1]
    raw=read_json(root/'configs/v125_expanded_eval.json')
    legacy=read_json(root/'configs/v124_harmonies_fullgame_20261020.json')
    assert raw['environment_indices']==list(range(legacy['environment_offset'],legacy['environment_offset']+legacy['panels'][-1]))
    assert raw['search_seeds']==legacy['search_seeds']
    assert not set(raw['environment_indices'])&set(range(96,108))
    plan=prepare_plan(raw,make_splits(train_count=8))
    jobs=make_jobs(plan);assert len(jobs)==480
    assert all(j['params']['seconds']==900 for j in jobs)
    pairs=[{(j['environment_seed'],j['search_seed']) for j in jobs if j['arm']==a['id']} for a in plan['arms']]
    assert all(p==pairs[0] for p in pairs) and len(pairs[0])==96
    calibration=prepare_plan(read_json(root/'configs/v125_expanded_calibration.json'),make_splits(train_count=8))
    assert len(make_jobs(calibration))==100 and calibration['split']=='train_calibration'
