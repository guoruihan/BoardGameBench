import json

import pytest

from boardbench.families.pool_resources import learner_config, cpu_rate, descendants, parse_proc_stat, enrich_report
from boardbench.families.pool_report import report


def test_only_python_learner_is_root_not_timeout_wrapper():
    command=['/env/bin/python','-m','boardbench.families.pool_training','--config','/run/config.json']
    assert learner_config(command)=='/run/config.json'
    assert learner_config(['timeout','120s']+command) is None
    assert learner_config(['/env/bin/python','-m','other','--config','/run/config.json']) is None
    assert learner_config(['/env/bin/python','-m']) is None


def test_proc_stat_handles_parentheses_and_cpu_identity_changes():
    fields=['S','10']+['0']*20
    fields[11:15]=['100','50','20','10'];fields[19]='300';fields[21]='25'
    row=parse_proc_stat('19 (name with ) spaces) '+' '.join(fields),100,4096)
    assert row==dict(ppid=10,start_ticks=300,cpu_seconds=1.8,rss_bytes=102400)
    previous=dict(alive=True,process_identity=[19,300],cpu_seconds=3.)
    current=dict(previous,cpu_seconds=7.)
    assert cpu_rate(current,previous,12.,10.)==2.
    assert cpu_rate(dict(current,process_identity=[19,301]),previous,12.,10.) is None
    assert descendants(2,{2:{'ppid':1},3:{'ppid':2},4:{'ppid':3},5:{'ppid':1}})=={2,3,4}


def test_resume_resource_summary_uses_only_new_decisions_and_costs(tmp_path,monkeypatch):
    import boardbench.families.pool_resources as module
    monkeypatch.setattr(module.time,'time',lambda:110.)
    (tmp_path/'config.json').write_text(json.dumps({'resume_provenance':{
        'starting_learner_decisions':1000,'starting_costs':{'rollout_seconds':9.}}}))
    (tmp_path/'state.json').write_text(json.dumps(dict(began_unix=100.,learner_decisions=1300,costs={'rollout_seconds':13.})))
    summary=enrich_report({'jobs':[{'job':'game/B'}]},[dict(game='game',arm='B',output=str(tmp_path))])
    assert summary['jobs'][0]['decisions_per_wall_second']==30.
    assert summary['jobs'][0]['cost_seconds']['rollout_seconds']==4.


def test_report_follows_multiple_replacements_and_rejects_cycle(tmp_path):
    original=tmp_path/'runs/game/B';second=tmp_path/'repair';last=tmp_path/'resume'
    for path in (original,second,last):
        path.mkdir(parents=True)
        (path/'config.json').write_text(json.dumps(dict(game='game',arm='B')))
    (original/'replacement.json').write_text(json.dumps(dict(output=str(second))))
    (second/'replacement.json').write_text(json.dumps(dict(output=str(last))))
    (last/'state.json').write_text(json.dumps(dict(phase='training',next_update=12)))
    assert report(tmp_path)[0]['updates']==12
    (last/'replacement.json').write_text(json.dumps(dict(output=str(original))))
    with pytest.raises(ValueError,match='Cycle'):report(tmp_path)
