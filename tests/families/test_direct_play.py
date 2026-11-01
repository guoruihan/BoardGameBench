import json
from pathlib import Path
import sys

import pytest

from boardbench.families import direct_play as d
from boardbench.families.registry import game_class, GAME_IDS
from boardbench.families.play import FeaturePolicy


@pytest.mark.parametrize('game',[g for g in GAME_IDS if g!='five_tribes'])
def test_catalog_and_public_request(game):
    env=game_class(game)();obs=env.reset(170001)
    legal=env.legal_actions();catalog=d.public_catalog(game)
    request=d.request_for(obs,legal,'printed rules',catalog,'private plan')
    json.dumps(request,allow_nan=False)
    assert 'seed' not in request and 'snapshot' not in request
    assert request['observation']=={k:v for k,v in obs.items() if k!='legal_actions'}
    assert request['memory']=='private plan'
    table=request['legal_action_table']
    reconstructed=[{table['columns'][i]:row[i] for i in present}
                   for row,present in zip(table['rows'],table['present'])]
    assert reconstructed==legal
    assert json.dumps(legal)==json.dumps(env.legal_actions())


@pytest.mark.parametrize('text',['{"action_index":true}','{"action_index":-1}',
    '{"action_index":1}','{"action_index":0.0}','[]','not json',
    '{"action_index":0,"memory":3}', json.dumps({'action_index':0,'memory':'a'*2001})])
def test_invalid_answer(text):
    with pytest.raises((ValueError,TypeError)):d.parse_choice(text,1)


def test_answer_and_lossless_null():
    assert d.parse_choice('```json\n{"action_index":0,"memory":"plan"}\n```',2)==(0,'plan')
    table=d.action_table([{'type':'a'},{'type':'a','x':None}])
    assert table['present'][0]!=table['present'][1]


def test_matched_schedule():
    def entry(candidate=False):
        members={'legacy':{'spec':{'kind':'neural'}},'search_900':{'spec':{'kind':'terminal_search','seconds':900}}}
        if candidate:members['candidate']={'spec':{'kind':'guided_rich'}}
        return {'members':members}
    manifest={'games':{'one':entry(),'two':entry(True)}}
    jobs=d.make_jobs(manifest,['A','B','C'])
    assert len(jobs)==60
    for game,opponents in d.select_opponents(manifest).items():
        for opponent in opponents:
            panels=[]
            for label in ['A','B','C']:
                panels.append({(j['seed'],j['seat']) for j in jobs if
                    (j['game'],j['opponent'],j['label'])==(game,opponent,label)})
            assert panels[0]==panels[1]==panels[2]
            assert len(panels[0])==4


def test_real_complete_game_and_error_unknown(tmp_path, monkeypatch):
    game='toy_battle';cls=game_class(game)
    manifest={'manifest_sha256':'fixture','games':{game:{'members':{
        'legacy':{'policy_id':'fixture','spec':{'kind':'neural'}}}}}}
    monkeypatch.setattr(d,'load_opponent',lambda *args: FeaturePolicy(cls,{'kind':'heuristic'},42))
    def answer(command,request,timeout,cwd):
        table=request['legal_action_table']
        legal=[{table['columns'][i]:row[i] for i in present}
               for row,present in zip(table['rows'],table['present'])]
        action=FeaturePolicy(cls,{'kind':'heuristic'},71).decide(request['observation'],legal)
        return {'ok':True,'text':json.dumps({'action_index':legal.index(action)})}
    monkeypatch.setattr(d,'invoke',answer)
    job=dict(game=game,opponent='legacy',seat=0,seed=170003,label='fixture')
    result=d.play_job('.',manifest,job,{'command':[]},'rules',tmp_path/'good')
    assert result['status']=='complete' and result['outcome']['terminated']
    assert (tmp_path/'good/trajectory.jsonl').read_text()
    monkeypatch.setattr(d,'invoke',lambda *args:{'ok':False,'error':'rate_limit'})
    failed=d.play_job('.',manifest,job,{'command':[]},'rules',tmp_path/'failed')
    assert failed['status']=='provider_rate_limit'
    assert not failed['outcome']['terminated'] and failed['score_breakdown'] is None
    assert (tmp_path/'failed/resume.pkl').exists()
    monkeypatch.setattr(d,'invoke',answer)
    resumed=d.play_job('.',manifest,job,{'command':[]},'rules',tmp_path/'failed')
    assert resumed['status']=='complete'
    assert resumed['calls']==result['calls']+1
    assert resumed['outcome']==result['outcome']
    assert not (tmp_path/'failed/resume.pkl').exists()


def test_subprocess_timeout(tmp_path):
    response=d.invoke([sys.executable,'-c','import time; time.sleep(20)'],{},.1,tmp_path)
    assert not response['ok'] and response['error']=='adapter_timeout'
