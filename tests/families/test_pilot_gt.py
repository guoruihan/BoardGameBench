import json
import threading

import pytest

from boardbench.families import pilot_gt as gt
from boardbench.families.pilot_client import PilotClient, PilotError


class FakeEnv:
    def __init__(self): self.n=0; self.current_player=0
    def reset(self,seed,first_player=0): self.seed=seed
    def legal_actions(self): return [{'move':0},{'move':1}]
    def observe(self,seat): return {'player':seat,'turn':self.n}
    def step(self,action): self.n+=1; self.current_player=1-self.current_player
    def outcome(self): return {'terminated':self.n>=12,'winners':[0],'scores':[10,5]}


class FakeBot:
    def decide(self,obs,legal): return legal[0]


@pytest.fixture
def service(tmp_path,monkeypatch):
    manifest={'manifest_sha256':'pool','games':{'harmonies':{
        'config':gt.CONFIG,'rules_version':'1','members':{'legacy':{
            'policy_id':'policy','spec':{'kind':'harmonies_rl','sha256':'weight'}}}}}}
    monkeypatch.setattr(gt,'verify_pool',lambda _:manifest)
    monkeypatch.setattr(gt,'game_class',lambda _:FakeEnv)
    monkeypatch.setattr(gt,'load_opponent',lambda *args:FakeBot())
    root=tmp_path/'private'; gt.initialize(root,tmp_path/'pool')
    server=gt.PilotService(root)
    yield server
    server.close()


def call(s,c,method,key=None,**kwargs):
    import uuid
    return s.request(c['token'],key or uuid.uuid4().hex,dict(method=method,generation=c['generation'],**kwargs))


def test_disjoint_paired_seed_domains():
    key='01'*32
    domains=[{gt.private_seed(key,d,i) for i in range(50)} for d in gt.DOMAINS]
    assert len(set.union(*domains))==200
    assert gt.private_seed(key,'calibration',4)==gt.private_seed(key,'calibration',4)


def test_limits_auth_versions_and_persistent_retries(service):
    c=gt.create_run(service.root)
    assert not call(service,dict(c,token='bad'),'new_game')['ok']
    assert call(service,c,'new_game',seed=1)['error']=='fixed_schedule'
    first=call(service,c,'new_game',key='first')
    assert first['games']==1
    assert call(service,c,'new_game',key='first')==first
    assert call(service,c,'new_game')['error']=='active_game_limit'
    assert call(service,c,'act',version=999,action_index=0)['error']=='stale_observation'
    assert service.report(c['run_id'])['steps']==0
    assert call(service,c,'act',version=0,action_index=0,key='act')['steps']==2
    assert call(service,c,'act',version=0,action_index=0,key='act')['steps']==2
    assert call(service,c,'act',version=0,action_index=1,key='act')['error']=='request_id_conflict'
    for _ in range(9):
        call(service,c,'abort'); assert call(service,c,'new_game')['ok']
    call(service,c,'abort')
    assert call(service,c,'new_game')['error']=='game_limit'
    service.close()
    restarted=gt.PilotService(service.root)
    try:
        assert call(restarted,c,'new_game',key='first')==first
        assert call(restarted,c,'new_game')['error']=='game_limit'
    finally: restarted.close()


def test_test_schedule_lease_and_errors(service):
    c=gt.create_run(service.root,phase='calibration',pair_index=3,seat=1)
    assert call(service,c,'new_game',seat=0)['error']=='fixed_schedule'
    call(service,c,'new_game')
    assert service.live[c['run_id']]['seat']==1
    seed=service.live[c['run_id']]['env'].seed
    other=gt.create_run(service.root,phase='calibration',pair_index=3,seat=0,budget={'profile':'L40'})
    call(service,other,'new_game')
    assert service.live[other['run_id']]['env'].seed==seed
    rotated=service.rotate_controller(c['run_id'],'program')
    assert call(service,c,'act',version=1,action_index=0)['error']=='unauthorized'
    for _ in range(3): assert call(service,rotated,'act',version=1,action_index=-1)['error']=='invalid_action'
    report=service.report(c['run_id'])
    assert report['sources']['default_random']==1
    before=report['steps']
    service.fallback(c['run_id'],reason='decision_timeout')
    assert call(service,rotated,'act',version=before,action_index=0)['error']=='controller_revoked'
    service.stop(c['run_id'],'cpu_exhausted')
    assert service.report(c['run_id'])['outcomes'][-1]['win_points']==0
    assert call(service,rotated,'new_game')['error']=='run_closed'


def test_http_program_loop_and_no_hidden_api(service,tmp_path):
    c=gt.create_run(service.root,phase='smoke')
    c=service.rotate_controller(c['run_id'],'program')
    sock=tmp_path/'gt.sock'; server=gt.make_server(service,unix_socket=sock)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        client=PilotClient(token=c['token'],generation=c['generation'],unix_socket=str(sock))
        client.new_game()
        assert client.legal_actions(limit=1)['total']==2
        with pytest.raises(PilotError): client.request('snapshot')
        assert client.run(lambda obs,actions:None)['status']=='NEED_AGENT'
        assert service.report(c['run_id'])['errors']==0
        assert client.run(lambda obs,actions:actions[0]['action_index'],max_actions=5)['actions']==5
        assert client.run(lambda obs,actions:0)['status']=='complete'
        report=service.report(c['run_id'])
        assert report['sources']['program']==6 and report['sources']['opponent']==6
        assert report['outcomes'][0]['own_score']==10
        public=json.dumps(client.history())
        assert 'seed' not in public and 'weight' not in public
        with pytest.raises(PilotError,match='game_limit'): client.new_game()
    finally: server.shutdown(); server.server_close(); thread.join()


def test_crash_does_not_resume_or_reset(service):
    c=gt.create_run(service.root); call(service,c,'new_game')
    service.close(); restarted=gt.PilotService(service.root)
    try:
        assert restarted.report(c['run_id'])['status']=='infrastructure_interrupted'
        assert call(restarted,c,'new_game')['error']=='run_closed'
        assert restarted.report(c['run_id'])['games']==1
    finally: restarted.close()


def test_total_errors_force_takeover_and_cap(service,monkeypatch):
    monkeypatch.setattr(FakeEnv,'outcome',lambda self:dict(terminated=self.n>=100,winners=[0],scores=[10,5]))
    c=gt.create_run(service.root,phase='smoke'); call(service,c,'new_game')
    for _ in range(30):
        version=service.report(c['run_id'])['version']
        assert call(service,c,'act',version=version,action_index=-1)['error']=='invalid_action'
    report=service.report(c['run_id'])
    assert report['status']=='complete' and report['errors']==30
    assert report['sources']['default_random']==50
    monkeypatch.setattr(FakeEnv,'outcome',lambda self:dict(terminated=False,winners=[],scores=[0,0]))
    other=gt.create_run(service.root,phase='smoke'); call(service,other,'new_game')
    service.fallback(other['run_id'],finish=True)
    report=service.report(other['run_id'])
    assert report['status']=='step_cap' and report['steps']==3000
    assert 'own_score' not in report['outcomes'][0]


def test_forced_actions_do_not_use_controller(service,monkeypatch):
    monkeypatch.setattr(FakeEnv,'legal_actions',lambda self:[{'move':0}])
    c=gt.create_run(service.root,phase='smoke'); result=call(service,c,'new_game')
    assert result['status']=='complete'
    assert result['sources']=={'forced':6,'opponent':6}


def test_trusted_configured_gt_quota_survives_budget_display_update(service):
    c=gt.create_run(service.root,budget={'max_gt_games':12})
    service.update_budget(c['run_id'],{'limits':{'max_gt_games':999}})
    for _ in range(12):
        assert call(service,c,'new_game')['ok']
        assert call(service,c,'abort')['ok']
    assert call(service,c,'new_game')['error']=='game_limit'
    assert service.report(c['run_id'])['games']==12


def test_small_development_quota_and_nondev_limit(service):
    c=gt.create_run(service.root,budget={'max_gt_games':1})
    assert call(service,c,'new_game')['ok'];call(service,c,'abort')
    assert call(service,c,'new_game')['error']=='game_limit'
    with pytest.raises(ValueError):gt.create_run(service.root,phase='calibration',budget={'max_gt_games':2})


def test_trusted_shared_seed_keys_are_not_exported(service,tmp_path):
    key='ab'*32
    identity=gt.initialize(tmp_path/'paired',tmp_path/'pool',seed_key=key)
    assert 'seed_key' not in identity
    saved=json.loads((tmp_path/'paired/private.json').read_text())
    assert saved['seed_key']==key
    assert gt.private_seed(key,'development',0)!=gt.private_seed(key,'calibration',0)
