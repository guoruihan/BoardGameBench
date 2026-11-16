import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from boardbench.families import pilot_gateway as gw
from boardbench.families.pilot_client import _UnixConnection


LIMITS=dict(calls=3,input_tokens=5000,output_tokens=500,max_input_tokens=2000,max_output_tokens=100)
CONFIG=dict(base_url='http://trusted.invalid/v1',api_key='private-test-key',model='pinned',reasoning_effort='high')
BODY=dict(input=[dict(role='user',content='hello')],model='untrusted',reasoning={'effort':'low'})


@pytest.fixture
def gateway(tmp_path,monkeypatch):
    g=gw.Gateway(tmp_path,CONFIG)
    monkeypatch.setattr(g,'count_upstream',lambda body:(len(json.dumps(body))//3+1,'local_estimate'))
    yield g
    g.close()


def test_reserve_settle_pins_and_caps(gateway):
    c=gateway.create_account('a',LIMITS)
    key,body=gateway.reserve(c['token'],dict(BODY,max_output_tokens=9999))
    assert body['model']=='pinned' and body['reasoning']['effort']=='high'
    assert body['max_output_tokens']==100 and body['store'] is False
    status=gateway.status('a')
    assert status['usage']['calls']==1 and status['usage']['input_tokens']>=256
    assert status['pending_reservations']['calls']==1
    gateway.settle(key,dict(input_tokens=10,output_tokens=5,input_tokens_details={'cached_tokens':8},output_tokens_details={'reasoning_tokens':4}))
    status=gateway.status('a')
    assert status['usage']==dict(calls=1,input_tokens=10,output_tokens=5)
    assert not status['frozen'] and not status['strict_input_tokens']
    gateway.settle(key,dict(input_tokens=9999,output_tokens=9999))
    assert gateway.status('a')['usage']==status['usage']


def test_concurrent_calls_cannot_overbook(gateway):
    c=gateway.create_account('a',dict(LIMITS,calls=1))
    def attempt(_):
        try: return gateway.reserve(c['token'],BODY)[0]
        except gw.GatewayError as e: return e.code
    with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(attempt,range(8)))
    assert sum(r in ('calls','account_inflight') for r in results)==7
    assert gateway.status('a')['usage']['calls']==1


def test_unknown_retains_reservations_and_restart_fails_closed(gateway):
    c=gateway.create_account('a',LIMITS); key,_=gateway.reserve(c['token'],BODY)
    before=gateway.status('a')['usage']
    gateway.close(); restored=gw.Gateway(gateway.root,CONFIG)
    try:
        status=restored.status('a'); assert status['frozen']
        assert status['usage']==before
        with pytest.raises(gw.GatewayError,match='account_frozen'): restored.reserve(c['token'],BODY)
    finally: restored.close()


def test_underestimate_actual_freezes(gateway):
    c=gateway.create_account('a',LIMITS); key,_=gateway.reserve(c['token'],BODY)
    reserved=gateway.status('a')['usage']['input_tokens']
    gateway.settle(key,dict(input_tokens=reserved+1,output_tokens=1))
    status=gateway.status('a'); assert status['frozen']
    assert status['first_binding_limit']=='input_tokens_reservation_exceeded'
    assert status['usage']['input_tokens']==reserved+1


def test_prohibited_paths_tools_and_overinput(gateway):
    c=gateway.create_account('a',LIMITS)
    for change in ({'background':True},{'previous_response_id':'old'},{'tools':[{'type':'web_search'}]},{'tools':[{'type':'mcp'}]}):
        with pytest.raises(gw.GatewayError): gateway.reserve(c['token'],dict(BODY,**change))
    with pytest.raises(gw.GatewayError,match='max_input_tokens_estimate'):
        gateway.reserve(c['token'],dict(BODY,instructions='x'*10000))
    with pytest.raises(gw.GatewayError,match='unauthorized'): gateway.reserve('wrong',BODY)
    assert gateway.status('a')['usage']['calls']==0
    counted=gateway.count_input(c['token'],BODY)
    assert counted['litellm_count_source']=='local_estimate'
    assert counted['reserved_input_tokens']>=counted['input_tokens']*1.25+256


class Reply(io.BytesIO):
    def __init__(self,data,content_type): super().__init__(data); self.headers={'Content-Type':content_type}


def http_call(path,token,payload,socket_path):
    c=_UnixConnection(str(socket_path),3)
    c.request('POST',path,json.dumps(payload),{'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    r=c.getresponse(); status=r.status; data=r.read(); c.close(); return status,data


def test_sse_and_json_usage_over_unix(gateway,tmp_path,monkeypatch):
    c=gateway.create_account('a',LIMITS); seen=[]
    def upstream(body):
        seen.append(body)
        result={'id':'r','usage':{'input_tokens':10,'output_tokens':5}}
        if body.get('stream'):
            text='data: '+json.dumps({'type':'response.output_text.delta','delta':'hi'})+'\n\n'
            text+='data: '+json.dumps({'type':'response.completed','response':result})+'\n\ndata: [DONE]\n\n'
            return Reply(text.encode(),'text/event-stream')
        return Reply(json.dumps(result).encode(),'application/json')
    monkeypatch.setattr(gateway,'open_upstream',upstream)
    sock=tmp_path/'http.sock'; server=gw.make_server(gateway,unix_socket=sock)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        status,data=http_call('/v1/responses',c['token'],dict(BODY,stream=True),sock)
        assert status==200 and b'response.completed' in data
        status,data=http_call('/v1/responses',c['token'],BODY,sock)
        assert status==200 and json.loads(data)['usage']['output_tokens']==5
        status,data=http_call('/v1/responses/input_tokens',c['token'],BODY,sock)
        assert status==200 and json.loads(data)['litellm_count_source']=='local_estimate'
        status,data=http_call('/v1/chat/completions',c['token'],BODY,sock)
        assert status==404 and len(seen)==2
    finally: server.shutdown(); server.server_close(); thread.join()
    assert gateway.status('a')['usage']==dict(calls=2,input_tokens=20,output_tokens=10)


def test_missing_usage_and_transport_error_no_retry(gateway,tmp_path,monkeypatch):
    c=gateway.create_account('a',LIMITS); seen=[]
    def fail(body): seen.append(body); raise OSError('do not expose secret details')
    monkeypatch.setattr(gateway,'open_upstream',fail)
    sock=tmp_path/'http.sock'; server=gw.make_server(gateway,unix_socket=sock)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        status,data=http_call('/v1/responses',c['token'],BODY,sock)
        assert status==502 and b'secret' not in data
        status,data=http_call('/v1/responses',c['token'],BODY,sock)
        assert status==429 and len(seen)==1
    finally: server.shutdown(); server.server_close(); thread.join()
    status=gateway.status('a')
    assert status['frozen'] and status['usage']['output_tokens']==100 and status['unknown_requests']==1


def test_frozen_model_identity_and_referenced_input(gateway):
    c=gateway.create_account('a',LIMITS)
    with pytest.raises(gw.GatewayError,match='unmetered_input_reference_forbidden'):
        gateway.reserve(c['token'],dict(BODY,input=[{'type':'input_image','image_url':'https://example.invalid/image.png'}]))
    with pytest.raises(gw.GatewayError,match='unmetered_state_forbidden'):
        gateway.reserve(c['token'],dict(BODY,prompt={'id':'hidden_prompt'}))
    gateway.close()
    with pytest.raises(ValueError,match='Pinned upstream identity changed'):
        gw.Gateway(gateway.root,dict(CONFIG,model='changed'))


def test_count_endpoint_gets_complete_pinned_payload_and_failure_no_debit(gateway,monkeypatch):
    c=gateway.create_account('a',LIMITS); seen=[]
    def endpoint(body,operation):
        seen.append((body,operation))
        return Reply(json.dumps(dict(input_tokens=123,litellm_count_source='local_estimate')).encode(),'application/json')
    monkeypatch.setattr(gateway,'_open_endpoint',endpoint)
    monkeypatch.setattr(gateway,'count_upstream',gw.Gateway.count_upstream.__get__(gateway))
    body=dict(BODY,instructions='complete instructions',tools=[{'type':'function','name':'tool','parameters':{'type':'object'}}])
    key,pinned=gateway.reserve(c['token'],body)
    assert seen==[(gateway.canonical_count_payload(pinned),'input_tokens')]
    assert json.loads(seen[0][0]['input'][0]['content'][0]['text'])==pinned
    assert pinned['instructions']=='complete instructions' and pinned['tools']==body['tools']
    assert gateway.status('a')['usage']['input_tokens']==410
    assert gateway.status('a')['count_sources']=={'local_estimate':1}
    gateway.settle(key,dict(input_tokens=123,output_tokens=5))
    def fail(*args): raise gw.GatewayError('input_count_unavailable',502)
    monkeypatch.setattr(gateway,'count_upstream',fail)
    with pytest.raises(gw.GatewayError,match='input_count_unavailable'): gateway.reserve(c['token'],body)
    assert gateway.status('a')['usage']['calls']==1
    assert not gateway.preflighting


def test_one_inflight_even_with_many_calls_available(gateway):
    c=gateway.create_account('a',LIMITS)
    key,_=gateway.reserve(c['token'],BODY)
    with pytest.raises(gw.GatewayError,match='account_inflight'): gateway.reserve(c['token'],BODY)
    gateway.settle(key,dict(input_tokens=10,output_tokens=5))
    assert gateway.reserve(c['token'],BODY)


@pytest.mark.parametrize('terminal_type',['response.completed','response.incomplete'])
def test_terminal_settled_before_visible_without_waiting_for_eof(gateway,tmp_path,monkeypatch,terminal_type):
    c=gateway.create_account('a',LIMITS)
    terminal_seen=threading.Event(); settled=threading.Event(); waited_after_terminal=[]
    original_settle=gateway.settle
    def settle(*args,**kwargs):
        original_settle(*args,**kwargs); settled.set()
    monkeypatch.setattr(gateway,'settle',settle)
    class SlowEOF:
        headers={'Content-Type':'text/event-stream'}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def __iter__(self):
            yield b'event: '+terminal_type.encode()+b'\n'
            yield b'data: '+json.dumps({'type':terminal_type,'response':{'usage':{'input_tokens':10,'output_tokens':5}}}).encode()+b'\n'
            yield b'\n'
            waited_after_terminal.append(True)
            terminal_seen.wait(2)
    monkeypatch.setattr(gateway,'open_upstream',lambda body:SlowEOF())
    sock=tmp_path/'race.sock'; server=gw.make_server(gateway,unix_socket=sock)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        connection=_UnixConnection(str(sock),3)
        connection.request('POST','/v1/responses',json.dumps(dict(BODY,stream=True)),{'Authorization':'Bearer '+c['token']})
        response=connection.getresponse()
        while True:
            line=response.readline()
            if line.startswith(b'data:'):
                # The terminal payload cannot be visible until settlement is durable.
                assert settled.is_set()
                assert gateway.status('a')['pending_reservations']['calls']==0
                key,_=gateway.reserve(c['token'],BODY)
                gateway.settle(key,dict(input_tokens=10,output_tokens=5))
                terminal_seen.set()
                break
        response.read(); connection.close()
        assert not waited_after_terminal
    finally:
        terminal_seen.set(); server.shutdown(); server.server_close(); thread.join()
    assert gateway.status('a')['usage']==dict(calls=2,input_tokens=20,output_tokens=10)


def test_private_diagnostics_for_count_http_failure_no_generation(gateway,monkeypatch):
    import urllib.error
    c=gateway.create_account('a',LIMITS)
    class Opener:
        def open(self,request,timeout):
            assert timeout==120
            raise urllib.error.HTTPError(request.full_url,422,'private upstream rejection',{},
                io.BytesIO(b'{"error":"private-test-key unsupported field"}'))
    monkeypatch.setattr(gw.urllib.request,'build_opener',lambda *args:Opener())
    monkeypatch.setattr(gateway,'count_upstream',gw.Gateway.count_upstream.__get__(gateway))
    with pytest.raises(urllib.error.HTTPError): gateway.reserve(c['token'],BODY)
    traces=list((gateway.root/'requests').iterdir()); assert len(traces)==1
    payload=json.loads((traces[0]/'payload.json').read_text())
    assert payload['model']=='pinned' and payload['input']==BODY['input']
    error=json.loads((traces[0]/'input_tokens_error.json').read_text())
    assert error['status']==422 and 'unsupported field' in error['body']
    for path in traces[0].iterdir():
        assert path.stat().st_mode&0o777==0o600
        assert 'private-test-key' not in path.read_text() and c['token'] not in path.read_text()
    assert gateway.status('a')['usage']['calls']==0
    with gw._db(gateway.root) as db: assert db.execute('SELECT count(*) FROM requests').fetchone()[0]==0


def test_private_generate_http_failure_keeps_unknown_usage(gateway,monkeypatch):
    import urllib.error
    c=gateway.create_account('a',LIMITS); key,body=gateway.reserve(c['token'],BODY)
    class Opener:
        def open(self,request,timeout):
            raise urllib.error.HTTPError(request.full_url,500,'internal error',{},io.BytesIO(b'private server diagnostic'))
    monkeypatch.setattr(gw.urllib.request,'build_opener',lambda *args:Opener())
    with pytest.raises(urllib.error.HTTPError): gateway.open_upstream(body)
    gateway.settle(key,None,'upstream_or_transport_failure')
    error=json.loads((gateway.root/'requests'/key/'generate_error.json').read_text())
    assert error['status']==500
    assert gateway.status('a')['frozen'] and gateway.status('a')['unknown_requests']==1


def test_native_additional_tools_fully_counted_without_changing_generation(gateway,monkeypatch):
    c=gateway.create_account('native',dict(LIMITS,input_tokens=100000,max_input_tokens=32768))
    tools=[dict(type='namespace',name='functions',description='native tools',tools=[
        dict(type='custom',name='exec',description='execute local commands',format={'type':'grammar','syntax':'lark','definition':'start: /.+/'}),
        dict(type='function',name='local',parameters={'type':'object','properties':{'text':{'type':'string'}}})])]
    body=dict(BODY,input=[dict(type='additional_tools',id='local_tools',role='developer',tools=tools),
                         dict(type='message',role='user',content=[dict(type='input_text',text='continue')])],
              instructions='Full instructions',client_metadata={'route':'native'},include=['reasoning.encrypted_content'])
    observed=[]
    def counter(payload,operation):
        observed.append(payload)
        decoded=json.loads(payload['input'][0]['content'][0]['text'])
        assert decoded['input'][0]['tools']==tools
        assert decoded['instructions']==body['instructions']
        assert decoded['client_metadata']==body['client_metadata']
        return Reply(json.dumps(dict(input_tokens=12812,litellm_count_source='local_estimate')).encode(),'application/json')
    monkeypatch.setattr(gateway,'_open_endpoint',counter)
    monkeypatch.setattr(gateway,'count_upstream',gw.Gateway.count_upstream.__get__(gateway))
    key,pinned=gateway.reserve(c['token'],body)
    assert pinned['input']==body['input']
    assert gateway.status('native')['usage']['input_tokens']==16271
    assert gateway.status('native')['counting']=='canonical_serialized_input_margin_v2'
    assert json.loads((gateway.root/'requests'/key/'canonical_count_payload.json').read_text())==observed[0]
    gateway.settle(key,dict(input_tokens=10784,output_tokens=5))
    assert not gateway.status('native')['frozen']


@pytest.mark.parametrize('hosted',['web_search','web_search_preview','mcp','file_search','code_interpreter','computer_use_preview'])
def test_nested_hosted_tools_rejected_in_native_additional_tools(gateway,hosted):
    c=gateway.create_account('a',LIMITS)
    nested=dict(type='namespace',name='outer',tools=[dict(type='namespace',name='inner',tools=[dict(type=hosted)])])
    with pytest.raises(gw.GatewayError,match='remote_tools_forbidden'):
        gateway.reserve(c['token'],dict(BODY,input=[dict(type='additional_tools',role='developer',tools=[nested])]))
    with pytest.raises(gw.GatewayError,match='remote_tools_forbidden'):
        gateway.reserve(c['token'],dict(BODY,tools=[nested]))
    assert gateway.status('a')['usage']['calls']==0


def test_shared_wall_deadline_count_plus_slow_stream(gateway,tmp_path,monkeypatch):
    """Frequent SSE activity must not reset the count+generation wall deadline."""
    import time
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            if self.path.endswith('/input_tokens'):
                time.sleep(.08)
                payload=json.dumps(dict(input_tokens=10,litellm_count_source='local_estimate')).encode()
                self.send_response(200); self.send_header('Content-Length',str(len(payload))); self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(b'data: {"type":"response.output_text.delta","delta":"x"}\n\n'); self.wfile.flush()
                        time.sleep(.02)
                except OSError: pass
    upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream); upstream.daemon_threads=True
    thread_up=threading.Thread(target=upstream.serve_forever,daemon=True); thread_up.start()
    gateway.config['base_url']='http://127.0.0.1:'+str(upstream.server_port)+'/v1'
    monkeypatch.setattr(gateway,'count_upstream',gw.Gateway.count_upstream.__get__(gateway))
    c=gateway.create_account('a',LIMITS)
    sock=tmp_path/'deadline.sock'; server=gw.make_server(gateway,unix_socket=sock,request_timeout_seconds=.18)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        started=time.monotonic()
        status,data=http_call('/v1/responses',c['token'],dict(BODY,stream=True),sock)
        elapsed=time.monotonic()-started
        assert status==200 and b'response.output_text.delta' in data
        assert .12<elapsed<.6
        # The watchdog closes the client before the handler's finally block
        # reconciles its conservative reservation. Await that bounded cleanup.
        settled_deadline=time.monotonic()+1
        report=gateway.status('a')
        while not report['frozen'] and time.monotonic()<settled_deadline:
            time.sleep(.005)
            report=gateway.status('a')
        assert report['frozen'] and report['first_binding_limit']=='request_timeout'
        assert report['unknown_requests']==1 and report['usage']['calls']==1
        assert report['usage']['output_tokens']==100
    finally:
        server.shutdown(); server.server_close(); thread.join()
        upstream.shutdown(); upstream.server_close(); thread_up.join()


def test_count_stage_deadline_freezes_without_forwarding_generation(gateway,tmp_path,monkeypatch):
    import time
    def slow_count(body): time.sleep(.12); return 10,'local_estimate'
    monkeypatch.setattr(gateway,'count_upstream',slow_count)
    forwarded=[]
    monkeypatch.setattr(gateway,'open_upstream',lambda body:forwarded.append(body))
    c=gateway.create_account('a',LIMITS)
    sock=tmp_path/'countdeadline.sock'; server=gw.make_server(gateway,unix_socket=sock,request_timeout_seconds=.04)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        import http.client
        started=time.monotonic()
        with pytest.raises(http.client.RemoteDisconnected):
            http_call('/v1/responses',c['token'],BODY,sock)
        assert time.monotonic()-started<.1
        # The watchdog freezes promptly even while a synthetic Python counter
        # has not returned; no generation can follow its eventual return.
        time.sleep(.15)
        report=gateway.status('a')
        assert report['frozen'] and report['first_binding_limit']=='request_timeout_during_count'
        assert report['usage']['calls']==0 and not forwarded
    finally: server.shutdown(); server.server_close(); thread.join()


@pytest.mark.parametrize('before_headers', [False, True])
def test_disconnected_controller_drains_existing_generation(gateway,tmp_path,monkeypatch,before_headers):
    import socket
    import time
    released=threading.Event(); entered=threading.Event(); forwarded=[]
    class Stream:
        headers={'Content-Type':'text/event-stream'}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def __iter__(self):
            yield b'data: {"type":"response.output_text.delta","delta":"start"}\n'
            yield b'\n'
            assert released.wait(2)
            for _ in range(3):
                yield b'data: {"type":"response.output_text.delta","delta":"later"}\n'
                yield b'\n'
            yield b'data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":5}}}\n'
            yield b'\n'
    def upstream(body):
        forwarded.append(body);entered.set()
        if before_headers: assert released.wait(2)
        return Stream()
    monkeypatch.setattr(gateway,'open_upstream',upstream)
    account=gateway.create_account('a',LIMITS)
    sock=tmp_path/'disconnect.sock';server=gw.make_server(gateway,unix_socket=sock)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    client=socket.socket(socket.AF_UNIX);client.settimeout(2);client.connect(str(sock))
    body=json.dumps(dict(BODY,stream=True)).encode()
    headers=('POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer '+account['token']+
             '\r\nContent-Length: '+str(len(body))+'\r\n\r\n').encode()
    try:
        client.sendall(headers+body);assert entered.wait(2)
        if not before_headers: assert client.recv(4096)
        client.shutdown(socket.SHUT_RDWR);client.close();released.set()
        until=time.monotonic()+2
        while gateway.status('a')['pending_reservations']['calls'] and time.monotonic()<until:time.sleep(.005)
        status=gateway.status('a')
        assert status['usage']==dict(calls=1,input_tokens=10,output_tokens=5)
        assert not status['frozen'] and status['unknown_requests']==0 and status['settled_requests']==1
        assert len(forwarded)==1
        assert list((gateway.root/'requests').glob('*/client_disconnect.json'))
        key,_=gateway.reserve(account['token'],BODY)
        gateway.settle(key,dict(input_tokens=10,output_tokens=5))
    finally:
        released.set();client.close();server.shutdown();server.server_close();thread.join()


def test_status_exposes_preflight_before_generation_reservation(gateway,monkeypatch):
    account=gateway.create_account('a',LIMITS);observed=[]
    def counter(body):
        status=gateway.status('a')
        observed.append(status)
        return 10,'local_estimate'
    monkeypatch.setattr(gateway,'count_upstream',counter)
    key,_=gateway.reserve(account['token'],BODY)
    assert observed[0]['preflighting'] and observed[0]['pending_reservations']['calls']==0
    assert not gateway.status('a')['preflighting'] and gateway.status('a')['pending_reservations']['calls']==1
    gateway.settle(key,dict(input_tokens=10,output_tokens=5))


def test_close_model_rejects_new_requests_but_settles_existing(gateway):
    account=gateway.create_account('closing',LIMITS)
    request_id,_=gateway.reserve(account['token'],BODY)
    gateway.close_model('closing')
    gateway.settle(request_id,dict(input_tokens=10,output_tokens=5))
    status=gateway.status('closing')
    assert status['model_closed'] and not status['frozen']
    assert status['pending_reservations']['calls']==0
    assert status['usage']==dict(calls=1,input_tokens=10,output_tokens=5)
    with pytest.raises(gw.GatewayError,match='model_closed'):
        gateway.reserve(account['token'],BODY)


def test_close_during_preflight_cannot_admit_generation(gateway,monkeypatch):
    account=gateway.create_account('closing',LIMITS)
    def count(body):
        gateway.close_model('closing')
        return 1,'injected'
    monkeypatch.setattr(gateway,'count_upstream',count)
    with pytest.raises(gw.GatewayError,match='model_closed'):
        gateway.reserve(account['token'],BODY)
    assert gateway.status('closing')['usage']['calls']==0


def test_count_dns_retry_keeps_one_reservation(gateway,monkeypatch):
    import socket,urllib.error
    account=gateway.create_account('retry',LIMITS);attempts=[];sleeps=[]
    monkeypatch.setattr(gateway,'count_upstream',gw.Gateway.count_upstream.__get__(gateway))
    def endpoint(body,operation):
        attempts.append(operation)
        if len(attempts)<3:raise urllib.error.URLError(socket.gaierror(-2,'injected DNS failure'))
        return Reply(b'{"input_tokens":100}', 'application/json')
    monkeypatch.setattr(gateway,'_open_endpoint',endpoint)
    monkeypatch.setattr(gw.time,'sleep',sleeps.append)
    key,_=gateway.reserve(account['token'],BODY)
    assert attempts==['input_tokens']*3 and sleeps==[.5,1.]
    assert gateway.status('retry')['usage']['calls']==1
    assert gateway.status('retry')['pending_reservations']['calls']==1
    assert len(list((gateway.root/'requests'/key).glob('input_tokens_attempt_*_error.json')))==2
    gateway.settle(key,dict(input_tokens=100,output_tokens=1))


def test_count_dns_exhaustion_never_reserves_generation(gateway,monkeypatch):
    import socket,urllib.error
    account=gateway.create_account('retry',LIMITS);attempts=[]
    monkeypatch.setattr(gateway,'count_upstream',gw.Gateway.count_upstream.__get__(gateway))
    def endpoint(*args):
        attempts.append(1);raise urllib.error.URLError(socket.gaierror(-2,'injected'))
    monkeypatch.setattr(gateway,'_open_endpoint',endpoint)
    monkeypatch.setattr(gw.time,'sleep',lambda _:None)
    with pytest.raises(gw.GatewayError,match='upstream_dns_failure'):
        gateway.reserve(account['token'],BODY)
    assert len(attempts)==3
    status=gateway.status('retry')
    assert status['usage']['calls']==0 and status['pending_reservations']['calls']==0
    assert not status['preflighting']


def test_count_retry_does_not_extend_original_deadline(gateway,monkeypatch):
    import socket,urllib.error
    attempts=[];sleeps=[]
    class Deadline:
        def remaining(self):return .2
    gateway.diagnostic_context.deadline=Deadline()
    def endpoint(*args):
        attempts.append(1);raise urllib.error.URLError(socket.gaierror(-2,'injected'))
    monkeypatch.setattr(gateway,'_open_endpoint',endpoint)
    monkeypatch.setattr(gw.time,'sleep',sleeps.append)
    with pytest.raises(gw.GatewayError,match='upstream_dns_failure'):
        gw.Gateway.count_upstream(gateway,dict(BODY,model='pinned'))
    assert len(attempts)==1 and not sleeps


def test_generation_dns_failure_is_never_retried(gateway,monkeypatch):
    import socket,urllib.error
    attempts=[]
    class Opener:
        def open(self,*args,**kwargs):
            attempts.append(1);raise urllib.error.URLError(socket.gaierror(-2,'injected'))
    monkeypatch.setattr(gw.urllib.request,'build_opener',lambda *args:Opener())
    with pytest.raises(urllib.error.URLError):gateway.open_upstream(BODY)
    assert len(attempts)==1


@pytest.mark.parametrize('failure',['http','malformed'])
def test_count_nontransport_failure_not_retried(gateway,monkeypatch,failure):
    import urllib.error
    calls=[]
    def endpoint(*args):
        calls.append(1)
        if failure=='http':raise urllib.error.HTTPError('http://test.invalid',429,'quota',{},io.BytesIO(b'quota'))
        return Reply(b'not json','application/json')
    monkeypatch.setattr(gateway,'_open_endpoint',endpoint)
    with pytest.raises((urllib.error.HTTPError,ValueError)):
        gw.Gateway.count_upstream(gateway,dict(BODY,model='pinned'))
    assert len(calls)==1


def test_count_deadline_expiring_during_backoff_prevents_second_attempt(gateway,monkeypatch):
    import socket,urllib.error
    calls=[];expired=[]
    class Deadline:
        def remaining(self):
            if expired:raise gw.GatewayError('request_timeout',504)
            return 1.
    gateway.diagnostic_context.deadline=Deadline()
    def endpoint(*args):
        calls.append(1);raise urllib.error.URLError(socket.gaierror(-2,'injected'))
    monkeypatch.setattr(gateway,'_open_endpoint',endpoint)
    monkeypatch.setattr(gw.time,'sleep',lambda _:expired.append(True))
    with pytest.raises(gw.GatewayError,match='request_timeout'):
        gw.Gateway.count_upstream(gateway,dict(BODY,model='pinned'))
    assert len(calls)==1


def test_shared_transport_spacing_across_gateway_instances(tmp_path,monkeypatch):
    clock=[100.];sleeps=[]
    monkeypatch.setattr(gw.time,'monotonic',lambda:clock[0])
    def sleep(delay):sleeps.append(delay);clock[0]+=delay
    monkeypatch.setattr(gw.time,'sleep',sleep)
    config=dict(CONFIG,transport_interval_seconds=3.,transport_clock_path=str(tmp_path/'shared-clock'))
    a=gw.Gateway(tmp_path/'a',config);b=gw.Gateway(tmp_path/'b',config)
    try:
        a._wait_transport_slot();b._wait_transport_slot();a._wait_transport_slot()
        assert sleeps==[3.,3.]
        assert float((tmp_path/'shared-clock').read_text())==106.
    finally:a.close();b.close()


def test_transport_spacing_cannot_extend_request_deadline(gateway,tmp_path,monkeypatch):
    marker=tmp_path/'clock';marker.write_text('100')
    gateway.config.update(transport_interval_seconds=3.,transport_clock_path=str(marker))
    monkeypatch.setattr(gw.time,'monotonic',lambda:100.)
    class Deadline:
        def remaining(self):return 1.
    gateway.diagnostic_context.deadline=Deadline()
    with pytest.raises(gw.GatewayError,match='request_timeout'):gateway._wait_transport_slot()
    assert marker.read_text()=='100'
