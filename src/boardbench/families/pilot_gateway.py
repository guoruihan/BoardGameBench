"""Trusted Responses proxy with durable, conservative reservation accounting.

Input estimation uses the upstream Responses input_tokens endpoint over the
complete pinned payload serialized as a single text message (including native
additional_tools schemas), then a 25% + 256-token reservation margin. The upstream
may report local_estimate, so this is NOT exact token counting. Usage settles the
ledger; underestimated reservations freeze the account. This is explicitly an
estimated-input pilot, not a claim of exact hard input-token enforcement.
"""
from contextlib import contextmanager
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import secrets
from pathlib import Path
import socketserver
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request

LIMITS = ('calls','input_tokens','output_tokens','max_input_tokens','max_output_tokens')
COUNTING = 'canonical_serialized_input_margin_v2'


class GatewayError(Exception):
    def __init__(self, code, status=400):
        self.code=code; self.status=status
        super().__init__(code)


class _Deadline:
    """A single wall deadline shared by body, token counting and generation."""
    def __init__(self,gateway,seconds):
        self.gateway=gateway; self.end=time.monotonic()+seconds
        self.account=None; self.request_id=None; self.expired=False
        self.sockets=set(); self.lock=threading.Lock()
        self.timer=threading.Timer(seconds,self.expire); self.timer.daemon=True
        self.timer.start()

    def remaining(self):
        remaining=self.end-time.monotonic()
        if self.expired or remaining<=0: raise GatewayError('request_timeout',504)
        return remaining

    def track(self,sock):
        with self.lock:
            self.sockets.add(sock); expired=self.expired
        if expired:
            try: sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            raise GatewayError('request_timeout',504)
        return sock

    def untrack(self,sock):
        with self.lock: self.sockets.discard(sock)

    def expire(self):
        with self.lock:
            self.expired=True; sockets=list(self.sockets)
        # shutdown interrupts another thread's blocking read; close() can wait
        # on buffered-reader locks and must not be used by this watchdog.
        for sock in sockets:
            try: sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass
        if self.request_id:
            self.gateway.settle(self.request_id,None,'request_timeout')
        elif self.account:
            with self.gateway.mutex,_db(self.gateway.root) as db:
                d=self.gateway._account(db,self.account)
                self.gateway._freeze(d,'request_timeout_during_count')
                self.gateway._save(db,self.account,d)

    def finish(self): self.timer.cancel()


def _deadline_handlers(deadline):
    class TrackedHTTP(http.client.HTTPConnection):
        def connect(self):
            connector=self._create_connection
            def tracked(*args,**kwargs):
                deadline.remaining()
                return deadline.track(connector(*args,**kwargs))
            self._create_connection=tracked
            super().connect(); deadline.track(self.sock); deadline.remaining()
    class TrackedHTTPS(http.client.HTTPSConnection):
        def connect(self):
            connector=self._create_connection
            def tracked(*args,**kwargs):
                deadline.remaining()
                return deadline.track(connector(*args,**kwargs))
            self._create_connection=tracked
            http.client.HTTPConnection.connect(self)
            self.sock=self._context.wrap_socket(self.sock,
                server_hostname=self._tunnel_host or self.host,do_handshake_on_connect=False)
            deadline.track(self.sock); self.sock.do_handshake(); deadline.remaining()
    class HTTP(urllib.request.HTTPHandler):
        def http_open(self,request): return self.do_open(TrackedHTTP,request)
    class HTTPS(urllib.request.HTTPSHandler):
        def https_open(self,request): return self.do_open(TrackedHTTPS,request,context=self._context)
    return HTTP(),HTTPS()


def _hash(value): return hashlib.sha256(value.encode()).hexdigest()

def _json(value): return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)


@contextmanager
def _db(root):
    connection=sqlite3.connect(str(Path(root)/'gateway.sqlite'),timeout=30)
    connection.row_factory=sqlite3.Row
    connection.execute('PRAGMA synchronous=FULL')
    try:
        with connection: yield connection
    finally: connection.close()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): raise GatewayError('upstream_redirect_blocked',502)


class Gateway:
    def __init__(self,root,upstream_config):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.config=dict(upstream_config)
        if not all(isinstance(self.config.get(k),str) and self.config[k] for k in ('base_url','api_key','model','reasoning_effort')):
            raise ValueError('Incomplete private upstream config')
        self.mutex=threading.RLock(); self.preflighting=set(); self.diagnostic_context=threading.local()
        # A shared account database must have exactly one gateway owner.
        import fcntl
        self.lock=(self.root/'gateway.lock').open('a')
        try: fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException: self.lock.close(); raise
        with _db(self.root) as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS accounts(
              id TEXT PRIMARY KEY, token_hash TEXT UNIQUE, data TEXT);
              CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, account TEXT,
              state TEXT, input_reserved INTEGER, output_reserved INTEGER, data TEXT);
              CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);''')
            identity=_hash(_json({k:self.config[k] for k in ('base_url','model','reasoning_effort')}))
            saved=db.execute("SELECT value FROM metadata WHERE key='upstream_identity'").fetchone()
            if saved and saved[0]!=identity:
                self.lock.close()
                raise ValueError('Pinned upstream identity changed; use a new gateway ledger')
            db.execute("INSERT OR IGNORE INTO metadata VALUES('upstream_identity',?)",(identity,))
            for row in db.execute("SELECT * FROM requests WHERE state='pending'").fetchall():
                d=self._account(db,row['account']); self._freeze(d,'unknown_usage_after_restart')
                d['unknown_requests']+=1
                self._save(db,row['account'],d)
                db.execute("UPDATE requests SET state='unknown' WHERE id=?",(row['id'],))
        (self.root/'gateway.sqlite').chmod(0o600)

    def close(self): self.lock.close()

    def _account(self,db,account_id):
        row=db.execute('SELECT data FROM accounts WHERE id=?',(account_id,)).fetchone()
        if not row: raise GatewayError('unknown_account',404)
        return json.loads(row['data'])

    @staticmethod
    def _freeze(d,reason):
        d['frozen']=True
        if d['first_binding_limit'] is None: d['first_binding_limit']=reason
        d['events'].append(dict(kind=reason,time=time.time()))

    @staticmethod
    def _save(db,account_id,d):
        db.execute('UPDATE accounts SET data=? WHERE id=?',(_json(d),account_id))

    def create_account(self,account_id,limits):
        if not isinstance(account_id,str) or not account_id or set(limits)!=set(LIMITS):
            raise ValueError('Invalid account or limits')
        if any(type(limits[k]) is not int or limits[k]<0 for k in LIMITS):
            raise ValueError('Limits must be nonnegative integers')
        token=secrets.token_urlsafe(32)
        d=dict(limits=limits,usage=dict(calls=0,input_tokens=0,output_tokens=0),
               frozen=False,first_binding_limit=None,events=[],counting=COUNTING,
               strict_input_tokens=False,settled_requests=0,unknown_requests=0,count_sources={})
        with self.mutex,_db(self.root) as db:
            db.execute('INSERT INTO accounts VALUES(?,?,?)',(account_id,_hash(token),_json(d)))
        return dict(account_id=account_id,token=token)

    def status(self,account_id):
        with self.mutex,_db(self.root) as db:
            d=self._account(db,account_id)
            rows=db.execute("SELECT input_reserved,output_reserved FROM requests WHERE account=? AND state='pending'",(account_id,)).fetchall()
            d['pending_reservations']=dict(calls=len(rows),input_tokens=sum(r[0] for r in rows),output_tokens=sum(r[1] for r in rows))
            d['preflighting']=account_id in self.preflighting
            d['remaining']={k:max(0,d['limits'][k]-d['usage'][k]) for k in ('calls','input_tokens','output_tokens')}
            return dict(account_id=account_id,**d)

    def close_model(self, account_id, reason='controller_exited'):
        """Stop new generations without discarding settlement of existing ones."""
        with self.mutex, _db(self.root) as db:
            d = self._account(db, account_id)
            if not d.get('model_closed'):
                d['model_closed'] = True
                d['events'].append(dict(kind='model_closed', reason=reason, time=time.time()))
                self._save(db, account_id, d)

    def _authenticate(self,db,token):
        if not isinstance(token,str): raise GatewayError('unauthorized',401)
        row=db.execute('SELECT id,data FROM accounts WHERE token_hash=?',(_hash(token),)).fetchone()
        if not row: raise GatewayError('unauthorized',401)
        deadline=getattr(self.diagnostic_context,'deadline',None)
        if deadline: deadline.account=row['id']; deadline.remaining()
        return row['id'],json.loads(row['data'])

    def _body(self,body):
        if not isinstance(body,dict): raise GatewayError('invalid_request')
        body=json.loads(_json(body))
        if body.get('background') or body.get('previous_response_id') or body.get('conversation') or body.get('prompt'):
            raise GatewayError('unmetered_state_forbidden')
        if body.get('stream') not in (None,True,False): raise GatewayError('invalid_stream')
        def check_tools(tools,depth=0):
            if not isinstance(tools,list) or depth>8: raise GatewayError('remote_tools_forbidden')
            for tool in tools:
                if not isinstance(tool,dict): raise GatewayError('remote_tools_forbidden')
                kind=tool.get('type')
                if kind=='namespace': check_tools(tool.get('tools'),depth+1)
                elif kind not in ('function','custom'): raise GatewayError('remote_tools_forbidden')
        check_tools(body.get('tools',[]))
        if not isinstance(body.get('input'),(str,list)): raise GatewayError('input_required')
        def check_input(value):
            if isinstance(value,dict):
                if value.get('type')=='additional_tools':
                    check_tools(value.get('tools'))
                if value.get('type') in ('input_image','input_file','input_audio','item_reference','image_url','image','file'):
                    raise GatewayError('unmetered_input_reference_forbidden')
                for child in value.values(): check_input(child)
            elif isinstance(value,list):
                for child in value: check_input(child)
        check_input(body['input'])
        body['model']=self.config['model']; body['store']=False; body['background']=False
        reasoning=body.get('reasoning') or {}
        if not isinstance(reasoning,dict): raise GatewayError('invalid_reasoning')
        body['reasoning']=dict(reasoning,effort=self.config['reasoning_effort'])
        return body

    def _diagnostic(self,trace_id,name,value):
        """Trusted-only artifacts. Never include authorization headers or config."""
        directory=self.root/'requests'/trace_id
        directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        (self.root/'requests').chmod(0o700)
        payload=_json(value).replace(self.config['api_key'],'[REDACTED_UPSTREAM_KEY]')
        fd=os.open(directory/name,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as out: out.write(payload)

    def _begin_diagnostic(self,body,account=None):
        trace_id=secrets.token_hex(16); self.diagnostic_context.trace_id=trace_id
        self._diagnostic(trace_id,'payload.json',body)
        self._diagnostic(trace_id,'metadata.json',dict(account=account,started=time.time()))
        return trace_id

    @staticmethod
    def canonical_count_payload(body):
        # Some upstream counters ignore non-message input types, notably native
        # additional_tools. Serialize ALL generation fields into ordinary text
        # for counting only; generation still receives the untouched pinned body.
        return dict(model=body['model'],input=[dict(role='user',content=[
            dict(type='input_text',text=_json(body))])])

    def count_upstream(self,body):
        deadline=getattr(self.diagnostic_context,'deadline',None)
        if deadline: deadline.remaining()
        canonical=self.canonical_count_payload(body)
        trace_id=getattr(self.diagnostic_context,'trace_id',None)
        if trace_id: self._diagnostic(trace_id,'canonical_count_payload.json',canonical)
        with self._open_endpoint(canonical,'input_tokens') as response:
            payload=json.loads(response.read(1024*1024))
        if trace_id: self._diagnostic(trace_id,'input_tokens_response.json',payload)
        if deadline: deadline.remaining()
        count=payload.get('input_tokens')
        if type(count) is not int or count<0: raise GatewayError('input_count_unavailable',502)
        return count,payload.get('litellm_count_source','upstream_unspecified')

    def count_input(self,token,body):
        body=self._body(body)
        with self.mutex,_db(self.root) as db: account,_=self._authenticate(db,token)
        self._begin_diagnostic(body,account)
        estimate,source=self.count_upstream(body)
        return dict(object='response.input_tokens',input_tokens=estimate,
                    reserved_input_tokens=math.ceil(estimate*1.25)+256,
                    litellm_count_source=source,counting=COUNTING,strict_input_tokens=False)

    def reserve(self,token,body):
        body=self._body(body)
        # Counting is network I/O outside the ledger lock. A preflight lease
        # prevents same-account races without blocking other account/status work.
        with self.mutex,_db(self.root) as db:
            account,d=self._authenticate(db,token)
            if d['frozen']: raise GatewayError('account_frozen',429)
            if d.get('model_closed'): raise GatewayError('model_closed',429)
            if account in self.preflighting or db.execute(
                "SELECT 1 FROM requests WHERE account=? AND state='pending'",(account,)).fetchone():
                raise GatewayError('account_inflight',429)
            limits=d['limits']; used=d['usage']
            if used['calls']>=limits['calls']:
                if d['first_binding_limit'] is None: d['first_binding_limit']='calls'
                d['events'].append(dict(kind='rejected',limit='calls',time=time.time()))
                self._save(db,account,d); db.commit()
                raise GatewayError('calls',429)
            requested=body.get('max_output_tokens',limits['max_output_tokens'])
            if type(requested) is not int or requested<1: raise GatewayError('invalid_output_limit')
            allowed=min(requested,limits['max_output_tokens'],limits['output_tokens']-used['output_tokens'])
            if allowed<1:
                if d['first_binding_limit'] is None: d['first_binding_limit']='output_tokens'
                d['events'].append(dict(kind='rejected',limit='output_tokens',time=time.time()))
                self._save(db,account,d); db.commit()
                raise GatewayError('output_tokens',429)
            body['max_output_tokens']=allowed
            self.preflighting.add(account)
        try:
            request_id=self._begin_diagnostic(body,account)
            estimate,source=self.count_upstream(body)
            deadline=getattr(self.diagnostic_context,'deadline',None)
            if deadline: deadline.remaining()
            input_reserved=math.ceil(estimate*1.25)+256
            with self.mutex,_db(self.root) as db:
                db.execute('BEGIN IMMEDIATE'); d=self._account(db,account)
                limits=d['limits']; used=d['usage']
                reason=None
                if d['frozen']: reason='account_frozen'
                elif d.get('model_closed'): reason='model_closed'
                elif input_reserved>limits['max_input_tokens']: reason='max_input_tokens_estimate'
                elif input_reserved>limits['input_tokens']-used['input_tokens']: reason='input_tokens'
                if reason:
                    if d['first_binding_limit'] is None: d['first_binding_limit']=reason
                    d['events'].append(dict(kind='rejected',limit=reason,time=time.time()))
                    self._save(db,account,d); db.commit()
                    raise GatewayError(reason,429)
                used['calls']+=1; used['input_tokens']+=input_reserved; used['output_tokens']+=allowed
                d.setdefault('count_sources',{})[str(source)]=d.get('count_sources',{}).get(str(source),0)+1
                self._save(db,account,d)
                db.execute('INSERT INTO requests VALUES(?,?,?,?,?,?)',(request_id,account,'pending',input_reserved,allowed,
                    _json(dict(estimated_input=estimate,count_source=source,counting=COUNTING,body_sha256=_hash(_json(body)),started=time.time()))))
                if deadline: deadline.request_id=request_id
                return request_id,body
        finally:
            with self.mutex: self.preflighting.discard(account)

    def settle(self,request_id,usage,error=None):
        with self.mutex,_db(self.root) as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone()
            if not row or row['state']!='pending': return
            d=self._account(db,row['account']); record=json.loads(row['data'])
            valid=isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0 for k in ('input_tokens','output_tokens'))
            if not valid:
                self._freeze(d,error or 'unknown_usage'); d['unknown_requests']+=1; state='unknown'
            else:
                for key,reserved in (('input_tokens',row['input_reserved']),('output_tokens',row['output_reserved'])):
                    d['usage'][key]+=usage[key]-reserved
                    if usage[key]>reserved: self._freeze(d,key+'_reservation_exceeded')
                    if usage[key]>d['limits']['max_'+key] or d['usage'][key]>d['limits'][key]: self._freeze(d,key+'_actual_limit_exceeded')
                # Cached input is already within input_tokens; reasoning already
                # within output_tokens. Never add their details a second time.
                record['usage']=usage; d['settled_requests']+=1; state='settled'
            record.update(ended=time.time(),error=error)
            self._save(db,row['account'],d)
            db.execute('UPDATE requests SET state=?,data=? WHERE id=?',(state,_json(record),request_id))

    def _open_endpoint(self,body,operation):
        base=self.config['base_url'].rstrip('/')
        endpoint=base+'/responses' if base.endswith('/v1') else base+'/v1/responses'
        if operation=='input_tokens': endpoint+='/input_tokens'
        request=urllib.request.Request(endpoint,data=_json(body).encode(),headers={
            'Authorization':'Bearer '+self.config['api_key'],'Content-Type':'application/json',
            'Accept':'text/event-stream' if body.get('stream') and operation=='generate' else 'application/json'})
        # No redirects, retries, credential forwarding or alternate endpoints.
        trace_id=getattr(self.diagnostic_context,'trace_id',None)
        if trace_id is None: trace_id=self._begin_diagnostic(body)
        self._diagnostic(trace_id,operation+'_payload.json',body)
        try:
            deadline=getattr(self.diagnostic_context,'deadline',None)
            timeout=deadline.remaining() if deadline else 120
            handlers=(_NoRedirect(),)+(_deadline_handlers(deadline) if deadline else ())
            response=urllib.request.build_opener(*handlers).open(request,timeout=timeout)
            if deadline: deadline.remaining()
            self._diagnostic(trace_id,operation+'_status.json',dict(status=response.status,time=time.time()))
            return response
        except urllib.error.HTTPError as exc:
            # Diagnostic bodies are bounded and kept solely on the trusted side.
            raw=exc.read(1024*1024+1)
            self._diagnostic(trace_id,operation+'_error.json',dict(
                status=exc.code,body=raw[:1024*1024].decode('utf-8',errors='replace'),
                truncated=len(raw)>1024*1024,time=time.time()))
            raise
        except (OSError,urllib.error.URLError,GatewayError) as exc:
            self._diagnostic(trace_id,operation+'_error.json',dict(
                kind=type(exc).__name__,detail=str(exc),time=time.time()))
            raise

    def open_upstream(self,body): return self._open_endpoint(body,'generate')


def make_server(gateway,host='127.0.0.1',port=0,unix_socket=None,request_timeout_seconds=120):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send_json(self,status,value):
            payload=_json(value).encode(); self.send_response(status)
            self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        def do_POST(self):
            request_id=None; usage=None; headers_sent=False; error=None
            deadline=_Deadline(gateway,request_timeout_seconds)
            gateway.diagnostic_context.deadline=deadline
            deadline.track(self.connection)
            try:
                if self.path not in ('/v1/responses','/v1/responses/input_tokens'): raise GatewayError('endpoint_forbidden',404)
                self.connection.settimeout(deadline.remaining())
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16*1024*1024: raise GatewayError('invalid_body_size')
                body=json.loads(self.rfile.read(size))
                deadline.remaining()
                auth=self.headers.get('Authorization',''); token=auth[7:] if auth.startswith('Bearer ') else ''
                if self.path.endswith('/input_tokens'):
                    self.send_json(200,gateway.count_input(token,body)); return
                request_id,body=gateway.reserve(token,body)
                deadline.remaining()
                with gateway.open_upstream(body) as upstream:
                    deadline.remaining()
                    content_type=upstream.headers.get('Content-Type','application/json')
                    if body.get('stream') and 'text/event-stream' in content_type:
                        client_connected=True
                        def disconnected():
                            nonlocal client_connected
                            client_connected=False
                            gateway._diagnostic(request_id,'client_disconnect.json',dict(time=time.time(),
                                action='drain_existing_generation_until_terminal_or_original_deadline'))
                        headers_sent=True; self.close_connection=True
                        try:
                            self.send_response(200); self.send_header('Content-Type','text/event-stream')
                            self.send_header('Cache-Control','no-cache'); self.send_header('Connection','close')
                            self.end_headers()
                        except OSError:
                            disconnected()
                        event_lines=[]
                        def emit_event(lines):
                            nonlocal usage
                            deadline.remaining()
                            data=b'\n'.join(line[5:].strip() for line in lines if line.startswith(b'data:'))
                            terminal=False
                            if data and data!=b'[DONE]':
                                event=json.loads(data)
                                terminal=event.get('type') in ('response.completed','response.incomplete')
                                if terminal:
                                    usage=event.get('response',{}).get('usage')
                                    # Release the durable in-flight reservation BEFORE
                                    # exposing completion. A native client can submit
                                    # its next generation as soon as it sees this event.
                                    gateway.settle(request_id,usage)
                            if client_connected:
                                try:
                                    self.wfile.write(b''.join(lines)); self.wfile.flush()
                                except OSError:
                                    # A killed local controller does not cancel the
                                    # upstream charge. Drain this SAME request to
                                    # settle usage, without retrying or extending its
                                    # original count+generation deadline.
                                    disconnected()
                            return terminal
                        for line in upstream:
                            deadline.remaining()
                            event_lines.append(line)
                            if not line.strip():
                                terminal=emit_event(event_lines); event_lines=[]
                                if terminal:
                                    # A complete terminal event is the accounting
                                    # boundary; do not wait for a stalled upstream EOF.
                                    break
                        if event_lines: emit_event(event_lines)
                    else:
                        result=json.loads(upstream.read()); deadline.remaining(); usage=result.get('usage')
                        gateway.settle(request_id,usage)
                        self.send_json(200,result); headers_sent=True
            except GatewayError as exc:
                error=exc.code
                if not headers_sent:
                    try: self.send_json(exc.status,dict(error=dict(type='benchmark_gateway',code=exc.code,message=exc.code)))
                    except OSError: pass
            except (ValueError,TypeError) as exc:
                error='request_timeout' if deadline.expired else ('invalid_upstream_response' if request_id else 'invalid_request')
                if not headers_sent:
                    try: self.send_json(504 if deadline.expired else (502 if request_id else 400),dict(error=dict(code=error,message=error)))
                    except OSError: pass
            except (OSError,urllib.error.URLError):
                error='request_timeout' if deadline.expired else 'upstream_or_transport_failure'
                if not headers_sent:
                    try: self.send_json(504 if deadline.expired else 502,dict(error=dict(code=error,message=error)))
                    except OSError: pass
            finally:
                deadline.finish()
                if request_id: gateway.settle(request_id,usage,'request_timeout' if deadline.expired else error)
                gateway.diagnostic_context.deadline=None
    if unix_socket is not None:
        class UnixServer(socketserver.ThreadingMixIn,socketserver.UnixStreamServer): daemon_threads=True
        return UnixServer(str(unix_socket),Handler)
    return ThreadingHTTPServer((host,port),Handler)
