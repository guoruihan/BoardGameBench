"""Trusted Harmonies pilot boundary. Never mount this module on agent workers.

The persisted ledger survives restart; live games deliberately fail closed as
infrastructure_interrupted after restart. It is not a snapshot/branch API.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import secrets
import socketserver
import sqlite3
import threading
import time

from .joint_pool import verify_pool, load_opponent
from .registry import game_class
from .evaluate import code_identity

DOMAINS = ('smoke', 'development', 'calibration', 'formal')
CONFIG = 'side_a_base32_no_spirits_duo'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def private_seed(key, domain, pair_index):
    """Disjoint 62-bit domains; profile/control/run never affect paired seeds."""
    if domain not in DOMAINS or type(pair_index) is not int or pair_index < 0:
        raise ValueError('Invalid trusted seed assignment')
    bits = hmac.new(bytes.fromhex(key), str(pair_index).encode(), hashlib.sha256).digest()
    return (DOMAINS.index(domain) << 62) | (int.from_bytes(bits[:8], 'big') & ((1 << 62)-1))


@contextmanager
def connect(root):
    db = sqlite3.connect(str(Path(root)/'ledger.sqlite'), timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA synchronous=FULL')
    try:
        with db:
            yield db
    finally:
        db.close()


def initialize(root, pool, opponent='legacy', *, seed_key=None):
    if seed_key is not None and (not isinstance(seed_key,str) or len(seed_key)!=64
                                 or any(c not in '0123456789abcdef' for c in seed_key)):
        raise ValueError('Invalid trusted seed key')
    manifest = verify_pool(pool)
    game = manifest['games']['harmonies']
    member = game['members'][opponent]
    if game['config'] != CONFIG or member['spec']['kind'] != 'harmonies_rl':
        raise ValueError('Pilot requires fixed Harmonies neural opponent')
    root = Path(root); root.mkdir(mode=0o700, parents=True, exist_ok=False)
    config = dict(pool=str(Path(pool).resolve()), pool_sha256=manifest['manifest_sha256'],
                  opponent=opponent, policy_id=member['policy_id'], weights_sha256=member['spec']['sha256'],
                  config_id=CONFIG, rules_version=game['rules_version'], seed_key=seed_key or secrets.token_hex(32),
                  environment_source_sha256=code_identity()['content_sha256'],
                  inference=dict(device='cpu',policy='frozen_pool_loader',threads=1))
    (root/'private.json').write_text(json.dumps(config)); (root/'private.json').chmod(0o600)
    with connect(root) as db:
        db.executescript('''CREATE TABLE runs(id TEXT PRIMARY KEY, token_hash TEXT UNIQUE, data TEXT);
        CREATE TABLE requests(run TEXT,id TEXT,body_hash TEXT,response TEXT,PRIMARY KEY(run,id));
        CREATE TABLE events(run TEXT, seq INTEGER, data TEXT, PRIMARY KEY(run,seq));''')
    (root/'ledger.sqlite').chmod(0o600)
    return {k:v for k,v in config.items() if k not in ('pool','seed_key')}


def create_run(root, phase='development', pair_index=0, seat=0, budget=None):
    if phase not in DOMAINS or type(seat) is not int or seat not in (0,1) or type(pair_index) is not int or pair_index < 0:
        raise ValueError('Invalid trusted schedule')
    game_limit=(budget or {}).get('max_gt_games',10 if phase=='development' else 1)
    if type(game_limit) is not int or game_limit<1 or (phase!='development' and game_limit!=1):
        raise ValueError('Invalid trusted GT game limit')
    run_id=secrets.token_hex(12); token=secrets.token_urlsafe(32)
    data=dict(gt_game_limit=game_limit,phase=phase,pair_index=pair_index,seat=seat,budget=budget or {},generation=1,
              source='agent',games=0,status='ready',version=0,steps=0,errors=0,decision_errors=0,
              sources={},outcomes=[],closed=False,service_wall_seconds=0.,opponent_seconds=0.)
    with connect(root) as db:
        db.execute('INSERT INTO runs VALUES(?,?,?)',(run_id,digest(token),json.dumps(data)))
    return dict(run_id=run_id,token=token,generation=1)


class Rejected(Exception): pass


class PilotService:
    def __init__(self, root):
        self.root=Path(root); self.config=json.loads((self.root/'private.json').read_text())
        self.lock=(self.root/'worker.lock').open('a')
        try: fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException: self.lock.close(); raise
        self.mutex=threading.RLock(); self.live={}
        import torch
        torch.set_num_threads(1)
        try:
            self.manifest=verify_pool(self.config['pool'])
            if code_identity()['content_sha256'] != self.config['environment_source_sha256']:
                raise ValueError('Environment source changed; use frozen release')
            if self.manifest['manifest_sha256'] != self.config['pool_sha256']: raise ValueError('Pool changed')
            with connect(root) as db:
                for row in db.execute('SELECT * FROM runs').fetchall():
                    d=json.loads(row['data'])
                    if d['status']=='active':
                        d['status']='infrastructure_interrupted'; d['closed']=True
                        d['outcomes'].append(dict(status=d['status'],natural=False,steps=d['steps']))
                        self._save(db,row['id'],d)
                db.execute("UPDATE requests SET response=? WHERE response IS NULL",(json.dumps(dict(ok=False,error='interrupted_request')),))
        except BaseException: self.lock.close(); raise

    def close(self):
        with self.mutex: self.live.clear(); self.lock.close()

    def _save(self,db,rid,d):
        db.execute('UPDATE runs SET data=? WHERE id=?',(json.dumps(d),rid)); db.commit()

    def _event(self,db,rid,event):
        seq=db.execute('SELECT count(*) FROM events WHERE run=?',(rid,)).fetchone()[0]
        db.execute('INSERT INTO events VALUES(?,?,?)',(rid,seq,json.dumps(event)))

    def _public(self,d):
        return {k:d[k] for k in ('phase','generation','games','status','version','steps','errors','sources','budget','closed','outcomes')}

    def report(self,run_id):
        with self.mutex,connect(self.root) as db:
            row=db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
            if not row: raise ValueError('Unknown run')
            return dict(run_id=run_id,**json.loads(row['data']))

    def update_budget(self,run_id,budget):
        """Trusted runner publishes metered balances; agents cannot edit them."""
        with self.mutex,connect(self.root) as db:
            row=db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
            if not row: raise ValueError('Unknown run')
            d=json.loads(row['data']); d['budget']=budget; self._save(db,run_id,d)

    def rotate_controller(self,run_id,source='program'):
        if source not in ('agent','program'): raise ValueError('Invalid source')
        with self.mutex,connect(self.root) as db:
            row=db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
            if not row: raise ValueError('Unknown run')
            d=json.loads(row['data']); d['generation']+=1; d['source']=source
            token=secrets.token_urlsafe(32)
            db.execute('UPDATE runs SET token_hash=? WHERE id=?',(digest(token),run_id)); self._save(db,run_id,d)
            return dict(run_id=run_id,token=token,generation=d['generation'])

    def _end(self,db,rid,d,status):
        result=dict(status=status,natural=status=='complete',steps=d['steps'],sources=dict(d['sources']),errors=d['errors'])
        if status=='complete':
            outcome=self.live[rid]['env'].outcome(); seat=self.live[rid]['seat']
            winners=outcome['winners']
            result.update(own_score=outcome['scores'][seat],opponent_score=outcome['scores'][1-seat],
                          win_points=.5 if len(winners)!=1 else float(seat in winners))
        elif status in ('cpu_exhausted','wall_exhausted','agent_failure','memory_exhausted'):
            result['win_points']=0.
        d['status']=status; d['outcomes'].append(result); self.live.pop(rid,None)
        self._save(db,rid,d)

    def stop(self,run_id,reason='wall_exhausted'):
        with self.mutex,connect(self.root) as db:
            row=db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
            if not row: raise ValueError('Unknown run')
            d=json.loads(row['data']); d['closed']=True; d['generation']+=1
            if run_id in self.live: self._end(db,run_id,d,reason)
            self._save(db,run_id,d)
            return self._public(d)

    @staticmethod
    def _observation(env,seat):
        observation=env.observe(seat)
        observation.pop('legal_actions',None)  # Read the versioned paginated endpoint.
        return observation

    def _step(self,db,rid,d,action,source):
        ep=self.live[rid]; env=ep['env']; player=env.current_player
        env.step(action); d['version']+=1; d['steps']+=1
        d['sources'][source]=d['sources'].get(source,0)+1
        self._event(db,rid,dict(game=d['games'],version=d['version'],player=player,source=source,
                               action=action,observation=self._observation(env,ep['seat'])))
        self._save(db,rid,d)

    def _advance(self,db,rid,d):
        while rid in self.live:
            ep=self.live[rid]; env=ep['env']; outcome=env.outcome()
            if outcome.get('unresolved') or outcome.get('unresolved_rule'):
                self._end(db,rid,d,'unresolved_rule'); break
            if outcome['terminated']: self._end(db,rid,d,'complete'); break
            if d['steps']>=3000: self._end(db,rid,d,'step_cap'); break
            legal=env.legal_actions()
            if not legal: self._end(db,rid,d,'no_legal_action'); break
            if env.current_player!=ep['seat']:
                start=time.monotonic(); action=ep['bot'].decide(env.observe(env.current_player),json.loads(json.dumps(legal)))
                d['opponent_seconds']+=time.monotonic()-start; source='opponent'
            elif len(legal)==1: action=legal[0]; source='forced'
            elif d['errors']>=30: action=ep['rng'].choice(legal); source='default_random'
            else: break
            self._step(db,rid,d,action,source)

    def fallback(self,run_id,reason='no_controller',finish=False):
        """Trusted runner only; every call revokes the previous controller lease."""
        with self.mutex,connect(self.root) as db:
            row=db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
            if not row: raise ValueError('Unknown run')
            d=json.loads(row['data']); d['generation']+=1
            while run_id in self.live:
                ep=self.live[run_id]
                self._step(db,run_id,d,ep['rng'].choice(ep['env'].legal_actions()),'default_random')
                d['decision_errors']=0; self._advance(db,run_id,d)
                if not finish: break
            self._event(db,run_id,dict(event='fallback',reason=reason))
            self._save(db,run_id,d); return self._public(d)

    def request(self,token,request_id,body):
        if not isinstance(token,str) or not isinstance(request_id,str) or not 1<=len(request_id)<=128:
            return dict(ok=False,error='invalid_request')
        try: body_hash=digest(body)
        except (ValueError,TypeError): return dict(ok=False,error='invalid_request')
        with self.mutex,connect(self.root) as db:
            row=db.execute('SELECT * FROM runs WHERE token_hash=?',(digest(token),)).fetchone()
            if not row: return dict(ok=False,error='unauthorized')
            rid=row['id']; d=json.loads(row['data'])
            old=db.execute('SELECT * FROM requests WHERE run=? AND id=?',(rid,request_id)).fetchone()
            if old:
                return (json.loads(old['response']) if old['response'] else dict(ok=False,error='interrupted_request')) if old['body_hash']==body_hash else dict(ok=False,error='request_id_conflict')
            db.execute('INSERT INTO requests VALUES(?,?,?,NULL)',(rid,request_id,body_hash)); db.commit()
            start=time.monotonic()
            try:
                if not isinstance(body,dict): raise Rejected('invalid_request')
                method=body.get('method')
                if method in ('new_game','act','abort'):
                    if d['closed']: raise Rejected('run_closed')
                    if type(body.get('generation')) is not int or body['generation']!=d['generation']: raise Rejected('controller_revoked')
                if method=='new_game':
                    if set(body)-{'method','generation'}: raise Rejected('fixed_schedule')
                    if rid in self.live: raise Rejected('active_game_limit')
                    if d['games'] >= d.get('gt_game_limit',10 if d['phase']=='development' else 1): raise Rejected('game_limit')
                    index=d['games'] if d['phase']=='development' else d['pair_index']
                    seat=d['games']%2 if d['phase']=='development' else d['seat']
                    seed=private_seed(self.config['seed_key'],d['phase'],index)
                    d.update(games=d['games']+1,status='active',steps=0,errors=0,decision_errors=0,sources={})
                    self._save(db,rid,d)
                    env=game_class('harmonies')(); env.reset(seed,first_player=0)
                    self.live[rid]=dict(env=env,seat=seat,rng=random.Random(seed^731),
                                       bot=load_opponent(self.config['pool'],'harmonies',self.config['opponent'],1-seat,seed,self.manifest))
                    self._advance(db,rid,d); result=self._public(d)
                elif method in ('budget','status'): result=self._public(d)
                elif method=='history':
                    offset,limit=self._page(body)
                    rows=db.execute('SELECT data FROM events WHERE run=? ORDER BY seq LIMIT ? OFFSET ?',(rid,limit,offset)).fetchall()
                    total=db.execute('SELECT count(*) FROM events WHERE run=?',(rid,)).fetchone()[0]
                    result=dict(items=[json.loads(r[0]) for r in rows],offset=offset,total=total)
                elif method in ('observe','legal_actions','act','abort'):
                    if rid not in self.live: raise Rejected('game_not_active')
                    ep=self.live[rid]; env=ep['env']
                    if method=='observe': result=dict(**self._public(d),seat=ep['seat'],observation=self._observation(env,ep['seat']))
                    elif method=='legal_actions':
                        offset,limit=self._page(body); legal=env.legal_actions()
                        result=dict(version=d['version'],items=[dict(action_index=i,action=legal[i]) for i in range(offset,min(offset+limit,len(legal)))],offset=offset,total=len(legal))
                    elif method=='abort': self._end(db,rid,d,'aborted'); result=self._public(d)
                    else:
                        legal=env.legal_actions(); index=body.get('action_index')
                        error=None
                        if type(body.get('version')) is not int or body['version']!=d['version']: error='stale_observation'
                        elif type(index) is not int or not 0<=index<len(legal): error='invalid_action'
                        if error:
                            d['errors']+=1; d['decision_errors']+=1
                            if d['decision_errors']>=3 or d['errors']>=30:
                                self._step(db,rid,d,ep['rng'].choice(legal),'default_random'); d['decision_errors']=0
                                self._advance(db,rid,d)
                            raise Rejected(error)
                        self._step(db,rid,d,legal[index],d['source']); d['decision_errors']=0
                        self._advance(db,rid,d); result=self._public(d)
                else: raise Rejected('unknown_method')
                response=dict(ok=True,**result)
            except Rejected as exc: response=dict(ok=False,error=str(exc))
            except Exception as exc:
                if rid in self.live: self._end(db,rid,d,'infrastructure_error')
                else: d['status']='infrastructure_error'; d['closed']=True
                with (self.root/'errors.jsonl').open('a') as f: f.write(json.dumps(dict(run=rid,error=repr(exc)))+'\n')
                response=dict(ok=False,error='infrastructure_error')
            d['service_wall_seconds']+=time.monotonic()-start; self._save(db,rid,d)
            db.execute('UPDATE requests SET response=? WHERE run=? AND id=?',(json.dumps(response),rid,request_id))
            return response

    @staticmethod
    def _page(body):
        offset=body.get('offset',0); limit=body.get('limit',64)
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=256: raise Rejected('invalid_page')
        return offset,limit


def make_server(service,host='127.0.0.1',port=0,unix_socket=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            response=dict(ok=False,error='invalid_request')
            try:
                self.connection.settimeout(10)
                size=int(self.headers.get('Content-Length','0'))
                if self.path=='/v1/rpc' and 0<size<=65536:
                    auth=self.headers.get('Authorization','')
                    response=service.request(auth[7:] if auth.startswith('Bearer ') else '',self.headers.get('Idempotency-Key',''),json.loads(self.rfile.read(size)))
            except (ValueError,TimeoutError): pass
            payload=json.dumps(response).encode(); self.send_response(200 if response['ok'] else 400)
            self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
    if unix_socket is not None:
        class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads=True
        return UnixHTTPServer(str(unix_socket),Handler)
    return ThreadingHTTPServer((host,port),Handler)
