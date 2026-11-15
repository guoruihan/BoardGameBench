"""Standalone stdlib-only public SDK. Contains no engine or opponent imports."""
import http.client
import json
import os
import socket
import uuid
from urllib.parse import urlsplit


class PilotError(RuntimeError):
    def __init__(self, response):
        self.response=response
        super().__init__(response.get('error','request_failed'))


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self,path,timeout):
        super().__init__('localhost',timeout=timeout); self.path=path
    def connect(self):
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout); self.sock.connect(self.path)


class PilotClient:
    def __init__(self, url=None, token=None, generation=None, unix_socket=None, timeout=120):
        self.url=url or os.environ.get('PILOT_GT_URL','http://localhost')
        self.token=token or os.environ['PILOT_GT_TOKEN']
        self.generation=int(generation if generation is not None else os.environ.get('PILOT_GT_GENERATION','1'))
        self.unix_socket=unix_socket or os.environ.get('PILOT_GT_SOCKET'); self.timeout=timeout

    def request(self,method,request_id=None,**kwargs):
        body=dict(method=method,**kwargs)
        if method in ('new_game','act','abort'): body['generation']=self.generation
        target=urlsplit(self.url)
        if self.unix_socket: connection=_UnixConnection(self.unix_socket,self.timeout)
        elif target.scheme=='https': connection=http.client.HTTPSConnection(target.hostname,target.port,timeout=self.timeout)
        elif target.scheme=='http': connection=http.client.HTTPConnection(target.hostname,target.port,timeout=self.timeout)
        else: raise ValueError('Unsupported service URL')
        try:
            connection.request('POST','/v1/rpc',json.dumps(body),headers={
                'Authorization':'Bearer '+self.token,'Idempotency-Key':request_id or uuid.uuid4().hex,
                'Content-Type':'application/json'})
            response=json.loads(connection.getresponse().read())
        finally: connection.close()
        if not response.get('ok'): raise PilotError(response)
        return response

    def new_game(self,request_id=None): return self.request('new_game',request_id)
    def observe(self): return self.request('observe')
    def legal_actions(self,offset=0,limit=64): return self.request('legal_actions',offset=offset,limit=limit)
    def history(self,offset=0,limit=64): return self.request('history',offset=offset,limit=limit)
    def budget(self): return self.request('budget')
    def status(self): return self.request('status')
    def act(self,action_index,version,request_id=None): return self.request('act',request_id,action_index=action_index,version=version)
    def abort(self,request_id=None): return self.request('abort',request_id)

    def run(self,policy,max_actions=None):
        """Run a user policy over fresh observation/pages; None means NEED_AGENT.

        This loop executes in the untrusted worker and is metered there. Source
        labels come from trusted controller registration, never a client field.
        """
        count=0
        while max_actions is None or count<max_actions:
            status=self.status()
            if status['status']!='active': return status
            observation=self.observe(); actions=[]; offset=0
            while True:
                page=self.legal_actions(offset=offset)
                if page['version']!=observation['version']: raise PilotError(dict(error='stale_observation'))
                actions.extend(page['items']); offset+=len(page['items'])
                if offset>=page['total']: break
            selected=policy(observation,actions)
            if selected is None or selected=='NEED_AGENT': return dict(status='NEED_AGENT',actions=count)
            result=self.act(selected,observation['version']); count+=1
            if result['status']!='active': return result
        return dict(status='action_limit',actions=count)
