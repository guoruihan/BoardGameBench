"""Resident CPU search with parent-owned per-call deadlines and scoped suspension.

Parent/child share one pinned CPU. The child is stopped between calls, including
when native code ignores its cooperative deadline. Buffered framing survives a
timeout halfway through a message. OS scheduling excess is still measured.
Only this locally created child is signaled; no external process is touched.
"""
import base64
import json
import multiprocessing as mp
import os
import pickle
import select
import signal
import socket
import time

from .search import DecisionDeadline


class Channel:
    def __init__(self,sock):
        self.sock=sock;sock.setblocking(False);self.incoming=bytearray();self.outgoing=bytearray()

    def send(self,message,deadline):
        self.outgoing.extend(json.dumps(message,allow_nan=False,separators=(',',':')).encode()+b'\n')
        while self.outgoing:
            if time.monotonic()>=deadline:raise TimeoutError('send deadline')
            if not select.select([], [self.sock], [], max(0,deadline-time.monotonic()))[1]:
                raise TimeoutError('send deadline')
            n=self.sock.send(self.outgoing)
            if not n:raise EOFError()
            del self.outgoing[:n]

    def receive(self,deadline):
        while True:
            if time.monotonic()>=deadline:raise TimeoutError('receive deadline')
            pos=self.incoming.find(b'\n')
            if pos>=0:
                raw=bytes(self.incoming[:pos]);del self.incoming[:pos+1]
                return json.loads(raw)
            if not select.select([self.sock],[],[],max(0,deadline-time.monotonic()))[0]:
                raise TimeoutError('receive deadline')
            chunk=self.sock.recv(65536)
            if not chunk:raise EOFError()
            self.incoming.extend(chunk)
            if len(self.incoming)>4*1024*1024:raise ValueError('IPC message too large')


def pack(plans):
    # Trusted local process memory only; never deserialize a user/network file.
    return base64.b64encode(pickle.dumps(plans,protocol=4)).decode()


def unpack(value):
    return pickle.loads(base64.b64decode(value))


def worker(sock,search):
    channel=Channel(sock)
    channel.send({'kind':'ready'},time.monotonic()+30)
    while True:
        request=channel.receive(time.monotonic()+86400)
        if request['deadline']<=time.monotonic():continue
        plans=unpack(request['plans'])
        def update(action,info):
            try:
                channel.send(dict(kind='progress',id=request['id'],action=action,info=info,plans=pack(plans)),request['deadline'])
            except TimeoutError:raise DecisionDeadline()
        search.on_update=update
        search.seconds=max(.001,min(request['seconds'],request['deadline']-time.monotonic()))
        try:
            action=search.decide(request['observation'],plans)
            channel.send(dict(kind='action',id=request['id'],action=action,info=search.last,plans=pack(plans)),request['deadline'])
        except TimeoutError:
            pass  # Partial outgoing frames remain queued and carry a stale id.


class TimedSearch:
    def __init__(self,search):
        self.seconds=search.seconds;self.spent=0.;self.counter=0;self.last={}
        parent,child=socket.socketpair();self.channel=Channel(parent)
        self.process=mp.get_context('fork').Process(target=worker,args=(child,search),daemon=True)
        self.process.start();child.close()
        if self.channel.receive(time.monotonic()+30)['kind']!='ready':raise RuntimeError('search startup failed')
        os.kill(self.process.pid,signal.SIGSTOP)

    def decide(self,obs,plans):
        begin=time.monotonic();deadline=begin+max(.001,self.seconds-.006);self.counter+=1
        action=dict(obs['legal_actions'][0]);info={};finished=False
        if not self.process.is_alive():raise RuntimeError('resident search exited')
        try:
            os.kill(self.process.pid,signal.SIGCONT)
            self.channel.send(dict(id=self.counter,observation=obs,plans=pack(plans),
                                   seconds=self.seconds,deadline=deadline),deadline)
            while True:
                message=self.channel.receive(deadline)
                if message.get('id')!=self.counter:continue
                if message['action'] not in obs['legal_actions']:raise RuntimeError('illegal resident action')
                action=message['action'];info=message['info'];plans[:]=unpack(message['plans'])
                if message['kind']=='action':finished=True;break
        except TimeoutError:
            pass
        finally:
            if self.process.is_alive():os.kill(self.process.pid,signal.SIGSTOP)
        elapsed=time.monotonic()-begin;self.spent+=elapsed
        self.last={**dict(completed_rounds=0,simulation_steps=0,candidates=[],fallback=True),**info,
            'budget_scope':'per_actual_atomic_decision','budget_seconds':self.seconds,
            'decision_seconds':elapsed,'cumulative_decision_seconds':self.spent,
            'budget_excess_seconds':max(0.,elapsed-self.seconds),'resident_timeout':not finished,
            'forced_action':len(obs['legal_actions'])==1}
        return action

    def close(self):
        if self.process.is_alive():os.kill(self.process.pid,signal.SIGKILL)
        self.process.join(timeout=5);self.channel.sock.close()
