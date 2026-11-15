"""Copied read-only into /runtime; executed ONLY by namespace PID 1 (Python 3.8)."""
import http.client
import json
import os
import signal
import socket
import subprocess
import threading
import time


class Connection(http.client.HTTPConnection):
    def connect(self):
        self.sock=socket.socket(socket.AF_UNIX)
        self.sock.settimeout(120)
        self.sock.connect(os.environ['PILOT_CONTROL_SOCKET'])


def request(method, **fields):
    connection=Connection('localhost',timeout=120)
    try:
        connection.request('POST','/control',json.dumps(dict(method=method,**fields)),
                           {'Authorization':'Bearer '+os.environ['PILOT_CONTROL_TOKEN'],'Content-Type':'application/json'})
        value=json.loads(connection.getresponse().read())
        if not value.get('ok'):raise RuntimeError(value.get('error','broker_error'))
        return value
    finally:connection.close()


def run(config):
    # The external supervisor publishes the outer PID immediately after Popen;
    # do not launch a controller until the kernel peer identity is accepted.
    for attempt in range(100):
        try:request('broker_poll');break
        except RuntimeError:time.sleep(.05)
    else:raise RuntimeError('broker_identity_not_accepted')
    controller=subprocess.Popen(config['argv'],close_fds=True,start_new_session=True)
    children={controller.pid:('controller',controller,[])}
    controller_done=False; code=0; stopping=False
    io_errors=[]

    def forward(pipe, job_id, stream):
        try:
            while True:
                data=os.read(pipe.fileno(),4096)
                if not data:break
                request('broker_event',kind='job_output',job_id=job_id,stream=stream,text=data.decode('utf-8',errors='replace'))
        except Exception as error:io_errors.append(type(error).__name__)
        finally:pipe.close()

    while True:
        polled=request('broker_poll')
        if polled.get('stop_model') and not controller_done and not stopping:
            stopping=True
            try:os.killpg(controller.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for job in polled['jobs']:
            try:
                process=subprocess.Popen(job['argv'],cwd='/work',close_fds=True,start_new_session=True,
                                         stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                request('broker_event',kind='job_started',job_id=job['job_id'],pid=process.pid)
                readers=[]
                for pipe,stream in [(process.stdout,'stdout'),(process.stderr,'stderr')]:
                    thread=threading.Thread(target=forward,args=(pipe,job['job_id'],stream),daemon=True)
                    thread.start();readers.append(thread)
                children[process.pid]=(job['job_id'],process,readers)
            except OSError as error:
                request('broker_event',kind='job_exit',job_id=job['job_id'],exit_code=127,signal=None)
        no_children=False
        while True:
            try:pid,status,usage=os.wait4(-1,os.WNOHANG)
            except ChildProcessError:no_children=True;break
            if pid==0:break
            value=os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status)
            record=children.pop(pid,None)
            if record is None:continue  # adopted unmanaged descendant, still metered/reaped
            name,process,readers=record;process.returncode=value
            if name=='controller':
                controller_done=True;code=value
                request('broker_event',kind='controller_exit',exit_code=value)
            else:
                for thread in readers:thread.join()
                request('broker_event',kind='job_exit',job_id=name,exit_code=value,signal=-value if value<0 else None)
        # Admission is sealed by controller_exit. Poll again on the next turn
        # before exiting, so jobs admitted just before that event are dispatched.
        if controller_done and no_children and not polled['jobs'] and not request('broker_pending')['active']:
            if io_errors:raise RuntimeError('job_log_transport_failed')
            return code if code>=0 else 128-code
        time.sleep(.05)
