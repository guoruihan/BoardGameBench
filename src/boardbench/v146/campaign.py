"""Frozen CPU-only teacher gate, conditional expansion and train-only trajectories."""
import argparse
from collections import deque
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from boardbench.artifacts.store import digest,read_json,write_json,source_files,source_id
from boardbench.v12.budget import kill_tree
from boardbench.v122.campaign import physical_cpus
from .report import summarize,pilot_choices,teacher_choice


def valid_result(data,job):
    for key in ('id','method','split','environment_seed','seat','search_seed','source_id','student_sha256','opponent_sha256'):
        if data[key]!=job[key]:raise ValueError('completed game metadata mismatch: '+key)
    if not data['natural_terminal'] or data['status']!='completed' or data['budget_excess_seconds']>.1:
        raise ValueError('incomplete/over-budget game')
    if digest(data['trajectory'])!=data['trajectory_sha256']:
        raise ValueError('completed trajectory digest mismatch')
    return data


class Runner:
    WORKER_MODULE='boardbench.v146.worker'
    def __init__(self,root):
        self.root=Path(root);self.config=read_json(self.root/'config.json');self.launch=read_json(self.root/'launch.json')
        c=self.config
        if (source_id()!=self.launch['source_id'] or digest(self.root/'config.json')!=self.launch['config_sha256']
                or os.uname().nodename!=c['host'] or time.time()>=c['deadline_unix']):
            raise ValueError('frozen source/host/config/original deadline mismatch')
        physical=physical_cpus();self.cpus=self.launch['cpus']
        if not set(self.cpus)<=set(physical) or len(self.cpus)>len(physical)//2:
            raise ValueError('per-machine half-physical-CPU cap')
        lock_path=Path(c['repo'])/'outputs/v122'/(c['host']+'.cpu.lock')
        lock_path.parent.mkdir(parents=True,exist_ok=True);self.lock=lock_path.open('a+')
        fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.sched_setaffinity(0,{self.cpus[-1]});self.active=[];self.stop=False;self.phase='starting'
        self.receipts=read_json(self.root/'cost.json').get('attempts',[]) if (self.root/'cost.json').exists() else []
        self.controller_started=time.time();self.prior_controller=read_json(self.root/'cost.json').get('controller_seconds',0) if (self.root/'cost.json').exists() else 0.
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,self.request_stop)

    def request_stop(self,*_):self.stop=True

    def report(self,queued=0):
        now=time.time();controller=self.prior_controller+now-self.controller_started
        write_json(self.root/'status.json',dict(phase=self.phase,pid=os.getpid(),updated_unix=now,queued=queued,
            deadline_unix=self.config['deadline_unix'],remaining_seconds=max(0,self.config['deadline_unix']-now),
            active=[dict(id=r['id'],pid=p.pid,cpu=r['cpu'],started_unix=r['started_unix']) for p,_,r,_ in self.active]))
        write_json(self.root/'cost.json',dict(attempts=self.receipts,controller_seconds=controller,
            allocated_cpu_core_seconds=controller+sum(r.get('elapsed_seconds',now-r['started_unix']) for r in self.receipts),
            gpu_seconds=0,previous_run=self.config['previous_run'],original_deadline_unix=self.config['deadline_unix'],
            note='Single physical CPU per child; controller counted; previous V145/setup cost retained separately.'))

    def jobs(self,methods,seeds,stage):
        c=self.config;result=[]
        for seed in seeds:
            for seat in (0,1):
                for method in methods:
                    result.append(dict(id=f'{stage}_{method}_e{seed}_s{seat}',stage=stage,method=method,
                        environment_seed=seed,seat=seat,search_seed=c['search_seed'],
                        split='train' if stage=='expert' else 'development',seconds=c['seconds'],
                        deadline_unix=c['deadline_unix']-15,source_id=self.launch['source_id'],
                        **{k:c[k] for k in ('student','student_sha256','opponent','opponent_sha256','native_artifact')}))
        return result

    def run(self,jobs,stage):
        self.phase=stage;queue=deque();outputs=[];available=deque(self.cpus[:-1])
        for j in jobs:
            result=self.root/'episodes'/j['id']/'result.json'
            if result.exists():
                outputs.append(valid_result(read_json(result),j))
            else:queue.append(j)
        try:
            while queue or self.active:
                now=time.time()
                if self.stop or now>=self.config['deadline_unix']-30:
                    raise InterruptedError('stop/deadline; no budget renewal')
                # Leave enough for a complete900s focal game plus opponent/replay.
                while queue and available:
                    j=queue[0];needed=self.config['job_timeout_seconds'] if j['method'].startswith('search_') else 120
                    if self.config['deadline_unix']-time.time()<needed+30:
                        raise TimeoutError('insufficient remaining window for complete next game')
                    queue.popleft();cpu=available.popleft();target=self.root/'episodes'/j['id']
                    if target.exists():target.rename(target.with_name(target.name+f'.incomplete_{time.time_ns()}'))
                    path=self.root/'jobs'/(j['id']+'.json');write_json(path,j)
                    log=(self.root/'logs'/(j['id']+'.log')).open('a')
                    env={**os.environ,'PYTHONPATH':str(self.root/'source/src'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
                         'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
                    p=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-u','-m',self.WORKER_MODULE,
                        '--job',str(path),'--out',str(target)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                    r=dict(id=j['id'],cpu=cpu,pid=p.pid,started_unix=time.time(),timeout_seconds=needed)
                    self.receipts.append(r);self.active.append((p,log,r,j))
                failed=[]
                for item in list(self.active):
                    p,log,r,j=item
                    if p.poll() is None and time.time()-r['started_unix']>r['timeout_seconds']:
                        kill_tree(p.pid);r['timed_out']=True
                    if p.poll() is None:continue
                    p.wait();log.close();r.update(returncode=p.returncode,elapsed_seconds=time.time()-r['started_unix'])
                    available.append(r['cpu']);self.active.remove(item)
                    result=self.root/'episodes'/j['id']/'result.json'
                    if p.returncode or not result.exists():failed.append(j['id'])
                    else:
                        outputs.append(valid_result(read_json(result),j))
                self.report(len(queue))
                if failed:raise RuntimeError('failed games; no incomplete scores: '+','.join(failed))
                if self.active:time.sleep(2)
        finally:self.report(len(queue))
        return outputs

    def close(self):
        for p,log,r,_ in self.active:
            if p.poll() is None:p.send_signal(signal.SIGTERM)
        end=time.time()+20
        while any(p.poll() is None for p,_,_,_ in self.active) and time.time()<end:time.sleep(.5)
        for p,log,r,_ in self.active:
            if p.poll() is None:kill_tree(p.pid)
            p.wait();log.close();r.update(returncode=p.returncode,elapsed_seconds=time.time()-r['started_unix'])
        self.active.clear();self.report();self.lock.close()


def execute(root):
    r=Runner(root);root=r.root;c=r.config
    try:
        pilot=r.run(r.jobs(['model','habitat','search_habitat','search_model'],range(246000000,246000004),'pilot'),'pilot4')
        summary=summarize(pilot);choices=pilot_choices(summary)
        write_json(root/'pilot.json',dict(results=summary,expand=choices,note='Exploratory4-environment screen, not reliable strength evidence.'))
        if not choices:
            write_json(root/'summary.json',dict(status='teacher_gate_failed',stage='pilot',results=summary,expert_data_generated=False))
            r.phase='completed_no_teacher';return
        expanded=r.run(r.jobs(['model','habitat',*choices],range(246001000,246001016),'expanded'),'expanded16')
        summary=summarize(expanded);winner=teacher_choice(summary)
        write_json(root/'expanded.json',dict(results=summary,teacher=winner,uses_confirmation=False))
        if winner:
            # 24 fresh train seeds x both seats; retain ALL complete trajectories,
            # not just high scores. A teacher is a policy, not hindsight filtering.
            expert=r.run(r.jobs([winner],range(146000000,146000024),'expert'),'expert_train48')
            write_json(root/'expert_manifest.json',dict(format='v146_train_only_expert_manifest',teacher=winner,
                student=c['student'],student_sha256=c['student_sha256'],opponent=c['opponent'],
                source_id=r.launch['source_id'],split='train',games=expert,
                note='Development trajectories excluded. Frozen future RL must rebuild public states and validate masks/returns.'))
        write_json(root/'summary.json',dict(status='teacher_data_ready' if winner else 'teacher_gate_failed',
            stage='expanded',results=summary,teacher=winner,expert_data_generated=bool(winner)))
        r.phase='completed_teacher_data_ready' if winner else 'completed_no_teacher'
    except BaseException as exc:
        r.phase='interrupted' if isinstance(exc,(InterruptedError,TimeoutError)) else 'failed'
        write_json(root/(r.phase+'.json'),dict(error=repr(exc),time=time.time()));raise
    finally:r.close()


def launch(repo,root,config):
    repo=Path(repo).resolve();root=Path(root).resolve();c=read_json(config);cpus=physical_cpus()
    gate=read_json(c['preflight_receipt'])
    if (not gate['passed'] or gate['source_id']!=source_id(repo) or gate['config_sha256']!=digest(config)
            or gate['host']!=c['host']):
        raise ValueError('matching mature-model actual-host preflight required')
    if (os.uname().nodename!=c['host'] or not 2<=c['cpu_cores']<=len(cpus)//2
            or c['deadline_unix']<=time.time()+1200 or c['seconds']!=900):
        raise ValueError('authorized host, half CPUs and original bounded900s campaign required')
    if os.getloadavg()[0]>len(cpus)*.5:raise RuntimeError('CPU load guard; do not disturb other tasks')
    root.mkdir(parents=True,exist_ok=False)
    for name in ('jobs','logs','episodes'):(root/name).mkdir()
    for name in source_files(repo):
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    c.update(repo=str(repo));write_json(root/'config.json',c)
    write_json(root/'launch.json',dict(source_id=source_id(repo),source_commit=subprocess.check_output(
        ['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),config_sha256=digest(root/'config.json'),
        cpus=cpus[:c['cpu_cores']],created_unix=time.time()))
    guarded(root)


def guarded(root):
    root=Path(root).resolve();c=read_json(root/'config.json')
    if (root/'summary.json').exists() or time.time()>=c['deadline_unix']:
        raise ValueError('completed/expired campaign cannot resume or renew')
    child=subprocess.Popen([sys.executable,'-u','-m','boardbench.v146.campaign','--out',str(root),'--execute'],
                           env={**os.environ,'PYTHONPATH':str(root/'source/src')},start_new_session=True)
    os.sched_setaffinity(0,{read_json(root/'launch.json')['cpus'][-1]})
    def stop(*_):
        if child.poll() is None:child.send_signal(signal.SIGTERM)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:rc=child.wait(timeout=max(1,c['deadline_unix']-time.time()+25))
    except subprocess.TimeoutExpired:
        kill_tree(child.pid);child.wait()
        write_json(root/'guard_deadline.json',dict(deadline_unix=c['deadline_unix'],time=time.time(),pid=child.pid))
        raise
    if rc:raise SystemExit(rc)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--repo');p.add_argument('--config');p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.execute:execute(a.out)
    elif a.resume:guarded(a.out)
    else:launch(a.repo,a.out,a.config)
