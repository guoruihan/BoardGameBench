"""One bounded, recoverable value-fit / diagnosis / paired-evaluation experiment.

Uses frozen source and existing training infrastructure. No perpetual scheduler,
implicit budget renewal, GPU sharing with occupied cards, or test-set selection.
"""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import torch

from boardbench.artifacts.store import digest, read_json, write_json, source_id
from boardbench.v145.policy import load_policy, export_policy
from boardbench.v146.report import paired
from .data import select_teacher


class Campaign:
    def __init__(self,root):
        self.root=Path(root).resolve();self.c=read_json(self.root/'config.json')
        self.stop=False;self.phase='starting';self.costs=[]
        if digest(self.root/'config.json')!=read_json(self.root/'launch.json')['config_sha256']:
            raise ValueError('immutable campaign config changed')
        if self.c['source_id']!=source_id():raise ValueError('frozen source mismatch')
        if self.c['deadline_unix']<=time.time():raise ValueError('expired campaign; no automatic renewal')
        from boardbench.v122.campaign import physical_cpus
        physical=physical_cpus();chosen=self.c['worker_cpus']+[self.c['trainer_cpu'],self.c['controller_cpu']]
        if (len(chosen)>len(physical)//2 or len(set(chosen))!=len(chosen) or not set(chosen)<=set(physical)
                or os.uname().nodename!=self.c['host']):raise ValueError('host/CPU allocation mismatch')
        os.sched_setaffinity(0,{self.c['controller_cpu']})
        self.controller_started=time.time()
        self.lock=open(Path(self.c['repo'])/'outputs/v149'/f'{self.c["host"]}.lock','a+')
        fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (self.root/'costs.json').exists():self.costs=read_json(self.root/'costs.json')
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:setattr(self,'stop',True))

    def status(self,**extra):
        write_json(self.root/'status.json',dict(phase=self.phase,updated_unix=time.time(),
            deadline_unix=self.c['deadline_unix'],**extra))

    def execute(self,args,out,complete,cpu,gpu=False):
        complete=Path(complete)
        if complete.exists():return read_json(complete)
        if self.stop or time.time()>=self.c['deadline_unix']-30:raise InterruptedError('campaign deadline/stop')
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists() and '--resume' not in args:
            out.rename(out.with_name(out.name+f'.incomplete_{time.time_ns()}'))
        if gpu:
            # Data collection may take minutes after launch. Recheck immediately
            # before every GPU subprocess, never assume launch-time availability.
            for attempt in range(10):
                state=subprocess.check_output(['nvidia-smi','--id='+str(self.c['gpu']),
                    '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                memory,util=map(int,state.strip().split(','))
                if memory<=100 and util<=10:break
                if self.stop or time.time()>=self.c['deadline_unix']-30:raise InterruptedError('GPU wait deadline')
                time.sleep(1)
            else:raise RuntimeError('GPU became occupied; no sharing or preemption')
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
             'CUDA_VISIBLE_DEVICES':str(self.c['gpu']) if gpu else ''}
        # For evaluation each worker, including library threads, gets one CPU.
        command=['taskset','-c',str(cpu),sys.executable,*args]
        started=time.time()
        with open(str(out)+'.log','a') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            while process.poll() is None:
                if self.stop or time.time()>=self.c['deadline_unix']-15:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    break
                time.sleep(.2)
        self.costs.append(dict(command=command,out=str(out),exit_code=process.returncode,
            started_unix=started,wall_seconds=time.time()-started,cpu_slots=1,gpu_slots=int(gpu)))
        # Written by the coordinator after concurrent stages; avoid temp-file races.
        if process.returncode or not complete.exists():raise RuntimeError(f'job failed: {out}; see {out}.log')
        return read_json(complete)

    def panel(self,name,specs,seeds,split='development'):
        self.phase=name;self.status()
        slots=self.c['worker_cpus'];jobs=[(spec,seed,seat) for spec in specs for seed in seeds for seat in (0,1)]
        # One queue per CPU, rather than assigning two concurrent jobs to a core.
        queues=[jobs[i::len(slots)] for i in range(len(slots))]
        def work(index):
            rows=[]
            for spec,seed,seat in queues[index]:
                label=spec['label'];target=self.root/name/f'{label}_e{seed}_s{seat}'
                job={**self.c['job'],**spec,'source_id':self.c['source_id'],'id':target.name,
                     'environment_seed':seed,'seat':seat,'search_seed':149,'split':split,
                     'deadline_unix':self.c['deadline_unix']}
                path=self.root/'jobs'/(target.name+'_'+name+'.json');write_json(path,job)
                rows.append(self.execute(['-m','boardbench.v149.worker','--job',str(path),'--out',str(target)],
                                         target,target/'result.json',slots[index]))
            return rows
        rows=[]
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            for future in as_completed([pool.submit(work,i) for i in range(len(slots))]):
                try:rows.extend(future.result())
                except BaseException:self.stop=True;raise
                self.status(completed=len(rows),requested=len(jobs))
        write_json(self.root/'costs.json',self.costs)
        summary={}
        for spec in specs:
            selected=[r for r in rows if r['method']==spec['label']]
            metrics={k:statistics.mean(r[k] for r in selected) for k in
                ('score','animal','terrain','animals_placed','completed_cards','decision_seconds','decisions','fallback_rate','latency_p95')}
            metrics.update(games=len(selected),latency_max=max(r['latency_max'] for r in selected),
                           timing_gate=all(r['timing_gate'] for r in selected))
            if spec['method'].startswith('search_value'):
                baseline=[r for r in rows if r['method']==f'rollout_{spec["seconds"]}']
                if baseline:
                    metrics['vs_equal_budget_rollout']=paired(selected,baseline)
                    metrics['animal_vs_rollout']=paired(selected,baseline,'animal')
            summary[spec['label']]=metrics
        write_json(self.root/(name+'_summary.json'),dict(results=summary,rows=rows,
            split=split,note='Intervals are descriptive environment clusters, not selection-corrected.'))
        return summary

    def run(self):
        self.phase='data';self.status();data_dir=self.root/'data'
        select_teacher(self.c['dataset'],data_dir)
        initial=self.root/'random_value_init.pt'
        if not initial.exists():
            model=load_policy(self.c['job']['student'])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(149)
                for module in model.modules():
                    if isinstance(module,torch.nn.Linear):module.reset_parameters()
            export_policy(model,initial,dict(parent_sha256=self.c['job']['student_sha256'],
                source_id=self.c['source_id'],mode='fixed',progress_scale=0.,sil_weight=0.,gae_lambda=1.))
        configs=[]
        for name,lr in [('fresh',.0001),('fresh_fast',.0003)]:
            cfg=dict(model=str(initial),model_sha256=digest(initial),dataset=str(data_dir/'manifest.json'),
                dataset_sha256=digest(data_dir/'manifest.json'),mode='value',learning_rate=lr,anchor_weight=0.,
                value_weight=1.,minibatch=256,seed=149,steps=self.c.get('fit_steps',1800),evaluate_interval=100,
                deadline_unix=self.c['deadline_unix'],finish_reserve_seconds=30,source_id=self.c['source_id'])
            path=self.root/'configs'/(name+'.json');write_json(path,cfg);configs.append((name,path))
        gpu=self.c.get('gpu') is not None;device='cuda' if gpu else 'cpu'
        self.phase='device_preflight';self.status()
        self.execute(['-m','boardbench.v149.preflight','--config',str(configs[0][1]),'--out',str(self.root/'preflight'),
                      '--device',device],self.root/'preflight',self.root/'preflight/result.json',self.c['trainer_cpu'],gpu)
        values={'old':self.c['old_value']}
        for name,path in configs:
            self.phase='fit_'+name;self.status();out=self.root/'fit'/name
            args=['-m','boardbench.v148.fit','--config',str(path),'--out',str(out),'--device',device]
            if (out/'latest.pt').exists() and not (out/'done.json').exists():args+=['--resume']
            self.execute(args,out,out/'done.json',self.c['trainer_cpu'],gpu)
            manifest=out/'value_manifest.json';values[name]=dict(path=str(manifest),sha256=digest(manifest))
            write_json(self.root/'costs.json',self.costs)
        self.phase='ranking_diagnostic';self.status()
        from boardbench.v129.encoding import NativeEncoder
        from boardbench.v146.policy import ModelPlayer
        from .value import Value
        from .diagnostic import evaluate
        torch.set_num_threads(1)
        native=NativeEncoder.from_artifact(self.c['job']['native_artifact'])
        focal=ModelPlayer(load_policy(self.c['job']['student']),native)
        opponent=ModelPlayer(load_policy(self.c['job']['opponent']),native)
        for name,m in values.items():
            path=self.root/'diagnostics'/(name+'.json')
            prediction=self.root/'diagnostics'/(name+'_prediction.json')
            if not path.exists() or not prediction.exists():
                value=Value(m['path'],m['sha256'],native,self.c['job']['opponent_sha256'])
                if not path.exists():evaluate(data_dir/'manifest.json',value,focal,opponent,path)
                from .diagnostic import prediction_report
                if not prediction.exists():prediction_report(data_dir/'manifest.json',value,prediction)
        specs=[dict(label='model',method='model',seconds=.1),dict(label='habitat',method='habitat',seconds=.1)]
        for budget in (.1,1.):
            specs.append(dict(label=f'rollout_{budget}',method='search_rollout',seconds=budget))
            for name,m in values.items():
                specs.append(dict(label=f'{name}_turn_{budget}',method='search_value_turn',seconds=budget,
                                  value_manifest=m['path'],value_manifest_sha256=m['sha256']))
        for name,m in values.items():
            specs.append(dict(label=f'{name}_one_0.1',method='search_value_one',seconds=.1,
                              value_manifest=m['path'],value_manifest_sha256=m['sha256']))
            if name!='old':
                specs.append(dict(label=f'{name}_short_0.1',method='search_value_short',seconds=.1,
                                  value_manifest=m['path'],value_manifest_sha256=m['sha256']))
        summary=self.panel('pilot',specs,range(249000000,249000004))
        selected=[]
        for budget in (.1,1.):
            options=[s for s in specs if s['method'].startswith('search_value') and s['seconds']==budget
                     and summary[s['label']]['timing_gate']]
            if options:
                best=max(options,key=lambda s:summary[s['label']]['score'])
                # Expand the best value hypothesis even when four environments
                # are negative; a900s teacher gate must not halt this experiment.
                selected.append(best)
        expanded=None
        if selected and time.time()<self.c['deadline_unix']-300:
            keep=[s for s in specs if s['method'] in ('model','habitat','search_rollout')]+selected
            expanded=self.panel('expanded',keep,range(249001000,249001016))
        self.phase='completed';self.status()
        write_json(self.root/'summary.json',dict(status='completed',pilot=summary,expanded=expanded,
            retained=[s['label'] for s in selected],completed_unix=time.time(),
            costs=dict(allocated_cpu_seconds=sum(c['wall_seconds'] for c in self.costs)+time.time()-self.controller_started,
                       allocated_gpu_seconds=sum(c['wall_seconds']*c['gpu_slots'] for c in self.costs)),
            note='Old policies preserved. No automatic production promotion or claim of optimality.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    cls=Campaign
    if read_json(Path(a.root)/'config.json').get('feedback_parent'):
        from .feedback import Feedback
        cls=Feedback
    r=cls(a.root)
    try:r.run()
    except BaseException as e:
        r.phase='interrupted' if r.stop or time.time()>=r.c['deadline_unix']-30 else 'failed'
        r.status(error=repr(e));write_json(r.root/'costs.json',r.costs);raise
