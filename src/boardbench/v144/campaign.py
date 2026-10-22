"""Three result-driven rounds, four independent GPUs, one original eight-hour cap."""
import argparse
import json
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
from boardbench.v127.preflight import reserve
from .preflight import trial_config
from .decisions import first_round,next_round


def last_curve(path):
    if not path.exists():return {}
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-262144))
        lines=f.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except json.JSONDecodeError:pass
    return {}


def retained_records(root):
    records=[]
    for path in sorted((root/'rounds').glob('r*/*/done.json')):
        trial=path.parent;config=read_json(trial/'config.json');policy=trial/'best_policy.pt'
        records.append(dict(name=trial.parent.name+'_'+trial.name,config=config,config_path=str(trial/'config.json'),
            policy=str(policy),sha256=digest(policy),evaluation=read_json(trial/'decision.json')))
    return records


class Runner:
    def __init__(self,repo,root):
        self.repo,self.root=repo,root;self.c=read_json(root/'resolved_config.json');self.launch=read_json(root/'launch.json')
        self.deadline=self.c['deadline_unix'];self.started=self.c['started_unix'];self.phase='initializing'
        self.active=[];self.receipts=read_json(root/'cost.json')['attempts'] if (root/'cost.json').exists() else []
        self.stopping=False;self.controller_started=time.time()
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,self.stop)
        self.env={**os.environ,'PYTHONPATH':str(root/'source/src'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}

    def stop(self,*_):self.stopping=True

    def report(self):
        now=time.time();trials=[]
        for cfg in sorted((self.root/'rounds').glob('r*/*/config.json')):
            path=cfg.parent;last=last_curve(path/'curve.jsonl');done=path/'done.json'
            trials.append(dict(name=path.parent.name+'_'+path.name,batches=last.get('batches',0),actions=last.get('actions',0),
                objective=read_json(cfg)['objective'],best_development_metric=last.get('best_development_metric'),
                latest_development=last.get('development'),status='completed' if done.exists() else
                'failed' if (path/'failed.json').exists() else 'interrupted' if (path/'interrupted.json').exists() else 'running'))
        write_json(self.root/'status.json',dict(phase=self.phase,pid=os.getpid(),updated_unix=now,
            started_unix=self.started,deadline_unix=self.deadline,elapsed_seconds=now-self.started,
            remaining_seconds=max(0,self.deadline-now),active=[r for p,_,r,_ in self.active if p.poll() is None],trials=trials))
        write_json(self.root/'cost.json',dict(attempts=self.receipts,wall_seconds=now-self.started,deadline_unix=self.deadline,
            physical_gpu_seconds=sum(r.get('elapsed_seconds',now-r['started_unix']) for r in self.receipts),
            allocated_cpu_core_seconds=sum(len(r['cpus'])*r.get('elapsed_seconds',now-r['started_unix']) for r in self.receipts)+now-self.controller_started,
            setup_wall_seconds=self.controller_started-self.started,
            note='Process occupancy, not GPU utilization. Includes handoff overlap when imported, CUDA gates, trials, evaluations, retries and saves. Setup wall time is inside original eight hours; prelaunch editor/test CPU is not measured here.'))

    def start(self,spec,slot):
        if self.stopping or time.time()>=self.deadline:raise InterruptedError('no new work after stop/deadline')
        gpu=self.c['gpu_uuids'][slot];cpus=self.launch['cpus'][14*slot:14*slot+14]
        output=Path(spec['out']);args=list(spec['args'])
        if output.exists():
            if spec.get('training') and (output/'latest.pt').exists():args.append('--resume')
            else:output.rename(output.with_name(output.name+f'.incomplete_{time.time_ns()}'))
        log=(self.root/'logs'/(spec['name']+'.log')).open('a')
        p=subprocess.Popen(['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u',*args],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**self.env,'CUDA_VISIBLE_DEVICES':gpu})
        r=dict(stage=spec['name'],pid=p.pid,gpu=gpu,cpus=cpus,slot=slot,started_unix=time.time())
        self.receipts.append(r);self.active.append((p,log,r,spec));self.report()

    def run(self,specs,until,phase):
        self.phase=phase;pending=[s for s in specs if not Path(s['complete']).exists()]
        for offset in range(0,len(pending),4):
            wave=pending[offset:offset+4]
            if until-time.time()<10:raise TimeoutError('insufficient original stage allowance')
            for slot,spec in enumerate(wave):self.start(spec,slot)
            signaled=False;grace=until
            failures=[]
            while self.active:
                # Reap/retry each failed slot immediately, while its peers may
                # continue training for hours. Never move retries to GPU0.
                for item in list(self.active):
                    p,stream,r,spec=item
                    if p.poll() is None:continue
                    p.wait(timeout=20);stream.close();self.active.remove(item)
                    r.update(returncode=p.returncode,elapsed_seconds=time.time()-r['started_unix'])
                    if p.returncode==0 and Path(spec['complete']).exists():continue
                    failed_attempts=sum(q['stage']==spec['name'] and q.get('returncode') is not None
                        and (q['returncode']!=0 or q.get('missing_completion',False)) for q in self.receipts)
                    if p.returncode==0:r['missing_completion']=True;failed_attempts+=1
                    retry_floor=self.c.get('retry_min_remaining_seconds',180)
                    if (not self.stopping and failed_attempts<=1
                            and min(until,self.deadline)-time.time()>retry_floor):
                        write_json(self.root/'decisions'/f'retry_{time.time_ns()}.json',dict(jobs=[spec['name']],
                            slot=r['slot'],gpu=r['gpu'],cpus=r['cpus'],reason='immediate same-slot bounded retry',
                            failed_attempts=failed_attempts,until=until))
                        self.start(spec,r['slot'])
                    else:failures.append(spec['name'])
                if self.stopping and not signaled:
                    for p,_,_,_ in self.active:
                        if p.poll() is None:p.send_signal(signal.SIGTERM)
                    signaled=True;grace=min(until,time.time()+60)
                if time.time()>=min(grace,self.deadline):
                    for p,_,_,_ in self.active:
                        if p.poll() is None:kill_tree(p.pid)
                    break
                self.report()
                if self.active:time.sleep(2)
            # Only hard-stopped processes remain here; completed processes were
            # already reaped individually above. No deferred wave-level retry.
            for p,stream,r,spec in self.active:
                p.wait(timeout=20);stream.close()
                if 'elapsed_seconds' not in r:r.update(returncode=p.returncode,elapsed_seconds=time.time()-r['started_unix'])
                if p.returncode or not Path(spec['complete']).exists():failures.append(spec['name'])
            self.active.clear();self.report()
            if self.stopping:raise InterruptedError('graceful stop with original deadline retained')
            if failures:raise RuntimeError('incomplete jobs after bounded recovery: '+','.join(failures))
        self.report()

    def close(self):
        for p,stream,r,_ in self.active:
            if p.poll() is None:kill_tree(p.pid)
            p.wait(timeout=20);stream.close()
            r.update(returncode=p.returncode,elapsed_seconds=time.time()-r['started_unix'])
        self.active.clear();self.report()


def training_spec(root,config_path,target):
    return dict(name=target.parent.name+'_'+target.name,out=str(target),training=True,complete=str(target/'done.json'),
        args=['-m','boardbench.v144.training','--config',str(config_path),'--out',str(target)])


def evaluation_spec(root,record,seeds,stage,deadline=None):
    name=record['name'];target=root/stage/name;path=root/'configs'/(stage+'_'+name+'.json')
    spec=dict(config=record['config_path'],model=record['policy'],model_sha256=record['sha256'],seeds=seeds,
        stage='confirmation' if stage=='confirmation' else 'development')
    if deadline is not None:spec['deadline_unix']=deadline
    if path.exists() and read_json(path)!=spec:raise ValueError('evaluation spec changed on resume')
    write_json(path,spec)
    marker='summary.json' if stage=='confirmation' else 'result.json'
    return dict(name=stage+'_'+name,out=str(target),complete=str(target/marker),
        args=['-m','boardbench.v144.evaluate','--spec',str(path),'--out',str(target)])


def execute(repo,root):
    c=read_json(root/'resolved_config.json');launch=read_json(root/'launch.json')
    if os.uname().nodename!=c['host'] or source_id()!=launch['source_id']:raise ValueError('authorized host and frozen source required')
    if digest(root/'resolved_config.json')!=launch['resolved_config_sha256']:raise ValueError('immutable campaign config/deadline changed')
    if (root/'summary.json').exists():raise ValueError('completed campaign cannot be extended')
    if c['deadline_unix']-time.time()<120:raise ValueError('original deadline expired; no automatic extension')
    lock=reserve(repo,c);os.sched_setaffinity(0,{launch['cpus'][-1]});runner=Runner(repo,root)
    try:
        # V143 process occupancy overlapping this task, not its earlier cost.
        if not runner.receipts and c.get('handoff_cost'):
            for row in read_json(c['handoff_cost'])['attempts']:
                end=row['started_unix']+row['elapsed_seconds'];begin=max(c['started_unix'],row['started_unix'])
                if end>begin:runner.receipts.append({**row,'stage':'prior_handoff_'+row['stage'],'started_unix':begin,'elapsed_seconds':end-begin})
        gates=[]
        for i,arm in enumerate(first_round()):
            # Exercise diverse frozen/history exclusion before its later adaptive use.
            gate_arm={**arm,'mode':'diverse'} if i==1 else arm
            cfg=trial_config(root/'source',c,gate_arm);path=root/'configs'/f'gate{i}.json';write_json(path,cfg)
            target=root/f'preflight_gpu{i}'
            gates.append(dict(name=f'preflight_gpu{i}',out=str(target),complete=str(target/'result.json'),
                args=['-m','boardbench.v144.preflight','--config',str(path),'--out',str(target)]))
        runner.run(gates,min(c['deadline_unix'],time.time()+900),'cuda_preflight')
        for i in range(4):
            result=read_json(root/f'preflight_gpu{i}/result.json')
            if not result['passed'] or result['source_id']!=launch['source_id']:raise ValueError('CUDA gate provenance mismatch')
        schedule_path=root/'schedule.json'
        if schedule_path.exists():schedule=read_json(schedule_path)
        else:
            available=c['deadline_unix']-c['final_reserve_seconds']-time.time()
            if available<3*900:raise ValueError('insufficient three-round allowance')
            begin=time.time();schedule=dict(round_deadlines=[begin+available*(i+1)/3 for i in range(3)],deadline_unix=c['deadline_unix'])
            write_json(schedule_path,schedule)
        for round_index,until in enumerate(schedule['round_deadlines']):
            plan_path=root/'decisions'/f'round_{round_index+1}.json'
            if plan_path.exists():plan=read_json(plan_path);arms=plan['arms']
            else:
                if round_index==0:arms=first_round();decision=dict(rationale='paired objective, opponent-history and full-return contrasts',uses_confirmation=False)
                else:
                    records=[r for r in retained_records(root) if r['name'].startswith(f'r{round_index}_')]
                    arms,decision=next_round(records,round_index)
                plan=dict(arms=arms,decision=decision,created_unix=time.time(),deadline_unix=until)
                write_json(plan_path,plan)
            jobs=[]
            for arm in arms:
                name=arm['name'];target=root/'rounds'/f'r{round_index+1}'/name;path=root/'configs'/f'r{round_index+1}_{name}.json'
                cfg=trial_config(root/'source',c,arm,round_index)
                cfg.update(deadline_unix=until-20,max_seconds=max(0,until-time.time()))
                if path.exists():cfg=read_json(path)
                else:write_json(path,cfg)
                jobs.append(training_spec(root,path,target))
            runner.run(jobs,until,f'round_{round_index+1}')
            records=retained_records(root)
            write_json(root/'retained.json',[{k:v for k,v in r.items() if k not in ('evaluation','config')} for r in records])
        records=retained_records(root)
        first=records[0];initial=Path(first['policy']).parent/'initial_policy.pt'
        records.append(dict(name='initial_transfer',config_path=first['config_path'],policy=str(initial),sha256=digest(initial)))
        # All retained variants face the SAME final DEVELOPMENT panel before selection.
        selection_seeds=list(range(244100000,244100000+c['selection_environments']))
        specs=[evaluation_spec(root,r,selection_seeds,'selection',c['deadline_unix']-600) for r in records]
        runner.run(specs,min(c['deadline_unix']-600,time.time()+1200),'final_development_selection')
        selected_path=root/'selection.json'
        if selected_path.exists():selected=read_json(selected_path)['models']
        else:
            ranked=sorted(records,key=lambda r:read_json(root/'selection'/r['name']/'result.json')['score_mean'],reverse=True)
            # Keep the top three by own score plus the strongest margin-control
            # checkpoint by the SAME own-score panel; do not consult confirmation.
            selected=ranked[:3]
            controls=[r for r in ranked if r.get('config',{}).get('objective')=='margin']
            extra=next((r for r in controls if r['name'] not in {s['name'] for s in selected}),None)
            if extra is None:extra=next(r for r in ranked if r['name'] not in {s['name'] for s in selected})
            selected.append(extra)
            selected=[{k:v for k,v in r.items() if k not in ('config','evaluation')} for r in selected]
            write_json(selected_path,dict(models=selected,uses_confirmation=False,
                ranking=[dict(name=r['name'],score=read_json(root/'selection'/r['name']/'result.json')['score_mean']) for r in ranked]))
        seeds=list(range(344000000,344000000+c['confirmation_environments']))
        runner.run([evaluation_spec(root,r,seeds,'confirmation',c['deadline_unix']-15) for r in selected],c['deadline_unix']-15,'independent_confirmation')
        summary=dict(status='completed',selection_uses_confirmation=False,started_unix=c['started_unix'],deadline_unix=c['deadline_unix'],
            completed_unix=time.time(),selected=[dict(name=r['name'],**read_json(root/'confirmation'/r['name']/'summary.json')) for r in selected],
            rounds=[read_json(root/'decisions'/f'round_{i+1}.json') for i in range(3)])
        write_json(root/'summary.json',summary)
        # A self-contained human report is generated without further LLM turns.
        lines=['# V144 final report','', 'Selection uses development only. Official duo rules; 512 environments, both seats and two fixed opponents.', '',
               '| Policy | Own score | Animal | Margin | Win credit | Paired score gain vs transfer (95% CI) |',
               '|---|---:|---:|---:|---:|---|']
        for row in summary['selected']:
            m=row['result'];d=row['paired_vs_transfer']['metrics']['score']
            lines.append(f"| {row['name']} | {m['score_mean']:.3f} | {m['animal_mean']:.3f} | {m['margin_mean']:.3f} | {m['win_credit_mean']:.3%} | {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
        lines.extend(['','All checkpoints and full trajectories are retained. No deployment pointer or formal test was changed.',
                      'Resource occupancy: see cost.json. Decisions and rationale: decisions/round_*.json.'])
        (root/'report.md').write_text('\n'.join(lines)+'\n')
        runner.phase='completed';runner.report()
    except BaseException as exc:
        runner.phase='interrupted' if isinstance(exc,InterruptedError) else 'failed'
        write_json(root/(runner.phase+'.json'),dict(error=repr(exc),time=time.time()));raise
    finally:runner.close();lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--out',required=True);p.add_argument('--config');p.add_argument('--execute',action='store_true')
    a=p.parse_args();repo=Path(a.repo).resolve();root=Path(a.out).resolve()
    if a.execute:return execute(repo,root)
    c=read_json(a.config);physical=physical_cpus()
    if (os.uname().nodename!=c['host'] or len(physical)//2<57 or len(c['gpu_uuids'])!=4 or c['workers']!=13
            or c['deadline_unix']-c['started_unix']!=28800 or time.time()>=c['deadline_unix']):
        raise ValueError('authorized host, four GPUs,57 cores<=half and original eight-hour window required')
    root.mkdir(parents=True,exist_ok=False)
    for name in ('logs','configs','rounds','decisions','selection','confirmation'):(root/name).mkdir()
    for relative in source_files(repo):
        target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,target)
    resolved={**c,**{k:str(repo/c[k]) for k in ('parent','native_manifest','handoff_cost')}}
    resolved['evaluation_opponents']=[{**r,'path':str(repo/r['path'])} for r in c['evaluation_opponents']]
    write_json(root/'config.json',c);write_json(root/'resolved_config.json',resolved)
    write_json(root/'launch.json',dict(source_id=source_id(repo),cpus=physical[:57],resolved_config_sha256=digest(root/'resolved_config.json'),
        source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()))
    rc=subprocess.run([sys.executable,'-u','-m','boardbench.v144.campaign','--repo',str(repo),'--out',str(root),'--execute'],
        env={**os.environ,'PYTHONPATH':str(root/'source/src')}).returncode
    if rc:raise SystemExit(rc)


if __name__=='__main__':main()
