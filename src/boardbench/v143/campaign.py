"""Frozen-source, four independent self-play trials with bounded post-run crossplay."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from boardbench.artifacts.store import read_json,write_json,source_files,source_id
from boardbench.v12.budget import kill_tree
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.preflight import reserve
from .preflight import trial_config

ARMS=(('current',0),('current',1),('league',0),('league',1))
PAIRS=(('current_a','league_a'),('current_a','league_b'),('current_b','league_a'),('current_b','league_b'))


def report(root,jobs,started,deadline,phase):
    rows=[]
    for j in jobs:
        d=root/'trials'/j['name'];latest={};curve=d/'curve.jsonl'
        if curve.exists():
            for line in reversed(curve.read_text().splitlines()):
                try:latest=json.loads(line);break
                except json.JSONDecodeError:continue
        done=read_json(d/'done.json') if (d/'done.json').exists() else None
        rows.append(dict(trial=j['name'],pid=j.get('pid'),gpu=j['gpu'],cpus=j['cpus'],
            actions=latest.get('actions',0),batches=latest.get('batches',0),
            best_development_margin=latest.get('best_development_margin'),
            learner_actions_by_seat=latest.get('learner_actions_by_seat'),
            checkpoint=str(d/'latest.pt') if (d/'latest.pt').exists() else None,
            status='completed' if done else 'failed' if (d/'failed.json').exists() else
                   'interrupted' if phase=='interrupted' else 'running',
            confirmation=done.get('confirmation') if done else None))
    write_json(root/'status.json',dict(phase=phase,updated_unix=time.time(),started_unix=started,
        deadline_unix=deadline,elapsed_seconds=time.time()-started,jobs=rows))


def execute(repo,root,resume=False):
    launch=read_json(root/'launch.json');c=read_json(root/'config.json')
    if os.uname().nodename!=c['host'] or source_id()!=launch['source_id']:raise ValueError('authorized host/frozen source required')
    if resume:
        previous=read_json(root/'status.json')
        if previous['phase']=='completed':raise ValueError('completed campaign cannot be extended')
        started,deadline=previous['started_unix'],previous['deadline_unix']
    else:started=time.time();deadline=started+c['wall_seconds']
    if deadline-time.time()<60:raise ValueError('original deadline expired/insufficient; not renewed')
    lock=reserve(repo,c);cpus=launch['cpus'];os.sched_setaffinity(0,{cpus[-1]})
    env={**os.environ,'PYTHONPATH':str(root/'source/src'),'PYTHONNOUSERSITE':'1',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'NUMEXPR_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    receipts=read_json(root/'cost.json')['attempts'] if resume and (root/'cost.json').exists() else []
    active=[];jobs=[];stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopping.__setitem__(0,True))

    def costs():
        now=time.time()
        write_json(root/'cost.json',dict(attempts=receipts,wall_seconds=now-started,deadline_unix=deadline,
            allocated_cpu_core_seconds=sum(len(r['cpus'])*r.get('elapsed_seconds',now-r['started_unix']) for r in receipts)+now-started,
            physical_gpu_seconds=sum(r.get('elapsed_seconds',now-r['started_unix']) for r in receipts),
            includes='CUDA gates, initial self-battle, failures, training, confirmation, crossplay, saves and cleanup'))

    def start(command,cores,gpu,log,label):
        stream=log.open('a');child=subprocess.Popen(['taskset','-c',','.join(map(str,cores)),sys.executable,'-u',*command],
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={**env,'CUDA_VISIBLE_DEVICES':gpu})
        receipt=dict(stage=label,pid=child.pid,gpu=gpu,cpus=cores,started_unix=time.time())
        receipts.append(receipt);active.append((child,stream,receipt));costs();return child

    def wait(until,phase):
        signaled=False;grace_until=until
        while any(p.poll() is None for p,_,_ in active):
            for p,_,r in active:
                if p.poll() is not None and 'elapsed_seconds' not in r:
                    r.update(returncode=p.returncode,elapsed_seconds=time.time()-r['started_unix'])
            report(root,jobs,started,deadline,phase);costs()
            if stopping[0] and not signaled:
                for p,_,_ in active:
                    if p.poll() is None:p.send_signal(signal.SIGTERM)
                signaled=True;grace_until=min(until,time.time()+60)
            if time.time()>=grace_until:
                for p,_,_ in active:
                    if p.poll() is None:kill_tree(p.pid)
                break
            time.sleep(2)
        failed=False
        for p,stream,r in active:
            p.wait(timeout=20);stream.close()
            if 'elapsed_seconds' not in r:r.update(returncode=p.returncode,elapsed_seconds=time.time()-r['started_unix'])
            failed|=p.returncode!=0
        active.clear();costs()
        if stopping[0]:raise InterruptedError('graceful stop; original deadline retained')
        if failed:raise RuntimeError(phase+' failed or exceeded budget; inspect logs')

    try:
        report(root,jobs,started,deadline,'cuda_preflight');resolved=read_json(root/'resolved_config.json')
        for i,(mode,replica) in enumerate(ARMS):
            target=root/f'preflight_gpu{i}'
            if resume and (target/'result.json').exists():
                gate=read_json(target/'result.json')
                if gate['passed'] and gate['source_id']==launch['source_id']:continue
                raise ValueError('saved CUDA gate mismatch')
            start(['-m','boardbench.v143.preflight','--repo',str(root/'source'),'--config',str(root/'resolved_config.json'),
                   '--out',str(target),'--mode',mode],cpus[4*i:4*i+4],c['gpu_uuids'][i],root/'logs'/f'preflight_gpu{i}.log','preflight_'+mode)
        wait(min(deadline,time.time()+900),'cuda_preflight')
        for i in range(4):
            gate=read_json(root/f'preflight_gpu{i}/result.json')
            if not gate['passed'] or gate['source_id']!=launch['source_id']:raise ValueError('gate source mismatch')
        baseline=root/'initial_self_battle'
        if not (resume and (baseline/'result.json').exists() and (baseline/'self_battle_examples.json.gz').exists()):
            start(['-m','boardbench.v143.preflight','--repo',str(root/'source'),'--config',str(root/'resolved_config.json'),
                   '--out',str(baseline),'--mode','current','--baseline-only'],cpus[:14],c['gpu_uuids'][0],
                  root/'logs/initial_self_battle.log','initial_self_battle')
        wait(min(deadline,time.time()+900),'initial_self_battle')
        if read_json(baseline/'result.json')['source_id']!=launch['source_id']:raise ValueError('baseline source mismatch')
        for i,(mode,replica) in enumerate(ARMS):
            name=mode+'_'+'ab'[replica];path=root/'configs'/(name+'.json');target=root/'trials'/name
            if resume and path.exists():config=read_json(path)
            else:
                config=trial_config(root/'source',resolved,mode,replica)
                config.update(max_seconds=max(0,deadline-time.time()-30),deadline_unix=deadline-15);write_json(path,config)
            job=dict(name=name,cpus=cpus[14*i:14*i+14],gpu=c['gpu_uuids'][i]);jobs.append(job)
            if (target/'done.json').exists():continue
            if deadline-time.time()<config['finish_reserve_seconds']+60:raise ValueError('insufficient original training budget')
            command=['-m','boardbench.v143.training','--config',str(path),'--out',str(target)]
            if resume and target.exists():command.append('--resume')
            job['pid']=start(command,job['cpus'],job['gpu'],root/'logs'/(name+'.log'),name).pid
        write_json(root/'jobs.json',jobs);wait(deadline,'training')
        if not all((root/'trials'/j['name']/'done.json').exists() for j in jobs):raise RuntimeError('trial completion missing')
        for i,(a,b) in enumerate(PAIRS):
            label=a+'_vs_'+b;target=root/'crossplay'/label
            if resume and (target/'result.json').exists():continue
            start(['-m','boardbench.v143.crossplay','--config',str(root/'configs'/(a+'.json')),
                   '--model-a',str(root/'trials'/a/'best_policy.pt'),'--model-b',str(root/'trials'/b/'best_policy.pt'),
                   '--out',str(target)],cpus[14*i:14*i+14],c['gpu_uuids'][i],root/'logs'/(label+'.log'),'crossplay_'+label)
        wait(deadline,'crossplay')
        summary={a+'_vs_'+b:{k:v for k,v in read_json(root/'crossplay'/(a+'_vs_'+b)/'result.json').items() if k!='per_game'} for a,b in PAIRS}
        write_json(root/'crossplay.json',summary);report(root,jobs,started,deadline,'completed')
    except BaseException as exc:
        interrupted=isinstance(exc,InterruptedError)
        write_json(root/('interrupted.json' if interrupted else 'failed.json'),dict(error=repr(exc),time=time.time()))
        report(root,jobs,started,deadline,'interrupted' if interrupted else 'failed');raise
    finally:
        for p,stream,r in active:
            if p.poll() is None:kill_tree(p.pid)
            p.wait(timeout=20);stream.close();r.update(returncode=p.returncode,elapsed_seconds=time.time()-r['started_unix'])
        costs();lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--out',required=True);p.add_argument('--config')
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    repo,root=Path(a.repo).resolve(),Path(a.out).resolve()
    if a.execute:return execute(repo,root,a.resume)
    if a.resume:raise ValueError('resume requires frozen source and --execute')
    c=read_json(a.config);physical=physical_cpus()
    if os.uname().nodename!=c['host'] or len(physical)//2<57 or len(c['gpu_uuids'])!=4 or c['workers']!=13:
        raise ValueError('a226,4 independent GPUs,57 cores<=half and13 workers per job required')
    root.mkdir(parents=True,exist_ok=False)
    for name in ('logs','configs','trials','crossplay'):(root/name).mkdir()
    for relative in source_files(repo):
        dest=root/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,dest)
    resolved={**c,**{k:str(repo/c[k]) for k in ('parent','native_manifest')}}
    write_json(root/'config.json',c);write_json(root/'resolved_config.json',resolved)
    write_json(root/'launch.json',dict(source_id=source_id(repo),cpus=physical[:57],
        source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()))
    os.sched_setaffinity(0,{physical[56]})
    rc=subprocess.run([sys.executable,'-u','-m','boardbench.v143.campaign','--repo',str(repo),'--out',str(root),'--execute'],
        env={**os.environ,'PYTHONPATH':str(root/'source/src')}).returncode
    if rc:raise SystemExit(rc)


if __name__=='__main__':main()
