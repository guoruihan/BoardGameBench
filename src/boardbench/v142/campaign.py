"""Four independent single-GPU own-score trials, fixed deadline, frozen source."""
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
from .training import paired_delta

ARMS = (('solo',0),('solo',1),('duo',0),('duo',1))


def report(root,jobs,started,deadline,phase):
    rows=[]
    for job in jobs:
        d=root/'trials'/job['name'];curve=d/'curve.jsonl';latest={}
        if curve.exists():
            lines=curve.read_text().splitlines()
            # Concurrent append may expose a partial final line; use last complete.
            for line in reversed(lines):
                try: latest=json.loads(line);break
                except json.JSONDecodeError: continue
        done=read_json(d/'done.json') if (d/'done.json').exists() else None
        initial=read_json(d/'initial.json') if (d/'initial.json').exists() else {}
        rows.append(dict(trial=job['name'],pid=job.get('pid'),gpu=job['gpu'],cpus=job['cpus'],
            actions=latest.get('actions',0),batches=latest.get('batches',0),
            initial_development_score=initial.get('score_mean'),
            training_score=latest.get('score_mean'),best_development_score=latest.get('best_development_score'),
            checkpoint=str(d/'latest.pt') if (d/'latest.pt').exists() else None,
            status='completed' if done else 'failed' if (d/'failed.json').exists() else
                   'interrupted' if phase=='interrupted' else 'running',
            confirmation=done.get('confirmation') if done else None,
            paired_vs_initial=done.get('paired_vs_initial') if done else None))
    write_json(root/'status.json',dict(phase=phase,updated_unix=time.time(),started_unix=started,
        deadline_unix=deadline,elapsed_seconds=time.time()-started,jobs=rows))


def execute(repo,root,resume=False):
    launch=read_json(root/'launch.json');config=read_json(root/'config.json')
    if os.uname().nodename!=config['host'] or source_id()!=launch['source_id']:
        raise ValueError('authorized host and immutable source required')
    if resume:
        old=read_json(root/'status.json')
        if old['phase']=='completed': raise ValueError('completed campaign cannot be extended')
        started,deadline=old['started_unix'],old['deadline_unix']
    else: started=time.time();deadline=started+config['wall_seconds']
    if deadline-time.time()<config['training']['finish_reserve_seconds']+60:
        raise ValueError('insufficient original budget; not renewed')
    lock=reserve(repo,config);cpus=launch['cpus'];os.sched_setaffinity(0,{cpus[-1]})
    env={**os.environ,'PYTHONPATH':str(root/'source/src'),'PYTHONNOUSERSITE':'1',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'NUMEXPR_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    active=[];jobs=[];stopping=[False]
    receipts=read_json(root/'cost.json')['attempts'] if resume and (root/'cost.json').exists() else []
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stopping.__setitem__(0,True))

    def costs():
        now=time.time()
        write_json(root/'cost.json',dict(attempts=receipts,wall_seconds=now-started,deadline_unix=deadline,
            allocated_cpu_core_seconds=sum(len(r['cpus'])*r.get('elapsed_seconds',now-r['started_unix']) for r in receipts)+now-started,
            physical_gpu_seconds=sum(r.get('elapsed_seconds',now-r['started_unix']) for r in receipts),
            includes='all CUDA gates, initial comparison, failed attempts, training, evaluations, saves and cleanup'))

    def start(command,cores,gpu,log,label):
        stream=log.open('a');p=subprocess.Popen(['taskset','-c',','.join(map(str,cores)),sys.executable,'-u',*command],
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={**env,'CUDA_VISIBLE_DEVICES':gpu})
        receipt=dict(stage=label,pid=p.pid,gpu=gpu,cpus=cores,started_unix=time.time())
        receipts.append(receipt);active.append((p,stream,receipt));costs();return p

    def wait(until,phase):
        signaled=False;grace_until=until
        while any(p.poll() is None for p,_,_ in active):
            for p,_,r in active:
                if p.poll() is not None and 'elapsed_seconds' not in r:
                    r.update(returncode=p.returncode,elapsed_seconds=time.time()-r['started_unix'])
            report(root,jobs,started,deadline,phase);costs()
            if stopping[0] and not signaled:
                for p,_,_ in active:
                    if p.poll() is None: p.send_signal(signal.SIGTERM)
                signaled=True;grace_until=min(until,time.time()+60)
            if time.time()>=grace_until:
                for p,_,_ in active:
                    if p.poll() is None: kill_tree(p.pid)
                break
            time.sleep(2)
        failed=False
        for p,stream,r in active:
            p.wait(timeout=20);stream.close()
            if 'elapsed_seconds' not in r: r.update(returncode=p.returncode,elapsed_seconds=time.time()-r['started_unix'])
            failed |= p.returncode!=0
        active.clear();costs()
        if stopping[0]: raise InterruptedError('campaign stopped; checkpoints and original deadline preserved')
        return failed

    try:
        report(root,jobs,started,deadline,'cuda_preflight')
        resolved=read_json(root/'resolved_config.json')
        for i,(arm,replica) in enumerate(ARMS):
            target=root/f'preflight_gpu{i}'
            if resume and (target/'result.json').exists():
                gate=read_json(target/'result.json')
                if gate['passed'] and gate['source_id']==launch['source_id']: continue
                raise ValueError('saved gate mismatch')
            start(['-m','boardbench.v142.preflight','--repo',str(root/'source'),'--config',str(root/'resolved_config.json'),
                   '--out',str(target),'--initialization',arm],cpus[4*i:4*i+4],config['gpu_uuids'][i],
                  root/'logs'/f'preflight_gpu{i}.log','preflight_'+arm)
        if wait(min(deadline,time.time()+900),'cuda_preflight'): raise RuntimeError('CUDA gates failed; no formal training')
        for i in range(4):
            gate=read_json(root/f'preflight_gpu{i}/result.json')
            if not gate['passed'] or gate['source_id']!=launch['source_id']: raise ValueError('gate provenance mismatch')
        for i,arm in enumerate(('solo','duo')):
            target=root/('baseline_'+arm)
            if resume and (target/'result.json').exists():
                if read_json(target/'result.json')['source_id']!=launch['source_id']:
                    raise ValueError('baseline source mismatch')
                continue
            start(['-m','boardbench.v142.preflight','--repo',str(root/'source'),'--config',str(root/'resolved_config.json'),
                   '--out',str(target),'--initialization',arm,'--baseline-only'],cpus[14*i:14*i+14],
                  config['gpu_uuids'][i],root/'logs'/('baseline_'+arm+'.log'),'baseline_'+arm)
        if wait(min(deadline,time.time()+900),'initial_comparison'): raise RuntimeError('baseline comparison failed')
        solo=read_json(root/'baseline_solo/result.json');duo=read_json(root/'baseline_duo/result.json')
        write_json(root/'initial_comparison.json',dict(solo=solo,duo=duo,duo_minus_solo=paired_delta(duo,solo),
            note='same fresh development128, transferred models, not official solo or duo scores'))
        for i,(arm,replica) in enumerate(ARMS):
            name=arm+'_'+'ab'[replica];path=root/'configs'/(name+'.json');target=root/'trials'/name
            if resume and path.exists(): c=read_json(path)
            else:
                c=trial_config(root/'source',resolved,arm,replica)
                c.update(max_seconds=max(0,deadline-time.time()-30),deadline_unix=deadline-15)
                write_json(path,c)
            cores=cpus[14*i:14*i+14];gpu=config['gpu_uuids'][i]
            job=dict(name=name,gpu=gpu,cpus=cores);jobs.append(job)
            if (target/'done.json').exists(): continue
            command=['-m','boardbench.v142.training','--config',str(path),'--out',str(target)]
            if resume and target.exists(): command.append('--resume')
            job['pid']=start(command,cores,gpu,root/'logs'/(name+'.log'),name).pid
        write_json(root/'jobs.json',jobs)
        failed=wait(deadline,'training')
        if not failed and not all((root/'trials'/j['name']/'done.json').exists() for j in jobs): failed=True
        report(root,jobs,started,deadline,'failed' if failed else 'completed')
        if failed: raise RuntimeError('one or more trials failed or exhausted budget; inspect receipts')
    except BaseException as exc:
        write_json(root/('interrupted.json' if isinstance(exc,InterruptedError) else 'failed.json'),dict(error=repr(exc),time=time.time()))
        report(root,jobs,started,deadline,'interrupted' if isinstance(exc,InterruptedError) else 'failed')
        raise
    finally:
        for p,stream,r in active:
            if p.poll() is None: kill_tree(p.pid)
            p.wait(timeout=20);stream.close();r.update(returncode=p.returncode,elapsed_seconds=time.time()-r['started_unix'])
        costs();lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--out',required=True);p.add_argument('--config')
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args();repo,root=Path(args.repo).resolve(),Path(args.out).resolve()
    if args.execute: return execute(repo,root,args.resume)
    if args.resume: raise ValueError('resume with frozen source and --execute; never reset deadline')
    config=read_json(args.config)
    if os.uname().nodename!=config['host']: raise ValueError('a226 only')
    physical=physical_cpus()
    if len(physical)//2<57 or len(config['gpu_uuids'])!=4 or config['workers']!=13:
        raise ValueError('57 physical cores <=half, four independent GPUs,13 workers each required')
    root.mkdir(parents=True,exist_ok=False)
    for name in ('logs','configs','trials'): (root/name).mkdir()
    for relative in source_files(repo):
        dest=root/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,dest)
    resolved={**config,**{k:str(repo/config[k]) for k in ('parent','duo_parent','native_manifest')}}
    write_json(root/'config.json',config);write_json(root/'resolved_config.json',resolved)
    write_json(root/'launch.json',dict(source_id=source_id(repo),cpus=physical[:57],
        source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()))
    os.sched_setaffinity(0,{physical[56]})
    rc=subprocess.run([sys.executable,'-u','-m','boardbench.v142.campaign','--repo',str(repo),'--out',str(root),'--execute'],
        env={**os.environ,'PYTHONPATH':str(root/'source/src')}).returncode
    if rc: raise SystemExit(rc)


if __name__=='__main__': main()
