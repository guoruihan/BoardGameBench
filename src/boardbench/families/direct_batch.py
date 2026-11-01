"""Bounded parallel direct-play queue; API wiring stays in private adapter files."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .direct_play import dump


def summarize(root, jobs):
    groups=defaultdict(list)
    for job in jobs:
        p=root/'matches'/job['id']/'result.json'
        if p.exists():groups[(job['game'],job['opponent'],job['label'])].append(json.loads(p.read_text()))
    rows=[]
    scheduled=Counter((j['game'],j['opponent'],j['label']) for j in jobs)
    for key,count in scheduled.items():
        results=groups[key];complete=[r for r in results if r['status']=='complete']
        wins=sum(len(r['outcome']['winners'])==1 and r['job']['seat'] in r['outcome']['winners'] for r in complete)
        draws=sum(len(r['outcome']['winners'])==2 for r in complete)
        scores=[r['outcome']['scores'][r['job']['seat']] for r in complete]
        other=[r['outcome']['scores'][1-r['job']['seat']] for r in complete]
        rows.append(dict(game=key[0],opponent=key[1],label=key[2],scheduled=count,
            statuses=dict(Counter(r['status'] for r in results)),complete=len(complete),wins=wins,draws=draws,
            losses=len(complete)-wins-draws,mean_score=sum(scores)/len(scores) if scores and None not in scores else None,
            opponent_mean_score=sum(other)/len(other) if other and None not in other else None,
            calls=sum(r['calls'] for r in results),invalid=sum(r['invalid_responses'] for r in results),
            wall_seconds=sum(r['wall_seconds'] or 0 for r in results),
            search_seconds=sum(r['search_seconds'] or 0 for r in results)))
    dump(root/'summary.json',dict(updated_unix=time.time(),rows=rows))
    return Counter(r['status'] for rows in groups.values() for r in rows)


def run_one(root, job, pool, rules, adapter, core, deadline):
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    # Immutable release is placed first by the launcher and inherited by children.
    job_file=root/'jobs'/f"{job['id']}.json"
    out=root/'matches'/job['id']
    command=['taskset','-c',str(core),sys.executable,'-m','boardbench.families.direct_play',
        '--pool',str(pool),'--job',str(job_file),'--adapter',str(adapter),
        '--rules',str(rules/f"{job['game']}.md"),'--out',str(out)]
    timeout=max(1,min(10900,deadline-time.time()))
    with (root/'logs'/f"{job['id']}.log").open('a') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    path=out/'result.json'
    if path.exists():return json.loads(path.read_text())['status']
    out.mkdir(exist_ok=True)
    dump(path,dict(job=job,status='worker_error',outcome=dict(terminated=False,scores=[None,None],winners=[]),
        calls=len(list(out.glob('call_*.request.json'))),invalid_responses=0,
        wall_seconds=None,search_seconds=None))
    return 'worker_error'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--pool',required=True)
    parser.add_argument('--workers-per-label',type=int,default=8)
    parser.add_argument('--cores',required=True,help='Comma separated logical CPUs, one per worker')
    parser.add_argument('--hours',type=float,default=16)
    args=parser.parse_args()
    root=Path(args.root).resolve();pool=Path(args.pool).resolve()
    schedule=json.loads((root/'schedule.json').read_text());jobs=schedule['jobs']
    labels=sorted({j['label'] for j in jobs});cores=[int(c) for c in args.cores.split(',')]
    needed=len(labels)*args.workers_per_label
    if args.workers_per_label<1 or len(set(cores))<needed or needed>os.cpu_count()//2:
        raise ValueError('Invalid concurrency; require unique cores and <= half host logical CPUs')
    if not set(cores).issubset(os.sched_getaffinity(0)):raise ValueError('CPU outside allowed affinity')
    deadline=time.time()+args.hours*3600
    if (root/'deadline.json').exists():deadline=json.loads((root/'deadline.json').read_text())['unix']
    else:dump(root/'deadline.json',{'unix':deadline})
    for name in ('logs','jobs','matches'):(root/name).mkdir(exist_ok=True)
    queues={label:[] for label in labels};workers={};available=list(cores[:needed])
    retries=Counter();paused=defaultdict(float);stopped=set()
    for job in jobs:
        dump(root/'jobs'/f"{job['id']}.json",job)
        directory=root/'matches'/job['id'];result=directory/'result.json'
        if result.exists():
            old=json.loads(result.read_text())
            if old['status']!='provider_rate_limit' or not (directory/'resume.pkl').exists():continue
        elif directory.exists():
            # Do not repeat an interrupted 900s search without a resumable checkpoint.
            continue
        queues[job['label']].append(job)
    with ThreadPoolExecutor(max_workers=needed) as executor:
        while (any(queues.values()) or workers) and time.time()<deadline:
            counts=Counter(job['label'] for job,_ in workers.values())
            for label in labels:
                while queues[label] and counts[label]<args.workers_per_label and available \
                        and time.time()>=paused[label] and label not in stopped:
                    job=queues[label].pop(0);core=available.pop(0)
                    future=executor.submit(run_one,root,job,pool,root/'rules',
                        root/'adapters'/f'{label}.json',core,deadline)
                    workers[future]=(job,core);counts[label]+=1
            dump(root/'queue_status.json',dict(updated_unix=time.time(),deadline_unix=deadline,
                active=[j for j,_ in workers.values()],pending={k:len(v) for k,v in queues.items()},
                paused_until=dict(paused),stopped_labels=sorted(stopped)))
            if not workers:
                if all(not q or label in stopped for label,q in queues.items()):break
                time.sleep(20);continue
            done,_=wait(workers,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                job,core=workers.pop(future);available.append(core)
                try:status=future.result()
                except Exception as exc:status='worker_error:'+type(exc).__name__
                print(json.dumps(dict(id=job['id'],status=status,time=time.time())),flush=True)
                if status=='provider_rate_limit':
                    retries[job['id']]+=1;paused[job['label']]=time.time()+min(3600,600*2**(retries[job['id']]-1))
                    if retries[job['id']]<=4:queues[job['label']].insert(0,job)
                elif status=='provider_authentication':stopped.add(job['label'])
            if done:
                stats=summarize(root,jobs);print('STATUS',dict(stats),flush=True)
    stats=summarize(root,jobs)
    dump(root/'queue_finished.json',dict(ended_unix=time.time(),statuses=dict(stats),
        pending={k:len(v) for k,v in queues.items()},stopped_labels=sorted(stopped)))


if __name__=='__main__':main()
