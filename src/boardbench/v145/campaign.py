"""Three result-driven official-duo rounds inside a new immutable eight-hour window."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from boardbench.artifacts.store import digest,read_json,write_json,source_files,source_id
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.preflight import reserve
from boardbench.v144.campaign import Runner as BaseRunner,retained_records
from .preflight import trial_config
from .decisions import first_round,next_round


def device_occupancy(receipts,now):
    """Union occupied intervals per physical GPU; shared slots are not 4 GPUs."""
    by_device={}
    for r in receipts:
        begin=r['started_unix'];end=begin+r.get('elapsed_seconds',max(0,now-begin))
        by_device.setdefault(r['gpu'],[]).append((begin,end))
    total=0.
    for intervals in by_device.values():
        end=None
        for a,b in sorted(intervals):
            total+=max(0,b-max(a,end if end is not None else a));end=max(b,end if end is not None else b)
    return total


class Runner(BaseRunner):
    def report(self):
        super().report();cost=read_json(self.root/'cost.json')
        cost.update(logical_gpu_slot_seconds=cost['physical_gpu_seconds'],
            physical_gpu_seconds=device_occupancy(self.receipts,time.time()),
            physical_gpu_count=len(set(self.c['gpu_uuids'])),logical_gpu_slots=len(self.c['gpu_uuids']),
            note='Physical GPU occupancy is union of slot intervals per UUID, not utilization. CPU includes allocated slots; setup editor/test CPU is not precisely measured. Original deadline includes setup and retries.')
        write_json(self.root/'cost.json',cost)


def training_spec(root,path,target):
    return dict(name=target.parent.name+'_'+target.name,out=str(target),training=True,complete=str(target/'done.json'),
        args=['-m','boardbench.v145.training','--config',str(path),'--out',str(target)])


def evaluation_spec(root,record,seeds,stage,deadline):
    target=root/stage/record['name'];path=root/'configs'/(stage+'_'+record['name']+'.json')
    spec=dict(config=record['config_path'],model=record['policy'],model_sha256=record['sha256'],seeds=seeds,
        stage='confirmation' if stage=='confirmation' else 'development',deadline_unix=deadline)
    if path.exists() and read_json(path)!=spec:raise ValueError('immutable evaluation changed')
    write_json(path,spec)
    return dict(name=stage+'_'+record['name'],out=str(target),
        complete=str(target/('summary.json' if stage=='confirmation' else 'result.json')),
        args=['-m','boardbench.v145.evaluate','--spec',str(path),'--out',str(target)])


def execute(repo,root):
    c=read_json(root/'resolved_config.json');launch=read_json(root/'launch.json')
    if os.uname().nodename!=c['host'] or source_id()!=launch['source_id']:raise ValueError('authorized host/frozen source required')
    if digest(root/'resolved_config.json')!=launch['resolved_config_sha256']:raise ValueError('immutable config/deadline changed')
    if (root/'summary.json').exists() or c['deadline_unix']-time.time()<120:raise ValueError('completed/expired campaign cannot extend')
    lock=reserve(repo,c);os.sched_setaffinity(0,{launch['cpus'][-1]});runner=Runner(repo,root)
    try:
        gates=[]
        for i,arm in enumerate(first_round()):
            gate_arm={**arm,'progress_scale':1.,'mode':'mixed'} if i==3 else arm
            cfg=trial_config(root/'source',c,gate_arm)
            cfg.update(deadline_unix=c['deadline_unix'],replay_directory=str(root/f'preflight_slot{i}/replay'))
            path=root/'configs'/f'gate{i}.json';write_json(path,cfg);target=root/f'preflight_slot{i}'
            gates.append(dict(name=f'preflight_slot{i}',out=str(target),complete=str(target/'result.json'),
                args=['-m','boardbench.v145.preflight','--config',str(path),'--out',str(target)]))
        runner.run(gates,min(c['deadline_unix'],time.time()+900),'cuda_preflight')
        for i in range(4):
            gate=read_json(root/f'preflight_slot{i}/result.json')
            if not gate['passed'] or gate['source_id']!=launch['source_id']:raise ValueError('actual CUDA gate mismatch')
        schedule_path=root/'schedule.json'
        if schedule_path.exists():schedule=read_json(schedule_path)
        else:
            available=c['deadline_unix']-c['final_reserve_seconds']-time.time()
            if available<2700:raise ValueError('insufficient three-round time; no budget renewal')
            begin=time.time();schedule=dict(round_deadlines=[begin+available*(i+1)/3 for i in range(3)],deadline_unix=c['deadline_unix'])
            write_json(schedule_path,schedule)
        for index,until in enumerate(schedule['round_deadlines']):
            decision_path=root/'decisions'/f'round_{index+1}.json'
            if decision_path.exists():arms=read_json(decision_path)['arms']
            else:
                if index==0:arms=first_round();decision=dict(rationale='paired stationary-opponent, plan-credit and own-replay contrasts',uses_confirmation=False)
                else:
                    previous=[r for r in retained_records(root) if r['name'].startswith(f'r{index}_')]
                    arms,decision=next_round(previous,index)
                write_json(decision_path,dict(arms=arms,decision=decision,created_unix=time.time(),deadline_unix=until))
            jobs=[]
            for arm in arms:
                target=root/'rounds'/f'r{index+1}'/arm['name'];path=root/'configs'/f'r{index+1}_{arm["name"]}.json'
                if not path.exists():
                    cfg=trial_config(root/'source',c,arm,index)
                    cfg.update(deadline_unix=until-20,max_seconds=max(0,until-time.time()),replay_directory=str(target/'replay'))
                    write_json(path,cfg)
                jobs.append(training_spec(root,path,target))
            try:runner.run(jobs,until,f'round_{index+1}')
            except RuntimeError as exc:
                completed=[r for r in retained_records(root) if r['name'].startswith(f'r{index+1}_')]
                if len(completed)<2 or not any(r['config']['name']=='control' for r in completed):raise
                write_json(root/'decisions'/f'partial_round_{index+1}.json',dict(error=repr(exc),
                    retained=[r['name'] for r in completed],note='Failed arms excluded, not assigned scores; control retained.'))
            records=retained_records(root)
            write_json(root/'retained.json',[{k:v for k,v in r.items() if k not in ('evaluation','config')} for r in records])
        records=retained_records(root);first=records[0]
        records.append(dict(name='initial_parent',config_path=first['config_path'],policy=c['warm_start'],sha256=c['warm_start_sha256']))
        seeds=list(range(245100000,245100000+c['selection_environments']))
        selection_until=c['deadline_unix']-c['confirmation_reserve_seconds']
        runner.run([evaluation_spec(root,r,seeds,'selection',selection_until) for r in records],selection_until,'final_development_selection')
        selected_path=root/'selection.json'
        if selected_path.exists():selected=read_json(selected_path)['models']
        else:
            ranked=sorted(records,key=lambda r:read_json(root/'selection'/r['name']/'result.json')['score_mean'],reverse=True)
            selected=ranked[:3]
            controls=[r for r in ranked if r.get('config',{}).get('name')=='control']
            extra=next((r for r in controls if r['name'] not in {s['name'] for s in selected}),None)
            if extra is None:extra=next(r for r in ranked if r['name'] not in {s['name'] for s in selected})
            selected.append(extra);selected=[{k:v for k,v in r.items() if k not in ('config','evaluation')} for r in selected]
            write_json(selected_path,dict(models=selected,uses_confirmation=False,
                ranking=[dict(name=r['name'],score=read_json(root/'selection'/r['name']/'result.json')['score_mean']) for r in ranked]))
        seeds=list(range(345000000,345000000+c['confirmation_environments']))
        runner.run([evaluation_spec(root,r,seeds,'confirmation',c['deadline_unix']-15) for r in selected],
            c['deadline_unix']-15,'independent_confirmation')
        summary=dict(status='completed',selection_uses_confirmation=False,started_unix=c['started_unix'],deadline_unix=c['deadline_unix'],
            completed_unix=time.time(),selected=[dict(name=r['name'],**read_json(root/'confirmation'/r['name']/'summary.json')) for r in selected],
            rounds=[read_json(root/'decisions'/f'round_{i+1}.json') for i in range(3)])
        write_json(root/'summary.json',summary)
        lines=['# V145 official-duo score/animal report','',
            'Models selected by fresh development only; confirmation512, both seats, two fixed opponents.','',
            '| Policy | Score | Animals | Turn10 animals (early-terminal capped) | Gain vs V144 parent (95% CI) |',
            '|---|---:|---:|---:|---|']
        for row in summary['selected']:
            m=row['result'];d=row['paired_vs_parent']['metrics']['score']
            lines.append(f"| {row['name']} | {m['score_mean']:.3f} | {m['animal_mean']:.3f} | {m['prefix10_animal_mean']:.3f} | {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
        lines+=['','Official rules and raw scoring unchanged. Potential feedback conserves total own return; SIL is separate off-policy learning.',
            'No new model is guaranteed to beat the parent. All policies/traces retained; no deployment pointers changed.',
            'See cost.json for process occupancy, decisions/ for adaptive choices and partial-round failures.']
        (root/'report.md').write_text('\n'.join(lines)+'\n');runner.phase='completed';runner.report()
    except BaseException as exc:
        runner.phase='interrupted' if isinstance(exc,InterruptedError) else 'failed'
        write_json(root/(runner.phase+'.json'),dict(error=repr(exc),time=time.time()));raise
    finally:runner.close();lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--out',required=True)
    p.add_argument('--config');p.add_argument('--execute',action='store_true');a=p.parse_args()
    repo=Path(a.repo).resolve();root=Path(a.out).resolve()
    if a.execute:return execute(repo,root)
    c=read_json(a.config);physical=physical_cpus()
    if (os.uname().nodename!=c['host'] or len(physical)//2<57 or len(c['gpu_uuids'])!=4 or c['workers']!=13
            or not set(c['gpu_uuids'])<=set(c.get('authorized_gpu_uuids',c['gpu_uuids']))
            or (len(set(c['gpu_uuids']))<4 and not c.get('shared_device_slots'))
            or c['deadline_unix']-c['started_unix']!=28800 or time.time()>=c['deadline_unix']):
        raise ValueError('a226 four authorized GPU slots,57 cores<=half, immutable eight-hour window required')
    root.mkdir(parents=True,exist_ok=False)
    for name in ('logs','configs','rounds','decisions','selection','confirmation'):(root/name).mkdir()
    for relative in source_files(repo):
        target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,target)
    resolved={**c,**{k:str(repo/c[k]) for k in ('parent','warm_start','native_manifest')}}
    resolved['evaluation_opponents']=[{**r,'path':str(repo/r['path'])} for r in c['evaluation_opponents']]
    write_json(root/'config.json',c);write_json(root/'resolved_config.json',resolved)
    write_json(root/'launch.json',dict(source_id=source_id(repo),cpus=physical[:57],resolved_config_sha256=digest(root/'resolved_config.json'),
        source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()))
    rc=subprocess.run([sys.executable,'-u','-m','boardbench.v145.campaign','--repo',str(repo),'--out',str(root),'--execute'],
        env={**os.environ,'PYTHONPATH':str(root/'source/src')}).returncode
    if rc:raise SystemExit(rc)


if __name__=='__main__':main()
