"""Eight independent a218 jobs, frozen source, shared CPU lock and bounded tasks."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from boardbench.artifacts.store import digest, read_json, source_files, source_id, write_json
from boardbench.v12.budget import kill_tree
from boardbench.v122.campaign import physical_cpus


def build_jobs(config,repo):
    repo=Path(repo)
    dataset=repo/config['dataset'];manifest=repo/config['split_manifest']
    if digest(dataset)!=config['dataset_sha256'] or digest(manifest)!=config['split_sha256']:
        raise ValueError('pinned dataset or split mismatch')
    sets=read_json(manifest)['sets'];jobs=[]
    for kind in ('bc','soft','anchor','random'):
        for replicate in range(2):
            suffix='ab'[replicate]
            base={**config['training'],'research_version':'1.2.7','hidden':256,'seed':12701+replicate,
                  'reward_scale':150.,'dataset':str(dataset),'dataset_sha256':config['dataset_sha256'],
                  'split_manifest':str(manifest),'split_sha256':config['split_sha256'],
                  'curve_seeds':sets['development'][:32],'final_seeds':sets['development'][32:96],
                  'excluded_train_seeds':sets['train'][2176:2192]+sets['train'][2208:2224],
                  'bc_weight':.05 if kind=='anchor' else 0.,'bc_decay_actions':500000,
                  'initial_checkpoint':None,'initial_sha256':None,'arm':kind,'replicate':replicate}
            if kind!='random':
                name=('q256_' if kind=='soft' else 'bc256_')+suffix
                p=repo/config['parents']/name/'best.npz'
                base.update(initial_checkpoint=str(p),initial_sha256=digest(p))
            jobs.append((f'{kind}_{suffix}',base))
    return jobs


def validate_allocation(cpus,gpus,cores_per_job=15,jobs_per_gpu=1,job_count=8):
    if len(cpus)>128 or len(set(cpus))!=len(cpus): raise ValueError('CPU aggregate cap or duplicates')
    if job_count<1 or len(gpus)*jobs_per_gpu!=job_count or len(set(gpus))!=len(gpus): raise ValueError('distinct jobs and unique GPU list required')
    if cores_per_job<1 or jobs_per_gpu<1:raise ValueError('positive allocation required')
    if job_count*cores_per_job+1>len(cpus): raise ValueError('learners/workers/controller exceed CPU allocation')


def physical_gpu_seconds(attempts,now):
    """Union wall intervals per device, not sum of shared process lifetimes."""
    groups={}
    for a in attempts:
        end=a['started_unix']+a['elapsed_seconds'] if 'elapsed_seconds' in a else now
        groups.setdefault(a['gpu'],[]).append((a['started_unix'],max(a['started_unix'],end)))
    total=0.
    for intervals in groups.values():
        left=right=None
        for start,end in sorted(intervals):
            if left is None:left,right=start,end
            elif start<=right:right=max(right,end)
            else:total+=right-left;left,right=start,end
        if left is not None:total+=right-left
    return total


def attach_continuations(config,repo,jobs):
    if not config.get('continuation_root'):return
    parent=Path(repo)/config['continuation_root'];state=read_json(parent/'state.json')
    allowed={'finished'}
    if config.get('allow_interrupted_parent'):allowed.add('interrupted')
    if state['phase'] not in allowed:raise ValueError('parent campaign must be quiescent')
    for name,trial in jobs:
        path=parent/'trials'/name/'final.pt';summary=read_json(path.parent/'summary.json')
        statuses={'completed'}
        if config.get('allow_interrupted_parent'):statuses.add('interrupted')
        if summary['status'] not in statuses:raise ValueError('parent trial incomplete')
        meta=read_json(path.with_suffix('.json'))
        if digest(path)!=meta['sha256'] or summary['source_id']!=meta['source_id']:
            raise ValueError('parent final checkpoint/source mismatch')
        if config.get('carry_stage_budget'):
            trial['max_seconds']-=summary['wall_seconds']
            if trial['max_seconds']<=0:raise ValueError('parent exhausted stage budget')
        trial['continuation']={'checkpoint':str(path),'sha256':meta['sha256'],'source_id':meta['source_id']}


def report(root,state):
    root=Path(root);curves={};rows=[]
    for name in state['jobs']:
        p=root/'trials'/name
        progress=read_json(p/'progress.json') if (p/'progress.json').exists() else {}
        points=[]
        for f in (p/'evaluations').glob('*.json'):
            m=read_json(f)
            if not m['label'].startswith('final'): points.append({'actions':m['ppo_actions'],'score':m['mean_score'],
                'animal_score':m['animal_mean'],'path':str(f)})
        points.sort(key=lambda x:x['actions']);curves[name]=points
        latest=points[-1]['score'] if points else None
        rows.append({'trial':name,'phase':progress.get('phase','pending'),'actions':progress.get('ppo_actions',0),
                     'initial':points[0]['score'] if points else None,'latest_curve':latest,
                     'best_curve':max(x['score'] for x in points) if points else None,
                     'heartbeat_age':time.time()-progress['updated_unix'] if progress else None})
    write_json(root/'curves.json',curves);write_json(root/'results.json',rows)
    lines=['# Harmonies V1.2.7 在线 PPO','',f"状态：{state['phase']}",
           '纯模型、完整自然终局、GPU批量推理；曲线为固定32开发环境。训练分数是随机策略，不与贪心开发曲线混用。',
           '四组各两个种子：BC初始化、软标签初始化、BC初始化+衰减模仿约束、随机初始化。初始冻结策略作为逐环境对照。',
           '', '|任务|状态|PPO动作|初始|最近开发均分|最好开发均分|','|---|---|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{v:.2f}'
    for r in rows: lines.append(f"|{r['trial']}|{r['phase']}|{r['actions']}|{fmt(r['initial'])}|{fmt(r['latest_curve'])}|{fmt(r['best_curve'])}|")
    specification=read_json(root/'config.json');target=specification['training']['target_actions']
    lines+=['',f"资源记账：{state.get('cost',{})}",'历史费用单独保留，不重复计入。失败、预检、评估和清理均计费；同卡并发按物理GPU时间合并。',
            f'开发均分用于筛选，不代表正式测试或已超越高预算搜索。每任务累计目标{target}PPO动作；本阶段限时后保存停止。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    colors=['#2563eb','#60a5fa','#dc2626','#f87171','#059669','#34d399','#7c3aed','#a78bfa']
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="470" viewBox="0 0 960 470">',
         '<rect width="960" height="470" fill="white"/>',
         '<text x="55" y="25" font-family="sans-serif" font-size="16">Model-only greedy score: common 32 development games</text>']
    for score in (0,30,60,90,120,150):
        y=420-score*2.4;svg.append(f'<path d="M55 {y} H750" stroke="#ddd"/><text x="15" y="{y+4}" font-size="12">{score}</text>')
    for i,(name,pts) in enumerate(curves.items()):
        coords=' '.join(f"{55+min(p['actions']/target,1)*695},{420-p['score']*2.4}" for p in pts)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')
        svg.append(f'<text x="775" y="{60+24*i}" font-family="sans-serif" fill="{colors[i]}">{name}</text>')
        for p in pts: svg.append(f'<circle cx="{55+min(p["actions"]/target,1)*695}" cy="{420-p["score"]*2.4}" r="3" fill="{colors[i]}"/>')
    svg+=['<text x="55" y="450">0</text>',f'<text x="610" y="450">{target/1e6:g}M PPO actions</text>','</svg>']
    (root/'curves.svg').write_text('\n'.join(svg))


def execute(root,repo,*,training_module='boardbench.v127.training',reporter=report):
    root,repo=Path(root),Path(repo)
    launch=read_json(root/'launch.json');config=read_json(root/'config.json')
    if os.uname().nodename!='a218.cluster.helixon.com' or source_id()!=launch['source_id']:
        raise ValueError('run on a218 using this run frozen source')
    cores_per_job=config['training']['workers']+1
    jobs=read_json(root/'jobs.json')
    validate_allocation(launch['cpus'],config['gpu_uuids'],cores_per_job,config.get('jobs_per_gpu',1),len(jobs))
    os.sched_setaffinity(0,{launch['cpus'][-1]})
    lock=(repo/'outputs/v122'/f'{os.uname().nodename}.cpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read_json(root/'state.json') if (root/'state.json').exists() else {'jobs':{},'attempts':[]}
    active={};stopping=[False];started=time.monotonic();prior_controller=state.get('controller_seconds',0.)
    prior={sig:signal.signal(sig,lambda *_:stopping.__setitem__(0,True)) for sig in (signal.SIGINT,signal.SIGTERM)}
    def publish(phase):
        controller=prior_controller+time.monotonic()-started
        cpu=config.get('preflight_cpu_core_seconds',0)+controller
        gpu=config.get('preflight_gpu_seconds',0)+physical_gpu_seconds(state['attempts'],time.time())
        gpu_process=0.
        for attempt in state['attempts']:
            elapsed=attempt.get('elapsed_seconds',max(0.,time.time()-attempt['started_unix']))
            cpu+=len(attempt['cpus'])*elapsed;gpu_process+=elapsed
        state.update(phase=phase,pid=os.getpid(),updated_unix=time.time(),controller_seconds=controller,
            cost={'cpu_core_hours':cpu/3600,'gpu_hours':gpu/3600,'gpu_process_hours':gpu_process/3600,
                  'basis':'CPU affinity reservation; physical GPU interval union; process hours separate; all attempts and preflight'})
        write_json(root/'state.json',state);reporter(root,state)
    try:
        probe=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        inventory={v[0].strip():(int(v[1]),int(v[2])) for v in (line.split(',') for line in probe.splitlines())}
        write_json(root/'gpu_probe.json',{'unix':time.time(),'inventory':inventory})
        if any(g not in inventory or inventory[g][0]>=256 or inventory[g][1]>=5 for g in config['gpu_uuids']):
            raise RuntimeError('GPU busy or missing; do not preempt unrelated work')
        if os.getloadavg()[0]>64: raise RuntimeError('CPU load too high for new allocation')
        for i,(name,trial) in enumerate(jobs):
            state['jobs'].setdefault(name,{'status':'pending'})
            p=root/'trials'/name
            if (p/'summary.json').exists() and read_json(p/'summary.json')['status'] in ('completed','time_limit'):
                state['jobs'][name]['status']=read_json(p/'summary.json')['status'];continue
            p.mkdir(parents=True,exist_ok=True);write_json(p/'config.json',trial)
            previous_wall=0.
            if (p/'progress.json').exists(): previous_wall=read_json(p/'progress.json').get('wall_seconds',0.)
            if previous_wall>=trial['max_seconds']:
                state['jobs'][name]['status']='time_limit';continue
            env={**os.environ,'PYTHONPATH':str(root/'source/src'),'PYTHONNOUSERSITE':'1',
                 'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
                 'CUDA_VISIBLE_DEVICES':config['gpu_uuids'][i%len(config['gpu_uuids'])]}
            affinity=launch['cpus'][i*cores_per_job:(i+1)*cores_per_job]
            log=(root/'logs'/f'{name}.log').open('a')
            child=subprocess.Popen(['taskset','-c',','.join(map(str,affinity)),sys.executable,'-u','-m',
                training_module,'--config',str(p/'config.json'),'--out',str(p),'--device','cuda'],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            attempt={'trial':name,'pid':child.pid,'started_unix':time.time(),'gpu':env['CUDA_VISIBLE_DEVICES'],'cpus':affinity}
            state['attempts'].append(attempt);state['jobs'][name]={'status':'running','pid':child.pid}
            active[name]={'child':child,'log':log,'attempt':attempt,'deadline':time.monotonic()+trial['max_seconds']-previous_wall}
            publish('running')
        while active:
            for name,item in list(active.items()):
                child=item['child']
                if (stopping[0] or time.monotonic()>=item['deadline']) and 'terminated' not in item:
                    child.terminate();item['terminated']=time.monotonic()
                if child.poll() is None and time.monotonic()-item.get('terminated',time.monotonic())>90:
                    kill_tree(child.pid)
                if child.poll() is not None:
                    item['attempt'].update(returncode=child.returncode,elapsed_seconds=time.time()-item['attempt']['started_unix'])
                    summary=root/'trials'/name/'summary.json'
                    state['jobs'][name]['status']=read_json(summary)['status'] if child.returncode==0 and summary.exists() else 'failed'
                    item['log'].close();del active[name]
            publish('stopping' if stopping[0] else 'running')
            if active: time.sleep(10)
        publish('interrupted' if stopping[0] else 'finished')
    except BaseException as exc:
        state['error']=repr(exc);publish('failed');raise
    finally:
        for item in active.values():
            kill_tree(item['child'].pid);item['child'].wait();item['log'].close()
            item['attempt']['elapsed_seconds']=time.time()-item['attempt']['started_unix']
        if active: publish('failed')
        lock.close()
        for sig,handler in prior.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('launch','resume','run','report'))
    p.add_argument('--repo',required=True);p.add_argument('--root',required=True);p.add_argument('--config')
    a=p.parse_args();repo=Path(a.repo).resolve();root=Path(a.root).resolve()
    if a.mode=='report':report(root,read_json(root/'state.json'));return
    if a.mode in ('launch','resume'):
        if os.uname().nodename!='a218.cluster.helixon.com':raise ValueError('a218 only')
        if not any(root.is_relative_to(repo/'outputs'/v) for v in ('v127','v128','v129')): raise ValueError('project PPO output required')
        if a.mode=='launch':
            if root.exists(): raise ValueError('refusing run overwrite')
            config=read_json(a.config);cpus=physical_cpus()[:128]
            validate_allocation(cpus,config['gpu_uuids'],config['training']['workers']+1,config.get('jobs_per_gpu',1))
            jobs=build_jobs(config,repo)
            attach_continuations(config,repo,jobs)
            root.mkdir(parents=True);(root/'logs').mkdir()
            write_json(root/'config.json',config);write_json(root/'jobs.json',jobs)
            for relative in source_files(repo):
                dest=root/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,dest)
            write_json(root/'launch.json',{'host':os.uname().nodename,'started_unix':time.time(),
                'source_id':source_id(root/'source'),'cpus':cpus,'git_head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()})
        os.environ.update(PYTHONPATH=str(root/'source/src'),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        os.execv(sys.executable,[sys.executable,'-u','-m','boardbench.v127.launch','run','--repo',str(repo),'--root',str(root)])
    execute(root,repo)


if __name__=='__main__': main()
