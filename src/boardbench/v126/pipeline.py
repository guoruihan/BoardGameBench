"""A bounded, recoverable teacher -> student -> search -> DAgger experiment.

Every CPU campaign uses the existing lock/frozen snapshot/deadline supervisor.
The outer process never kills unrelated jobs or modifies an in-flight policy.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from boardbench.artifacts.store import read_json, write_json, digest, source_id, source_files
from boardbench.v12.budget import kill_tree
from boardbench.v122.campaign import physical_cpus, make_jobs

MODELS=[('bc128_a',128,0.,12601),('bc128_b',128,0.,12602),
        ('q128_a',128,.5,12601),('q128_b',128,.5,12602),('bc256_a',256,0.,12601)]


def validate_config(config):
    groups=[config[k] for k in ('train_indices','validation_indices','student_selection_indices','dagger_indices')]
    if any(not g for g in groups) or len(set(sum(groups,[])))!=sum(map(len,groups)):
        raise ValueError('training partitions overlap or are empty')
    if not config['pilot_indices'] or not config['fresh_indices'] or set(config['pilot_indices']) & set(config['fresh_indices']):
        raise ValueError('pilot and fresh cohorts overlap or are empty')
    if config.get('calibration_only'):
        if config['game_seconds']>1 or config['campaign_cores']>4 or config['gpu_uuids']:
            raise ValueError('calibration must be small and CPU-only')
        if set(sum(groups,[])) & set(config['pilot_indices']+config['fresh_indices']):
            raise ValueError('calibration train cohorts overlap')
    elif config['game_seconds']!=900 or config['campaign_cores']>101:
        raise ValueError('research requires900s and at most101 campaign cores')
    specs=config.get('model_specs',MODELS)
    if (not 1<=len(specs)<=8 or any(len(s)!=4 for s in specs) or
        len({s[0] for s in specs})!=len(specs)):
        raise ValueError('one to eight uniquely named model specifications required')
    if len(set(config['gpu_uuids']))!=len(config['gpu_uuids']):
        raise ValueError('duplicate GPU UUID')
    if config.get('require_cuda') and len(config['gpu_uuids'])<len(specs):
        raise ValueError('one authorized GPU per model required; no silent CPU fallback')


def validate_host(config, hostname):
    if config.get('calibration_only'):
        return
    profile=config.get('resource_profile','b35_reserve16_v1')
    if profile=='a218_128_v1' and hostname=='a218.cluster.helixon.com':
        return
    if profile=='b35_reserve16_v1' and hostname=='b35.cluster.helixon.com' and not config['gpu_uuids']:
        return
    raise ValueError('resource policy: a218 for mixed work; b35 CPU-only')


def verify_external_teacher(path, config, manifest_path):
    """Reuse completed work only with pinned provenance and exact train cohort."""
    path=Path(path)
    state=read_json(path/'status.json');allocation=read_json(path/'allocation.json')
    plan=read_json(path/'plan.json');manifest=read_json(manifest_path)
    if (state['phase']!='completed' or state.get('active_count',1) or
        not (path/'watchdog_result.json').exists()):
        raise ValueError('external teacher must be finalized and inactive')
    if (source_id(path/'source')!=allocation['source_id'] or
        allocation['source_id']!=config['external_teacher_source_id'] or
        digest(path/'plan.json')!=allocation['plan_sha256'] or
        allocation['plan_sha256']!=config['external_teacher_plan_sha256'] or
        digest(path/'splits.json')!=allocation['splits_sha256'] or
        digest(manifest_path)!=allocation['splits_sha256']):
        raise ValueError('external teacher source/plan/split hash mismatch')
    expected=[manifest['sets']['train'][i] for i in config['train_indices']+config['validation_indices']]
    jobs=read_json(path/'jobs.json')
    params=arm('teacher',seconds=config['game_seconds'])['params']
    if (plan['split']!='train_calibration' or sorted(j['environment_seed'] for j in jobs)!=sorted(expected) or
        any(j['params']!=params or j['arm']!='teacher' or j['search_seed']!=12620261020 for j in jobs)):
        raise ValueError('external teacher cohort or policy differs')
    if jobs!=make_jobs(plan):
        raise ValueError('external teacher jobs differ from frozen plan')
    if len(list((path/'resolved').glob('*.json')))!=len(jobs):
        raise ValueError('missing external teacher results')
    for p in (path/'resolved').glob('*.json'):
        r=read_json(p)
        if r['status']!='completed' or not r['natural_terminal']:
            raise ValueError('external teacher includes failed or censored games')
    return path


def arm(name,variant='mixed_roots',model=None,seconds=900):
    params={'mode':'student' if variant=='student' else 'timed','continuation':variant,'seconds':seconds}
    if model:
        params.update(checkpoint=model['checkpoint'],checkpoint_sha256=model['checkpoint_sha256'])
    return {'id':name,'method':'terminal','multiplier':1,'timeout_seconds':seconds+120,'params':params}


class Pipeline:
    def __init__(self,root,repo):
        self.root=Path(root).resolve();self.repo=Path(repo).resolve()
        self.config=read_json(self.root/'config.json')
        validate_config(self.config)
        self.manifest=read_json(self.repo/self.config['split_manifest'])
        self.cpus=read_json(self.root/'launch.json')['physical_cpus']
        if len(self.cpus)>128 or len(set(self.cpus))!=len(self.cpus):
            raise ValueError('per-node aggregate CPU allocation exceeds128 or duplicates cores')
        self.env={**os.environ,'PYTHONPATH':str(self.root/'source/src'),'PYTHONNOUSERSITE':'1',
                  'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
        self.state=read_json(self.root/'state.json') if (self.root/'state.json').exists() else {'stages':{},'decisions':[]}
        self.children=[]

    def publish(self,phase):
        self.state.update(phase=phase,updated_unix=time.time(),pid=os.getpid())
        if phase in ('completed','stopped_with_error'):
            self.state.setdefault('finished_unix',time.time())
        write_json(self.root/'state.json',self.state)
        lines=['# Harmonies V1.2.6 搜索指导学习','',f'状态：{phase}',
               '拟合与训练内检查点选择只用train；方案筛选用development。所有得分为自然终局，不代表最优或正式测试。',
               f"搜索每局累计{self.config['game_seconds']}秒，纯学生单独分组；每环境先平均搜索重复。",'']
        if self.config.get('calibration_only'):
            lines+=['仅基础设施校准，低预算分数不可用于正式方法选择。','']
        charge=0.;gpu=0.
        for name,stage in self.state['stages'].items():
            lines += [f'## {name}', '',f"状态：{stage['status']}；路径：{stage['path']}",'']
            p=Path(stage['path'])
            if (p/'summary.json').exists() and stage.get('kind')=='campaign':
                s=read_json(p/'summary.json')
                lines += [f"完整配对环境：{s.get('balanced_panel_size',0)}",'',
                          '|策略|总分|动物分|决策秒|','|---|---:|---:|---:|']
                for name,c in s.get('balanced_panel',{}).items():
                    fmt=lambda v:'—' if v is None else f'{v:.2f}'
                    lines.append('|'+name+'|'+'|'.join(fmt(c[k]) for k in ('mean','animal_mean','decision_seconds'))+'|')
                lines.append('')
        # All attempts, not just successful latest stages, remain charged.
        for p in (self.repo/'outputs/v126').glob(self.root.name+'_*'):
            if (p/'watchdog_result.json').exists():
                charge+=read_json(p/'watchdog_result.json')['allocated_cpu_core_seconds']
            elif (p/'allocation.json').exists():
                a=read_json(p/'allocation.json');charge+=len(a['cpu_affinity'])*max(0,time.time()-a['started_unix'])
        for p in (self.root/'receipts').glob('*.json'):
            r=read_json(p);charge+=r.get('cpu_core_seconds',0);gpu+=r.get('gpu_seconds',0)
        if 'active_training' in self.state:
            r=self.state['active_training'];elapsed=max(0.,time.time()-r['started_unix'])
            charge+=r['cpu_cores']*elapsed;gpu+=r['gpu_count']*elapsed
        launch=read_json(self.root/'launch.json')
        charge+=self.state.get('finished_unix',time.time())-launch['started_unix'] # one orchestration core
        charge+=self.config.get('preflight_cpu_core_seconds',0)
        self.state['cost']={'cpu_core_hours':charge/3600,'gpu_hours':gpu/3600,
                            'basis':'reserved CPU time including failed attempts, preflight and one orchestrator; GPU process occupancy'}
        lines+=['## 决策记录','']+[f'- {d}' for d in self.state['decisions']]
        lines+=['',f'累计记账约 {charge/3600:.2f} CPU核时，{gpu/3600:.3f} GPU小时。',
                'GPU探测和训练设备记录见 gpu_probe.json、各训练config/summary；CPU回退不冒充GPU训练。']
        if 'error' in self.state: lines+=['',self.state['error']]
        write_json(self.root/'state.json',self.state)
        temporary=self.root/'REPORT.md.tmp';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(self.root/'REPORT.md')

    def wait_ready(self,dependency=None):
        deadline=time.monotonic()+self.config.get('dependency_wait_seconds',7200)
        while True:
            ready=True
            if dependency:
                p=Path(dependency)
                ready=(p/'watchdog_result.json').exists() and read_json(p/'status.json').get('active_count',1)==0
            if ready and os.getloadavg()[0]<=len(self.cpus)*.5:
                return
            self.publish('waiting_for_prior_campaign_or_cpu_load')
            if time.monotonic()>deadline: raise RuntimeError('bounded resource wait expired; no jobs stopped')
            time.sleep(15)

    def run(self,command,log,timeout,cpus=None,extra_env=None):
        command=['taskset','-c',','.join(map(str,cpus or self.cpus))]+command
        log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
        start=time.monotonic()
        with log.open('a') as stream:
            p=subprocess.Popen(command,env={**self.env,**(extra_env or {})},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            self.children.append(p)
            try:
                while p.poll() is None:
                    self.publish('running')
                    if time.monotonic()-start>=timeout:
                        raise TimeoutError(f'stage timeout: {log}')
                    time.sleep(5)
                if p.returncode: raise RuntimeError(f'child failed ({p.returncode}): {log}')
            finally:
                if p.poll() is None:
                    # Let the campaign's independent watchdog reconcile attempts.
                    p.terminate()
                    try: p.wait(timeout=10)
                    except subprocess.TimeoutExpired: kill_tree(p.pid);p.wait(timeout=5)
                self.children.remove(p)

    def campaign(self,name,indices,arms,*,split='development',repeats=2,seconds=2400):
        previous=self.state['stages'].get(name)
        if previous and previous['status']=='completed':
            path=Path(previous['path'])
            if read_json(path/'status.json')['phase']!='completed': raise ValueError('completed stage corrupted')
            return path
        self.wait_ready()
        count=min(self.config.get('campaign_cores',101),len(indices)*repeats*len(arms)+1)
        spec={'research_round':name,'split':split,'environment_indices':indices,
              'split_manifest':self.config['split_manifest'],'panels':sorted(set([min(4,len(indices)),len(indices)])),
              'search_seeds':[12620261020+i for i in range(repeats)],'references':['mixed_roots','habitat'],
              'resource_profile':'default_half' if self.config.get('calibration_only') else self.config.get('resource_profile','b35_reserve16_v1'),
              'cpu_cores':count,'task_seconds':seconds,'arms':arms}
        if self.config.get('calibration_only'): spec['split']='train_calibration'
        attempt=1
        if previous:
            parent=Path(previous['path'])
            if parent.exists() and not (parent/'watchdog_result.json').exists():
                raise RuntimeError('prior stage not finalized; inspect live process before resume')
            if parent.exists(): spec['resume_from']=str(parent)
            attempt=previous.get('attempt',1)+1
        path=self.repo/'outputs/v126'/f'{self.root.name}_{name}_a{attempt}'
        config=self.root/'plans'/f'{name}_a{attempt}.json';write_json(config,spec)
        self.state['stages'][name]={'status':'running','path':str(path),'kind':'campaign','attempt':attempt}
        self.publish('running')
        self.run([sys.executable,'-u','-m','boardbench.v126.campaign','launch','--root',str(path),
                  '--repo',str(self.repo),'--plan',str(config)],self.root/'logs'/f'{name}_a{attempt}.log',seconds+90)
        s=read_json(path/'summary.json')
        if read_json(path/'status.json')['phase']!='completed' or any(c['failed_or_censored'] for c in s['cells'].values()):
            raise RuntimeError(f'campaign incomplete or failed: {path}')
        self.state['stages'][name]['status']='completed';self.publish('running')
        return path

    def dataset(self,name,runs):
        path=self.root/'datasets'/f'{name}.npz'
        if path.exists():
            if read_json(path.with_suffix('.json'))['sha256']!=digest(path): raise ValueError('dataset changed')
            return path
        validation=[self.manifest['sets']['train'][i] for i in self.config['validation_indices']]
        begin=time.monotonic()
        try:
            self.run([sys.executable,'-m','boardbench.v126.data','--runs',*map(str,runs),'--out',str(path),
                      '--validation-seeds',*map(str,validation)],self.root/'logs'/f'data_{name}.log',900,cpus=self.cpus[-5:-1])
        finally:
            write_json(self.root/'receipts'/f'data_{name}_{time.time_ns()}.json',{'cpu_core_seconds':4*(time.monotonic()-begin)})
        return path

    def probe_gpus(self):
        # Explicit UUIDs; recheck busy state and actual CUDA, no resetting devices.
        tested=[];usable=[]
        for uuid in self.config['gpu_uuids']:
            ok=False
            try:
                check=subprocess.run(['nvidia-smi','-i',uuid,'--query-gpu=memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
                reason=check.stderr.strip()
                if check.returncode==0:
                    memory,util=map(int,check.stdout.strip().split(','))
                    if memory<256 and util<5:
                        p=subprocess.run([sys.executable,'-c','import torch; print(torch.ones(2,device="cuda").sum().item())'],
                                         env={**self.env,'CUDA_VISIBLE_DEVICES':uuid},capture_output=True,text=True,timeout=30)
                        ok=p.returncode==0;reason=p.stderr[-1500:]
                    else: reason='GPU is busy; do not reserve'
            except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
                reason=f'GPU probe unavailable: {exc}'
            tested.append({'uuid':uuid,'cuda_execution_ok':ok,'reason':reason})
            if ok: usable.append(uuid)
        write_json(self.root/'gpu_probe.json',{'updated_unix':time.time(),'devices':tested})
        return usable

    def training(self,name,dataset,specs):
        out=self.root/'models'/name;out.mkdir(parents=True,exist_ok=True)
        complete=[]
        for ident,*_ in specs:
            p=out/ident/'summary.json'
            if p.exists(): complete.append(read_json(p))
        if len(complete)==len(specs): return dict(zip([s[0] for s in specs],complete))
        self.wait_ready()
        lock=(self.repo/'outputs/v122'/f'{os.uname().nodename}.cpu.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic();children=[];results={};gpus=[]
        self.state['stages'][name]={'status':'running','path':str(out),'kind':'training'}
        self.state['active_training']={'started_unix':time.time(),'cpu_cores':2*len(specs),'gpu_count':0}
        try:
            gpus=self.probe_gpus()
            if self.config.get('require_cuda') and len(gpus)<len(specs):
                raise RuntimeError('not enough idle working authorized GPUs; no CPU fallback for this run')
            self.state['active_training']['gpu_count']=min(len(gpus),len(specs))
            for i,(ident,hidden,q_weight,seed) in enumerate(specs):
                directory=out/ident
                gpu=gpus[i] if i<len(gpus) else None
                config={'dataset':str(dataset),'dataset_sha256':digest(dataset),'seed':seed,'hidden':hidden,
                        'q_weight':q_weight,'epochs':self.config.get('epochs',120),'threads':2,
                        'max_seconds':self.config.get('train_seconds',1800),'device':'cuda' if gpu else 'cpu'}
                # Resume using the original device to preserve the optimizer/RNG contract.
                if (directory/'config.json').exists():
                    config=read_json(directory/'config.json')
                    if config['device']=='cpu': gpu=None
                    elif gpu is None: raise RuntimeError('CUDA checkpoint resume requires an available CUDA device')
                path=self.root/'plans'/f'{name}_{ident}.json';write_json(path,config)
                stream=(out/f'{ident}.log').open('a')
                cpus=self.cpus[2*i:2*i+2]
                process=subprocess.Popen(['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u','-m',
                    'boardbench.v126.training','--config',str(path),'--out',str(directory)],
                    env={**self.env,'CUDA_VISIBLE_DEVICES':gpu or ''},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                self.children.append(process);children.append((ident,process,stream,time.monotonic(),gpu))
            while any(p.poll() is None for _,p,_,_,_ in children):
                self.publish('training')
                if time.monotonic()-start>self.config.get('train_seconds',1800)+180:
                    raise TimeoutError('training task deadline')
                time.sleep(5)
            for ident,p,_,_,_ in children:
                if p.returncode: raise RuntimeError(f'training failed: {out/ident}')
                results[ident]=read_json(out/ident/'summary.json')
            self.state['stages'][name]['status']='completed'
        finally:
            gpu_seconds=0.
            for ident,p,stream,started,gpu in children:
                if p.poll() is None: kill_tree(p.pid);p.wait(timeout=5)
                if gpu: gpu_seconds+=time.monotonic()-started
                stream.close();self.children.remove(p)
            write_json(self.root/'receipts'/f'{name}_{time.time_ns()}.json',{
                'cpu_core_seconds':2*len(specs)*(time.monotonic()-start),'gpu_seconds':gpu_seconds})
            self.state.pop('active_training',None)
            lock.close()
        self.publish('running');return results

    def execute(self):
        if self.state.get('phase')=='completed': return
        self.state.pop('error',None);self.state.pop('finished_unix',None)
        self.wait_ready(self.config.get('wait_for'))
        budget=self.config.get('game_seconds',900)
        if self.config.get('external_teacher'):
            teacher=verify_external_teacher(self.repo/self.config['external_teacher'],self.config,
                                            self.repo/self.config['split_manifest'])
            self.state['stages']['teacher']={'status':'completed','path':str(teacher),
                                            'kind':'campaign','external':True}
            charge=read_json(teacher/'watchdog_result.json')['allocated_cpu_core_seconds']
            parent=self.config.get('inherited_pipeline')
            if parent:
                parent=self.repo/parent
                old=read_json(parent/'state.json')
                if old['phase'] not in ('completed','stopped_with_error'):
                    raise ValueError('prior pipeline must be stopped before ownership transfer')
                if Path(old['stages']['teacher']['path']).resolve()!=teacher.resolve():
                    raise ValueError('inherited pipeline does not own this teacher')
                # Includes teacher, original preflight and old orchestration exactly once.
                charge=max(charge,old['cost']['cpu_core_hours']*3600)
            write_json(self.root/'receipts/external_teacher.json',{'cpu_core_seconds':charge,'gpu_seconds':0,
                       'origin':str(teacher),'accounting':'imported prior costs once; do not sum old and new totals'})
            self.publish('running')
        else:
            teacher=self.campaign('teacher',self.config['train_indices']+self.config['validation_indices'],
                                  [arm('teacher',seconds=budget)],split='train_calibration',repeats=1,seconds=3600)
        dataset=self.dataset('initial',[teacher])
        models=self.training('initial',dataset,self.config.get('model_specs',MODELS))
        self.campaign('pure_students',self.config['student_selection_indices'],
                      [arm(k,'student',v) for k,v in models.items()],split='train_calibration',repeats=1,seconds=900)
        arms=[arm('mixed_roots',seconds=budget),arm('habitat','habitat',seconds=budget)]
        lookup={}
        for name,model in models.items():
            for suffix,variant in [('roots','student_roots'),('rollout','student_rollout')]:
                key=f'{name}_{suffix}';lookup[key]=(name,variant)
                arms.append(arm(key,variant,model,seconds=budget))
        pilot=self.campaign('pilot',self.config['pilot_indices'],arms,seconds=self.config.get('pilot_seconds',2400))
        panel=read_json(pilot/'summary.json')['balanced_panel']
        winner=max(lookup,key=lambda k:panel[k]['mean'])
        model_name,variant=lookup[winner];model=models[model_name]
        decision=(f"Pilot selects {winner} provisionally for correction: {panel[winner]['mean']:.3f} "
                  f"vs mixed_roots {panel['mixed_roots']['mean']:.3f}; {len(self.config['pilot_indices'])} "
                  f"{'train calibration' if self.config.get('calibration_only') else 'development'} environments, not a final superiority claim. "
                  f'Selection uses {budget}s search, not pure-student score; retain all original models.')
        if decision not in self.state['decisions']: self.state['decisions'].append(decision)
        self.publish('running')
        dagger=self.campaign('dagger',self.config['dagger_indices'],[arm('student_rollin','dagger',model,seconds=budget)],
                             split='train_calibration',repeats=1,seconds=1800)
        augmented=self.dataset('augmented',[teacher,dagger])
        corrected=self.training('corrected',augmented,[('dagger_a',model['hidden'],model['q_weight'],12611),
                                                     ('dagger_b',model['hidden'],model['q_weight'],12612)])
        self.campaign('corrected_pure',self.config['student_selection_indices'],
                      [arm(k,'student',v) for k,v in corrected.items()],split='train_calibration',repeats=1,seconds=900)
        final_arms=[arm('mixed_roots',seconds=budget),arm('habitat','habitat',seconds=budget),arm('prior_best',variant,model,seconds=budget)]
        for name,m in corrected.items():
            final_arms += [arm(name+'_roots','student_roots',m,seconds=budget),arm(name+'_rollout','student_rollout',m,seconds=budget)]
        final=self.campaign('fresh_comparison',self.config['fresh_indices'],final_arms,seconds=3600)
        s=read_json(final/'summary.json');panel=s['balanced_panel']
        best=max(panel,key=lambda k:panel[k]['mean'])
        self.state['decisions'].append(f"Fresh development panel best mean: {best} {panel[best]['mean']:.3f}; "
            f"mixed_roots {panel['mixed_roots']['mean']:.3f}. See paired uncertainty in {final}/summary.json. "
            'No deployment promotion; no PPO launched automatically without evidence from this learning cycle.')
        write_json(self.root/'retained_models.json',{'initial':models,'corrected':corrected,'pilot_choice':winner})
        self.publish('completed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('launch','run','report'))
    p.add_argument('--root',required=True);p.add_argument('--repo',required=True);p.add_argument('--config')
    a=p.parse_args();root=Path(a.root).resolve();repo=Path(a.repo).resolve()
    if a.mode=='launch':
        if not root.is_relative_to(repo/'outputs/v126') or root.exists(): raise ValueError('new project V126 root required')
        config=read_json(a.config);validate_config(config)
        validate_host(config,os.uname().nodename)
        cpus=physical_cpus()[:128]
        if len(cpus)<128: raise ValueError('expected at least128 physical cores')
        orchestrator=cpus[111] if os.uname().nodename=='b35.cluster.helixon.com' else cpus[-1]
        root.mkdir(parents=True)
        shutil.copy2(a.config,root/'config.json')
        for relative in source_files(repo):
            target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,target)
        write_json(root/'launch.json',{'started_unix':time.time(),'source_id':source_id(root/'source'),
            'physical_cpus':cpus,'orchestrator_cpu':orchestrator,'host':os.uname().nodename,
            'git_head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()})
        os.sched_setaffinity(0,{orchestrator})
        os.environ.update(PYTHONPATH=str(root/'source/src'),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        os.execv(sys.executable,[sys.executable,'-u','-m','boardbench.v126.pipeline','run','--root',str(root),'--repo',str(repo)])
    if source_id()!=read_json(root/'launch.json')['source_id']:
        raise ValueError('must run the frozen pipeline source')
    if a.mode!='report':
        validate_host(read_json(root/'config.json'),os.uname().nodename)
        if read_json(root/'launch.json').get('host',os.uname().nodename)!=os.uname().nodename:
            raise ValueError('cross-host resume requires a new allocation, not old CPU affinities')
    os.sched_setaffinity(0,{read_json(root/'launch.json')['orchestrator_cpu']})
    pipeline=Pipeline(root,repo)
    if a.mode=='report': pipeline.publish(pipeline.state.get('phase','unknown'));return
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame): raise KeyboardInterrupt('pipeline interrupted; preserve artifacts')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        pipeline.execute()
    except BaseException:
        for child in pipeline.children:
            if child.poll() is None: kill_tree(child.pid)
        pipeline.state['error']=traceback.format_exc();pipeline.publish('stopped_with_error')
        raise
    finally:
        lock.close()


if __name__=='__main__':
    main()
