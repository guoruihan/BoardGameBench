"""Bootstrap value, learn new tree targets, feed models back to the search peer."""
import os
from pathlib import Path
import time

from boardbench.artifacts.store import digest,read_json,write_json,source_id
from boardbench.v127.preflight import reserve
from boardbench.v145.campaign import Runner as PreviousRunner
from boardbench.v147.campaign import trial_config as old_trial
from .evaluation import LAYOUT


class Runner(PreviousRunner):
    def __init__(self,repo,root):
        path=Path(root)/'cost.json'
        self.prior_controller=read_json(path).get('controller_seconds',0.) if path.exists() else 0.
        super().__init__(repo,root)

    def report(self):
        super().report();path=self.root/'cost.json';cost=read_json(path)
        cost['controller_seconds']=self.prior_controller+time.time()-self.controller_started
        cost['allocated_cpu_core_seconds']+=self.prior_controller
        cost['note']='V148 process/GPU reservation occupancy, not utilization; retries and prior controller time retained; original explicitly authorized deadline unchanged.'
        write_json(path,cost)


def evaluation_config(c,root,model,cycle):
    result=old_trial(c,root,dict(name='evaluation',expert_weight=0.),cycle)
    result.update(layout=LAYOUT,seed=14800+cycle,expert_manifest=None,
        warm_start=model['model'],warm_start_sha256=model['model_sha256'],
        development_seeds=list(range(248100000,248100000+c['development_environments'])),
        decision_seeds=list(range(248110000+cycle*1000,248110000+cycle*1000+c['decision_environments'])),
        confirmation_seeds=list(range(348000000,348000000+c['confirmation_environments'])))
    return result


def fit_config(c,data,model,evaluation,arm,deadline):
    return dict(**arm,objective='score',dataset=str(data),dataset_sha256=digest(data),**model,
        steps=c['fit_steps'],evaluate_interval=c['fit_evaluate_interval'],minibatch=256,seed=14801,
        mcts_fraction=.7,evaluation_config=evaluation,deadline_unix=deadline,
        finish_reserve_seconds=120,source_id=source_id())


def job(module,path,out,training=False):
    return dict(name=out.parent.name+'_'+out.name,out=str(out),training=training,
        complete=str(out/('result.json' if module.endswith('preflight') else 'done.json')),
        args=['-m',module,'--config',str(path),'--out',str(out)])


def data_job(r,manifests,name):
    target=r.root/'datasets'/name
    r.run([dict(name='data_'+name,out=str(target),complete=str(target/'manifest.json'),args=[
        '-m','boardbench.v148.data','--manifests',*manifests,'--out',str(target),
        '--native-manifest',r.c['native_manifest'],'--deadline',str(r.deadline-1800)])],
        min(r.deadline-1800,time.time()+1200),'build_'+name)
    return target/'manifest.json'


def value_record(paths):
    eligible=[]
    for path in paths:
        manifest=path/'value_manifest.json'
        if manifest.exists():
            m=read_json(manifest)
            if m['eligible_for_search']:eligible.append((m['metrics']['value_mse'],manifest))
    if not eligible:return None
    _,path=min(eligible)
    return dict(path=str(path),sha256=digest(path))


def wait_for(r,path):
    r.phase='await_search_data'
    began=time.time()
    while not path.exists():
        if r.stopping or time.time()>=r.deadline-1800:raise InterruptedError('bounded data wait')
        status=Path(r.c['search_root'])/'status.json'
        if (time.time()-began>60 and status.exists()
                and read_json(status)['phase'] in ('failed','interrupted')):
            raise RuntimeError('search peer stopped')
        r.report();time.sleep(5)
    return read_json(path)


def evaluate_records(r,records,seeds,stage,until):
    jobs=[]
    for row in records:
        target=r.root/stage/row['name'];path=r.root/'configs'/(stage+'_'+row['name']+'.json')
        spec=dict(config=row['config'],model=row['model'],model_sha256=row['model_sha256'],seeds=seeds,
                  stage='confirmation' if stage=='confirmation' else 'development',deadline_unix=until,
                  baseline=r.c['student'],baseline_sha256=r.c['student_sha256'])
        if path.exists():
            saved=read_json(path)
            # Resume the original stage deadline, never make a new relative one.
            spec['deadline_unix']=saved['deadline_unix']
            if saved!=spec:raise ValueError('immutable evaluation spec')
            until=min(until,saved['deadline_unix'])
        write_json(path,spec)
        jobs.append(dict(name=stage+'_'+row['name'],out=str(target),
            complete=str(target/('summary.json' if stage=='confirmation' else 'result.json')),
            args=['-m','boardbench.v148.evaluate','--spec',str(path),'--out',str(target)]))
    r.run(jobs,until,stage)


def execute(root):
    root=Path(root);c=read_json(root/'resolved_config.json');launch=read_json(root/'launch.json')
    if (source_id()!=launch['source_id'] or digest(root/'resolved_config.json')!=launch['resolved_config_sha256']
            or os.uname().nodename!=c['host'] or time.time()>=c['deadline_unix']):raise ValueError('immutable source/host/window')
    lock=reserve(c['repo'],c);os.sched_setaffinity(0,{launch['cpus'][-1]});r=Runner(Path(c['repo']),root)
    records=[];manifests=list(c['initial_expert_manifests'])
    current=dict(model=c['student'],model_sha256=c['student_sha256'])
    try:
        data=data_job(r,manifests,'bootstrap');ev=evaluation_config(c,root,current,0)
        gates=[]
        for i in range(4):
            cfg=fit_config(c,data,current,ev,dict(mode='joint',learning_rate=.00001,anchor_weight=.1,value_weight=.25),c['deadline_unix'])
            path=root/'configs'/f'gate{i}.json';write_json(path,cfg)
            gates.append(job('boardbench.v148.preflight',path,root/f'preflight_gpu{i}'))
        r.run(gates,min(r.deadline,time.time()+900),'actual_cuda_preflight')
        for i in range(4):
            gate=read_json(root/f'preflight_gpu{i}/result.json')
            if not gate['passed'] or gate['source_id']!=source_id():raise ValueError('matching CUDA receipt required')
        bootstrap=[]
        for i,lr in enumerate((.00003,.0001)):
            target=root/'bootstrap'/f'value{i}';path=root/'configs'/f'bootstrap_value{i}.json'
            if not path.exists():write_json(path,fit_config(c,data,current,ev,
                dict(mode='value',learning_rate=lr,anchor_weight=0.,value_weight=1.),min(r.deadline-1800,time.time()+1200)))
            bootstrap.append(target)
        r.run([job('boardbench.v148.fit',root/'configs'/f'bootstrap_value{i}.json',t,True) for i,t in enumerate(bootstrap)],
              min(r.deadline-1800,time.time()+1200),'bootstrap_value_models')
        ready=root/'ready_0.json'
        if not ready.exists():write_json(ready,dict(student=current,value=value_record(bootstrap),uses_confirmation=False))
        for cycle in range(c['cycles']):
            ready=read_json(root/f'ready_{cycle}.json');current=ready['student']
            expert=Path(c['search_root'])/f'expert_{cycle}.json';meta=wait_for(r,expert)
            if meta.get('stop'):
                write_json(root/f'ready_{cycle+1}.json',dict(stop=True,student=current,value=None));break
            manifests.append(str(expert));data=data_job(r,manifests,f'cycle{cycle}')
            ev=evaluation_config(c,root,current,cycle);ecpath=root/'configs'/f'evaluation_{cycle}.json';write_json(ecpath,ev)
            arms=[dict(name='visits',mode='policy',learning_rate=.00001,anchor_weight=0.,value_weight=0.),
                  dict(name='anchored',mode='policy',learning_rate=.00001,anchor_weight=.2,value_weight=0.),
                  dict(name='joint',mode='joint',learning_rate=.00001,anchor_weight=.1,value_weight=.25),
                  dict(name='value',mode='value',learning_rate=.00003,anchor_weight=0.,value_weight=1.)]
            jobs=[];targets=[]
            for arm in arms:
                target=root/'rounds'/f'r{cycle+1}'/arm['name'];path=root/'configs'/f'r{cycle+1}_{arm["name"]}.json'
                if not path.exists():write_json(path,fit_config(c,data,current,ev,arm,min(r.deadline-1800,time.time()+1800)))
                targets.append(target);jobs.append(job('boardbench.v148.fit',path,target,True))
            r.run(jobs,min(r.deadline-1800,time.time()+1800),f'learn_cycle{cycle}')
            candidates=[]
            for target in targets[:3]:
                chosen=read_json(target/'selection.json');policy=target/'best_policy.pt'
                # A fallback to step0 is literally the parent; do not relabel it
                # as an independent new learned model or duplicate experiments.
                if chosen['best_step']==0:continue
                candidates.append(dict(name=target.parent.name+'_'+target.name,model=str(policy),
                    model_sha256=digest(policy),config=str(ecpath),evaluation=read_json(target/'decision.json')))
            if candidates:
                # The parent participates on this same new panel through every
                # trial's initial checkpoint; explicitly evaluate for paired ranking.
                parent=dict(name=f'parent_c{cycle}',**current,config=str(ecpath))
                pool=[parent,*[{k:v for k,v in q.items() if k!='evaluation'} for q in candidates]]
                stage=f'select_cycle{cycle}'
                evaluate_records(r,pool,ev['decision_seeds'],stage,min(r.deadline-1500,time.time()+900))
                best=max(pool,key=lambda q:read_json(root/stage/q['name']/'result.json')['score_mean'])
                current={k:best[k] for k in ('model','model_sha256')}
                records.extend({k:v for k,v in q.items() if k!='evaluation'} for q in candidates)
            next_ready=root/f'ready_{cycle+1}.json'
            if not next_ready.exists():write_json(next_ready,dict(student=current,value=value_record(targets),uses_confirmation=False))
            write_json(root/'retained.json',records)
        if records:
            ev=evaluation_config(c,root,current,0);path=root/'configs'/'final_evaluation.json';write_json(path,ev)
            records.append(dict(name='initial_parent',model=c['student'],model_sha256=c['student_sha256'],config=str(path)))
            evaluate_records(r,records,list(range(248200000,248200000+c['selection_environments'])),
                             'selection',r.deadline-900)
            selected=sorted(records,key=lambda q:-read_json(root/'selection'/q['name']/'result.json')['score_mean'])[:2]
            write_json(root/'selected.json',dict(models=selected,uses_confirmation=False))
            evaluate_records(r,selected,list(range(348000000,348000000+c['confirmation_environments'])),
                             'confirmation',r.deadline-15)
            results=[dict(name=q['name'],**read_json(root/'confirmation'/q['name']/'summary.json')) for q in selected]
        else:results=[]
        write_json(root/'summary.json',dict(status='completed',selected=results,retained=records,
            completed_unix=time.time(),selection_uses_confirmation=False,
            note='No new retained models means all candidates fell back; never report search scores as pure model scores.'))
        r.phase='completed';r.report()
    except BaseException as exc:
        r.phase='interrupted' if isinstance(exc,(InterruptedError,TimeoutError)) else 'failed'
        write_json(root/(r.phase+'.json'),dict(error=repr(exc),time=time.time()));raise
    finally:r.close();lock.close()
