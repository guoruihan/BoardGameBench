"""Expert initialization and three development-driven whole-game PPO rounds."""
from pathlib import Path
import os
import time

from boardbench.artifacts.store import digest,read_json,write_json,source_id
from boardbench.v127.preflight import reserve
from boardbench.v122.campaign import physical_cpus
from boardbench.v144.campaign import retained_records
from boardbench.v145.campaign import Runner as PreviousRunner
from boardbench.v144.evaluation import paired_delta
from .training import LAYOUT


class Runner(PreviousRunner):
    def report(self):
        super().report()
        previous=self.c.get('previous_learning_run')
        if previous:
            old=read_json(Path(previous)/'cost.json');current=read_json(self.root/'cost.json')
            current.update(previous_learning_run=previous,prior_learning_cost_sha256=digest(Path(previous)/'cost.json'),
                combined_physical_gpu_seconds=current['physical_gpu_seconds']+old['physical_gpu_seconds'],
                combined_allocated_cpu_core_seconds=current['allocated_cpu_core_seconds']+old['allocated_cpu_core_seconds'],
                accounting_note='Combined fields add the preserved, stopped first attempt; per-attempt fields remain separate. Same original wall deadline.')
            write_json(self.root/'cost.json',current)


def trial_config(c,root,arm,index=0):
    result={**c['training'],**arm}
    result.update(layout=LAYOUT,rules_version='harmonies_duo_a_base32_v1',objective='score',
        parent=c['parent'],parent_sha256=c['parent_sha256'],
        warm_start=arm.get('warm_start',c['student']),warm_start_sha256=arm.get('warm_start_sha256',c['student_sha256']),
        training_anchor=c['opponent'],training_anchor_sha256=c['opponent_sha256'],
        native_artifact=c['native_artifact'],evaluation_opponents=[],seed=14701+index*100+arm.get('replica',0),
        workers=c['workers'],batch_matches=c['batch_matches'],source_id=source_id(),
        development_seeds=list(range(247100000,247100000+c['development_environments'])),
        decision_seeds=list(range(247110000+index*1000,247110000+index*1000+c['decision_environments'])),
        confirmation_seeds=list(range(347000000,347000000+c['confirmation_environments'])),
        replay_directory=str(root/'unused_replay'),deadline_unix=c['deadline_unix'])
    return result


def spec(module,path,target,training=False):
    return dict(name=target.parent.name+'_'+target.name,out=str(target),training=training,
        complete=str(target/('result.json' if module.endswith('preflight') else 'done.json')),
        args=['-m',module,'--config',str(path),'--out',str(target)])


def record(target):
    target=Path(target);path=target/'best_policy.pt'
    return dict(name=target.parent.name+'_'+target.name,policy=str(path),sha256=digest(path),
        config_path=str(target/'config.json'),config=read_json(target/'config.json'),evaluation=read_json(target/'decision.json'))


def complementary_records(targets):
    records=[]
    for target in map(Path,targets):
        if not (target/'skill_decision.json').exists():continue
        row=record(target);row.update(name=row['name']+'_skill',policy=str(target/'skill_policy.pt'),
            sha256=digest(target/'skill_policy.pt'),evaluation=read_json(target/'skill_decision.json'))
        records.append(row)
    return records


def initial_arms(initial,skills):
    best=max(initial,key=lambda row:row['evaluation']['score_mean'])
    eligible=[row for row in skills if row['evaluation']['score_mean']>=best['evaluation']['score_mean']-20
              and row['evaluation']['animal_mean']>best['evaluation']['animal_mean']+3]
    skilled=max(eligible,key=lambda row:row['evaluation']['animal_mean']) if eligible else best
    warm=lambda row:dict(warm_start=row['policy'],warm_start_sha256=row['sha256'])
    arms=[dict(name='control',expert_weight=0.),dict(name='skill_recover',**warm(skilled),expert_weight=0.),
          dict(name='bc_retain',**warm(best),expert_weight=.1),
          dict(name='skill_long_credit',**warm(skilled),expert_weight=.03,gae_lambda=1.)]
    if not eligible:
        # If every BC branch collapses to the same initial weights, make the
        # second arm a distinct exploration/step-size test instead of an exact duplicate.
        arms[1].update(name='explore',entropy=.015,learning_rate=.00002)
    return arms,dict(selected_bc=best['name'],selected_skill=skilled['name'] if eligible else None,
        uses_confirmation=False,rationale='retain total-score champion and complementary animal skill; PPO repairs full-game deficits; no duplicate fallback arm')


def choose_arms(records,index):
    if not records:raise ValueError('no previous complete policies')
    for r in records:
        if any(not 247000000<=g['seed']<248000000 for g in r['evaluation']['per_game']):raise ValueError('no confirmation feedback')
    best=max(records,key=lambda r:r['evaluation']['score_mean'])
    control=next((r for r in records if r['config'].get('name')=='control'),best)
    gap=paired_delta(best['evaluation'],control['evaluation'])
    warm=lambda r:dict(warm_start=r['policy'],warm_start_sha256=r['sha256'])
    animal_gap=gap['metrics']['animal']['delta']
    # Different credit/retention choices respond to actual animal transfer.
    strong=.12 if animal_gap<=2 else .04
    arms=[dict(name='control',**warm(control),expert_weight=0.),
          dict(name='continue',**warm(best),expert_weight=0.),
          dict(name='expert_retain',**warm(best),expert_weight=strong,expert_decay_actions=4000000),
          dict(name='full_return' if index==1 else 'selfplay_mix',**warm(best),
               expert_weight=.03,gae_lambda=1. if index==1 else .95,mode='fixed' if index==1 else 'mixed')]
    return arms,dict(selected=best['name'],paired_vs_control=gap,uses_confirmation=False,
        rationale='retain control; continue best; strengthen expert retention if animal gain is weak; test full-return credit then mixed own-score play')


def dataset(r,root,c,index):
    if digest(c['initial_expert_manifest'])!=c['initial_expert_manifest_sha256']:
        raise ValueError('initial expert archive manifest changed')
    manifests=[c['initial_expert_manifest']]
    search=Path(c['search_root'])
    # Consume only completed immutable manifests; never half-written trajectories.
    if index:
        path=search/'expert_manifest.json'
        if not path.exists():path=search/'expert_main_manifest.json'
        if path.exists():manifests.append(str(path))
    target=root/'datasets'/f'v{index}';complete=target/'manifest.json'
    job=dict(name=f'data_v{index}',out=str(target),complete=str(complete),args=['-m','boardbench.v147.data',
        '--manifests',*manifests,'--out',str(target),'--native-manifest',c['native_manifest'],
        '--deadline',str(min(c['deadline_unix']-1800,time.time()+1800))])
    r.run([job],min(c['deadline_unix']-1800,time.time()+1800),f'dataset_v{index}')
    return dict(expert_manifest=str(complete),expert_manifest_sha256=digest(complete))


def evaluate_specs(root,records,seeds,stage,until,c):
    jobs=[]
    for row in records:
        target=root/stage/row['name'];path=root/'configs'/(stage+'_'+row['name']+'.json')
        value=dict(config=row['config_path'],model=row['policy'],model_sha256=row['sha256'],seeds=seeds,
            stage='confirmation' if stage=='confirmation' else 'development',deadline_unix=until,
            baseline=c['student'],baseline_sha256=c['student_sha256'])
        if path.exists() and read_json(path)!=value:raise ValueError('immutable evaluation')
        write_json(path,value)
        jobs.append(dict(name=stage+'_'+row['name'],out=str(target),complete=str(target/('summary.json' if stage=='confirmation' else 'result.json')),
            args=['-m','boardbench.v147.evaluate','--spec',str(path),'--out',str(target)]))
    return jobs


def execute(root):
    root=Path(root);c=read_json(root/'resolved_config.json');launch=read_json(root/'launch.json');repo=Path(c['repo'])
    if source_id()!=launch['source_id'] or digest(root/'resolved_config.json')!=launch['resolved_config_sha256']:
        raise ValueError('frozen source/config mismatch')
    if os.uname().nodename!=c['host'] or time.time()>=c['deadline_unix'] or (root/'summary.json').exists():
        raise ValueError('wrong host/completed/expired run')
    physical=physical_cpus()
    if not set(launch['cpus'])<=set(physical) or len(launch['cpus'])>len(physical)//2:
        raise ValueError('resume must still obey per-machine half-physical-CPU cap')
    lock=reserve(repo,c);os.sched_setaffinity(0,{launch['cpus'][-1]});r=Runner(repo,root)
    try:
        data=dataset(r,root,c,0)
        gates=[]
        for i in range(4):
            cfg=trial_config(c,root,dict(name='gate',**data));path=root/'configs'/f'gate{i}.json';write_json(path,cfg)
            gates.append(spec('boardbench.v147.preflight',path,root/f'preflight_gpu{i}'))
        r.run(gates,min(c['deadline_unix'],time.time()+900),'real_cuda_gates')
        for i in range(4):
            gate=read_json(root/f'preflight_gpu{i}/result.json')
            if not gate['passed'] or gate['source_id']!=launch['source_id']:raise ValueError('actual CUDA gate mismatch')
        schedule=root/'schedule.json'
        if not schedule.exists():
            now=time.time();init_until=min(now+2400,c['deadline_unix']-7200)
            training_end=c['deadline_unix']-c['final_reserve_seconds']
            if training_end-init_until<5400:raise ValueError('insufficient original window')
            write_json(schedule,dict(initialization_until=init_until,round_deadlines=[init_until+(training_end-init_until)*(i+1)/3 for i in range(3)]))
        timeline=read_json(schedule);init=[]
        for name,q,value,lr in [('bc',0.,0.,.0001),('bc_q',.5,0.,.0001),('bc_value',0.,.1,.0001),('bc_cautious',.5,.1,.00003)]:
            path=root/'configs'/f'init_{name}.json';target=root/'initialization'/name
            if not path.exists():write_json(path,trial_config(c,root,dict(name=name,**data,learning_rate=lr,
                expert_q_weight=q,expert_value_weight=value,deadline_unix=timeline['initialization_until'])))
            # trial_config sets global deadline; finite stage remains explicit.
            cfg=read_json(path)
            if cfg['deadline_unix']!=timeline['initialization_until']:
                cfg['deadline_unix']=timeline['initialization_until'];write_json(path,cfg)
            init.append(spec('boardbench.v147.imitation',path,target,True))
        r.run(init,timeline['initialization_until'],'imitation_initialization')
        initial=[record(Path(s['out'])) for s in init]
        skills=complementary_records([s['out'] for s in init])
        for index,until in enumerate(timeline['round_deadlines']):
            decision_path=root/'decisions'/f'round{index+1}.json'
            if decision_path.exists():decision=read_json(decision_path);arms=decision['arms'];data=decision['data']
            else:
                if index==0:
                    arms,reason=initial_arms(initial,skills)
                else:
                    previous=[row for row in retained_records(root) if row['name'].startswith(f'r{index}_')]
                    arms,reason=choose_arms(previous,index);data=dataset(r,root,c,index)
                    # New demonstrations need a genuine initialization opportunity,
                    # not just a few tiny auxiliary updates after millions of PPO actions.
                    refresh=root/'refresh'/f'r{index+1}';refresh_path=root/'configs'/f'refresh{index+1}.json'
                    if not refresh_path.exists():
                        cfg=trial_config(c,root,{**arms[2],**data,'learning_rate':.00003,'imitation_steps':400},index+1)
                        cfg['deadline_unix']=min(until-900,time.time()+1200);write_json(refresh_path,cfg)
                    r.run([spec('boardbench.v147.imitation',refresh_path,refresh,True)],
                          read_json(refresh_path)['deadline_unix'],f'refresh_expert_initialization{index+1}')
                    policy=refresh/'best_policy.pt'
                    arms[2].update(warm_start=str(policy),warm_start_sha256=digest(policy))
                    reason['refreshed_expert_initialization']=str(refresh)
                decision=dict(arms=arms,data=data,reason=reason,deadline_unix=until,created_unix=time.time())
                write_json(decision_path,decision)
            jobs=[]
            for arm in arms:
                target=root/'rounds'/f'r{index+1}'/arm['name'];path=root/'configs'/f'r{index+1}_{arm["name"]}.json'
                if not path.exists():
                    cfg=trial_config(c,root,{**arm,**data},index+1)
                    cfg.update(deadline_unix=until-30,max_seconds=max(0,until-time.time()),replay_directory=str(target/'replay'))
                    write_json(path,cfg)
                jobs.append(spec('boardbench.v147.training',path,target,True))
            try:r.run(jobs,until,f'ppo_round{index+1}')
            except RuntimeError as exc:
                completed=[row for row in retained_records(root) if row['name'].startswith(f'r{index+1}_')]
                if len(completed)<2 or not any(row['config']['name']=='control' for row in completed):raise
                write_json(root/'decisions'/f'partial_round{index+1}.json',dict(error=repr(exc),
                    retained=[row['name'] for row in completed],note='Failed arms excluded, not assigned scores; complete control retained.'))
            write_json(root/'retained.json',[{k:v for k,v in row.items() if k not in ('evaluation','config')} for row in retained_records(root)])
        records=retained_records(root)+initial+skills+[record(path.parent) for path in sorted((root/'refresh').glob('*/done.json'))]
        records.append(dict(name='initial_parent',config_path=initial[0]['config_path'],policy=c['student'],sha256=c['student_sha256']))
        select_until=c['deadline_unix']-c['confirmation_reserve_seconds']
        r.run(evaluate_specs(root,records,list(range(247200000,247200000+c['selection_environments'])),
                            'selection',select_until,c),select_until,'fresh_model_selection')
        selection=root/'selection.json'
        if not selection.exists():
            ranked=sorted(records,key=lambda row:read_json(root/'selection'/row['name']/'result.json')['score_mean'],reverse=True)
            selected=ranked[:3];control=next(row for row in ranked if row.get('config',{}).get('name')=='control')
            if control['name'] not in {row['name'] for row in selected}:selected.append(control)
            write_json(selection,dict(models=[{k:v for k,v in row.items() if k not in ('evaluation','config')} for row in selected],uses_confirmation=False))
        selected=read_json(selection)['models']
        r.run(evaluate_specs(root,selected,list(range(347000000,347000000+c['confirmation_environments'])),
            'confirmation',c['deadline_unix']-15,c),c['deadline_unix']-15,'independent_model_confirmation512')
        write_json(root/'summary.json',dict(status='completed',selected=[dict(name=row['name'],**read_json(root/'confirmation'/row['name']/'summary.json')) for row in selected],
            selection_uses_confirmation=False,completed_unix=time.time(),deadline_unix=c['deadline_unix'],
            note='Pure model scores; search900s evaluated separately. No deployment pointer automatically replaced.'))
        r.phase='completed';r.report()
    except BaseException as exc:
        r.phase='interrupted' if isinstance(exc,(InterruptedError,TimeoutError)) else 'failed'
        write_json(root/(r.phase+'.json'),dict(error=repr(exc),time=time.time()));raise
    finally:r.close();lock.close()
