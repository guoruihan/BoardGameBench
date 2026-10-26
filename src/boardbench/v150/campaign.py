"""Result-driven, deadline-bounded component/ranking/value feedback experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from boardbench.artifacts.store import read_json,write_json,digest
from boardbench.v149.campaign import Campaign as Previous
from boardbench.v146.report import paired
from .branches import select_cases,merge


def strategy_key(spec):
    return tuple(spec.get(k) for k in ('method','seconds','width','value_manifest_sha256','rank_mix'))


def adapt(summary,incumbent,stall):
    value=summary[incumbent];ref=summary['rollout_1.0']
    return dict(terrain_weight=2. if ref['terrain']-value['terrain']>5 else 1.,
        animal_weight=2. if ref['animal']-value['animal']>3 else 1.,
        branch_cases=384 if stall>=2 else 192,branch_samples=4 if stall>=2 else 2,
        width=4 if value['fallback_rate']>.2 else 6 if stall<2 else 8,
        expanded_environments=32 if stall>=2 else 16,
        continuation='actor' if ref['terrain']-value['terrain']>5 else 'habitat',
        reason=dict(terrain_gap=ref['terrain']-value['terrain'],animal_gap=ref['animal']-value['animal'],
                    fallback_rate=value['fallback_rate'],rounds_without_reliable_gain=stall))


class Campaign(Previous):
    def __init__(self,root):
        super().__init__(root)
        self.work_deadline=self.c['deadline_unix']-self.c.get('final_reserve_seconds',3600)
        self.decisions=read_json(self.root/'decisions.json') if (self.root/'decisions.json').exists() else []

    def panel(self,name,specs,seeds,split='development'):
        self.phase=name;self.status();slots=self.c['worker_cpus'];jobs=[(s,e,seat) for s in specs for e in seeds for seat in (0,1)]
        deadline=self.c['deadline_unix']-60 if split=='confirmation' else self.work_deadline
        def work(index):
            rows=[]
            for spec,seed,seat in jobs[index::len(slots)]:
                target=self.root/name/f'{spec["label"]}_e{seed}_s{seat}'
                job={**self.c['job'],**spec,'id':target.name,'source_id':self.c['source_id'],
                    'environment_seed':seed,'seat':seat,'search_seed':150,'split':split,'deadline_unix':deadline}
                path=self.root/'jobs'/name/(target.name+'.json');write_json(path,job)
                rows.append(self.execute(['-m','boardbench.v150.worker','--job',str(path),'--out',str(target)],
                    target,target/'result.json',slots[index]))
            return rows
        rows=[]
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            for future in as_completed([pool.submit(work,i) for i in range(len(slots))]):
                rows.extend(future.result());self.status(completed=len(rows),requested=len(jobs))
        summary={}
        for spec in specs:
            rs=[r for r in rows if r['method']==spec['label']]
            summary[spec['label']]={k:statistics.mean(r[k] for r in rs) for k in
                ('score','animal','terrain','animals_placed','completed_cards','decision_seconds','fallback_rate')}
            summary[spec['label']].update(games=len(rs),timing_gate=all(r['timing_gate'] for r in rs),
                latency_max=max(r['latency_max'] for r in rs),over_budget_calls=sum(r['over_budget_calls'] for r in rs))
        result=dict(split=split,results=summary,rows=rows,completed_unix=time.time())
        write_json(self.root/name/'summary.json',result);write_json(self.root/'costs.json',self.costs)
        return result

    def dataset(self,source,name):
        self.phase=name;self.status();out=self.root/name
        job=dict(source=str(source),native_artifact=self.c['job']['native_artifact'],deadline_unix=self.work_deadline)
        path=self.root/'jobs'/(name+'.json');write_json(path,job)
        self.execute(['-m','boardbench.v150.data','--job',str(path),'--out',str(out)],out,out/'manifest.json',self.c['trainer_cpu'])
        return out/'manifest.json'

    def branches(self,data,name,settings):
        out=self.root/name
        if (out/'manifest.json').exists():return out/'manifest.json'
        self.phase=name;self.status()
        cases=select_cases(data,settings['branch_cases']);slots=self.c['worker_cpus']
        # At least two roots per shard; empty shards are not submitted.
        queues=[cases[i::min(len(slots),max(1,len(cases)//4))] for i in range(min(len(slots),max(1,len(cases)//4)))]
        def work(i):
            target=out/f'shard{i:03d}';job={**self.c['job'],'cases':queues[i],
                'samples':settings['branch_samples'],'width':min(6,settings['width']),
                'continuation':settings.get('continuation','habitat'),'deadline_unix':self.work_deadline,
                'source_id':self.c['source_id']}
            path=self.root/'jobs'/name/f'shard{i:03d}.json';write_json(path,job)
            return self.execute(['-m','boardbench.v150.branches','--job',str(path),'--out',str(target)],
                target,target/'result.json',slots[i])
        with ThreadPoolExecutor(max_workers=len(queues)) as pool:results=list(pool.map(work,range(len(queues))))
        merge(results,out);write_json(self.root/'costs.json',self.costs);return out/'manifest.json'

    def fit(self,name,data,pairs,arm,settings,warm):
        self.phase=name;self.status();out=self.root/name
        cfg=dict(base_model=self.c['base_model'],base_sha256=self.c['base_sha256'],dataset=str(data),
            dataset_sha256=digest(data),pairs=str(pairs),pairs_sha256=digest(pairs),seed=150,steps=1200,
            learning_rate=.0001,component_loss=0. if arm=='scalar' else 1.,rank_loss=1. if arm=='rank' else 0.,
            terrain_weight=settings['terrain_weight'],animal_weight=settings['animal_weight'],deadline_unix=self.work_deadline)
        if warm:
            m=read_json(warm)
            if m['format']=='v150_component_value':cfg.update(warm_head=m['head'],warm_sha256=m['head_sha256'])
        path=self.root/'jobs'/(name+'.json');write_json(path,cfg)
        if not (out/'done.json').exists():
            state=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
            available=state.returncode==0 and any(i in self.c['allowed_gpus'] and m<=100 and u<=10 for i,m,u in
                (map(int,line.split(',')) for line in state.stdout.splitlines()))
            device='cuda' if available else 'cpu'
            args=['-m','boardbench.v150.train','--config',str(path),'--out',str(out),'--device',device]
            if (out/'latest.pt').exists():args+=['--resume']
            try:self.execute(args,out,out/'done.json',self.c['trainer_cpu'],available)
            except RuntimeError as e:
                if not available:raise
                write_json(self.root/'gpu_fallback'/f'{name.replace("/","_")}.json',dict(reason=str(e),time=time.time()))
                args=['-m','boardbench.v150.train','--config',str(path),'--out',str(out),'--device','cpu']
                if (out/'latest.pt').exists():args+=['--resume']
                self.execute(args,out,out/'done.json',self.c['trainer_cpu'],False)
        write_json(self.root/'costs.json',self.costs)
        return out/'value_manifest.json'

    def final_report(self,state,final=None):
        costs=dict(allocated_job_cpu_hours=sum(r['wall_seconds'] for r in self.costs)/3600,
                   controller_wall_hours=(time.time()-self.controller_started)/3600,
                   allocated_gpu_hours=sum(r['wall_seconds']*r['gpu_slots'] for r in self.costs)/3600)
        result=dict(status=state,rounds=len(self.decisions),decisions=self.decisions,costs=costs,
            confirmation=final,completed_unix=time.time(),deadline_unix=self.c['deadline_unix'])
        if final:
            baseline=[r for r in final['rows'] if r['method']=='rollout_1.0']
            result['paired_vs_rollout']={name:{metric:paired([r for r in final['rows'] if r['method']==name],baseline,metric)
                for metric in ('score','animal','terrain')} for name in final['results'] if name!='rollout_1.0'}
        write_json(self.root/'summary.json',result)
        lines=['# V150 eight-hour results',f'Status: {state}; completed rounds: {len(self.decisions)}.',
               'Policy scores are own raw totals against the frozen opponent. No UI promotion.',
               f'Resource receipts: {costs}.','', '| Method | Score | Animal | Terrain | Seconds/game | Timing gate |',
               '|---|---:|---:|---:|---:|---|']
        if final:
            for name,r in final['results'].items():lines.append(f'|{name}|{r["score"]:.3f}|{r["animal"]:.3f}|{r["terrain"]:.3f}|{r["decision_seconds"]:.3f}|{r["timing_gate"]}|')
        else:lines+=['No completed final confirmation panel. Consult decisions.json and per-stage summaries.']
        (self.root/'REPORT.md').write_text('\n'.join(lines)+'\n')

    def run(self):
        incumbent=dict(label='incumbent',method='search_value_turn',seconds=1.,width=6,rank_mix=0.,
            value_manifest=self.c['initial_value'],value_manifest_sha256=digest(self.c['initial_value']))
        initial=deepcopy(incumbent);initial['label']='v149_reference'
        specialist=deepcopy(incumbent);stall=0
        settings=dict(terrain_weight=1.,animal_weight=1.,branch_cases=192,branch_samples=2,width=6,expanded_environments=16)
        source=Path(self.c['initial_data'])
        for round_id in range(self.c.get('max_rounds',64)):
            name=f'rounds/r{round_id:02d}';decision_path=self.root/name/'decision.json'
            if decision_path.exists():
                d=read_json(decision_path);incumbent=d['incumbent'];specialist=d['specialist'];settings=d['next_settings'];stall=d['stall'];source=Path(d['next_source'])
                if not any(x['round']==round_id for x in self.decisions):self.decisions.append(d)
                continue
            if self.stop or time.time()>self.work_deadline-1200 or shutil.disk_usage(self.root).free<5*1024**3:break
            data=self.dataset(source,name+'/data');pairs=self.branches(data,name+'/pairs',settings)
            controls=[dict(label='model',method='model',seconds=1.),dict(label='rollout_1.0',method='search_rollout',seconds=1.,width=6),incumbent]
            candidates=[]
            for arm in ('scalar','split','rank'):
                m=self.fit(name+'/fit/'+arm,data,pairs,arm,settings,incumbent['value_manifest'])
                mixes=(0.,.25,.75) if arm=='rank' else (0.,)
                for mix in mixes:
                    candidates.append(dict(label=f'{arm}_{mix}',method='search_value_turn',seconds=1.,width=settings['width'],
                        rank_mix=mix,value_manifest=str(m),value_manifest_sha256=digest(m)))
                if arm=='split':candidates.append({**candidates[-1],'label':'split_short','method':'search_value_short','width':4})
            base=250100000+round_id*10000
            screen=self.panel(name+'/screen',controls+candidates,range(base,base+4))
            valid=[s for s in candidates if screen['results'][s['label']]['timing_gate']]
            if not valid:raise RuntimeError('all value hypotheses failed timing gates')
            byscore=sorted(valid,key=lambda s:screen['results'][s['label']]['score'],reverse=True)[:2]
            animal=max(valid,key=lambda s:screen['results'][s['label']]['animal'])
            chosen=byscore+([animal] if animal not in byscore else [])
            n=settings['expanded_environments']
            expanded=self.panel(name+'/expanded',controls+chosen,range(base+1000,base+1000+n))
            rows=expanded['rows'];old=[r for r in rows if r['method']=='incumbent'];comparisons={}
            for s in chosen:
                rs=[r for r in rows if r['method']==s['label']]
                comparisons[s['label']]=dict(score=paired(rs,old),animal=paired(rs,old,'animal'))
            qualified=[s for s in chosen if expanded['results'][s['label']]['timing_gate']
                and comparisons[s['label']]['score']['delta']>=1 and comparisons[s['label']]['score']['ci95'][0]>0
                and comparisons[s['label']]['animal']['delta']>=-2]
            promoted=bool(qualified)
            if promoted:
                incumbent={**max(qualified,key=lambda s:expanded['results'][s['label']]['score']),'label':'incumbent'};stall=0
            else:stall+=1
            # Animal specialists are retained separately and never called the score champion.
            animal=max(chosen,key=lambda s:expanded['results'][s['label']]['animal'])
            specialist={**animal,'label':'animal_specialist'}
            comparison_label=next((s['label'] for s in chosen if strategy_key(s)==strategy_key(incumbent)),'incumbent')
            settings=adapt(expanded['results'],comparison_label,stall)
            behavior=specialist if round_id%3==2 else incumbent
            if time.time()>self.work_deadline-900:
                next_source=str(source)
            else:
                self.panel(name+'/train',[{**behavior,'label':'training_behavior'}],
                    range(150100000+round_id*10000,150100064+round_id*10000),split='train')
                next_source=str(self.root/name/'train/summary.json')
            d=dict(round=round_id,source=str(source),promoted=promoted,incumbent=incumbent,specialist=specialist,
                comparisons=comparisons,next_settings=settings,stall=stall,next_source=next_source,
                expanded_summary=str(self.root/name/'expanded/summary.json'),time=time.time())
            write_json(decision_path,d);self.decisions.append(d);write_json(self.root/'decisions.json',self.decisions)
            source=Path(next_source);write_json(self.root/'retained.json',dict(incumbent=incumbent,specialist=specialist,reference=initial))
        if self.stop:raise InterruptedError('stop requested')
        # Freeze before touching confirmation seeds; confirmation never changes selection.
        finalists=[dict(label='model',method='model',seconds=1.),dict(label='rollout_1.0',method='search_rollout',seconds=1.,width=6),initial,incumbent]
        if strategy_key(specialist)!=strategy_key(incumbent):finalists.append(specialist)
        unique=[];seen=set()
        for spec in finalists:
            key=strategy_key(spec)
            if key not in seen:unique.append(spec);seen.add(key)
        finalists=unique
        if (self.root/'finalists.json').exists():
            frozen=read_json(self.root/'finalists.json')
            if frozen['specs']!=finalists:raise ValueError('confirmation selection already frozen')
        else:write_json(self.root/'finalists.json',dict(specs=finalists,frozen_unix=time.time(),selection_uses_confirmation=False))
        final=self.panel('confirmation',finalists,range(350100000,350100000+self.c.get('confirmation_environments',128)),split='confirmation')
        # Secondary0.1s check uses the already-frozen value; no extra tuning.
        fast={**incumbent,'label':'value_one_0.1','method':'search_value_one','seconds':.1,'width':4}
        self.panel('confirmation_fast',[dict(label='model',method='model',seconds=.1),
            dict(label='rollout_0.1',method='search_rollout',seconds=.1,width=4),fast],
            range(350200000,350200064),split='confirmation')
        self.phase='completed';self.status();self.final_report('completed',final)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Campaign(a.root)
    try:r.run()
    except BaseException as e:
        r.phase='stopped' if r.stop or time.time()>=r.work_deadline else 'failed'
        r.status(error=repr(e));write_json(r.root/'costs.json',r.costs);r.final_report(r.phase);raise
