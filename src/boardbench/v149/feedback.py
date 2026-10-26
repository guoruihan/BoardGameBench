"""One empirical policy-evaluation / improvement cycle, within the parent window.

Collect NEW training games with the selected planner, fit its realized returns,
then evaluate new planners on separate development seeds. Do not pool returns
from different continuation policies or silently train on development games.
"""
from pathlib import Path
import time

from boardbench.artifacts.store import digest,read_json,write_json
from boardbench.v129.encoding import NativeEncoder
from boardbench.v146.report import paired
from .campaign import Campaign
from .data import build_feedback
from .diagnostic import prediction_report
from .value import Value


class Feedback(Campaign):
    def run(self):
        parent=Path(self.c['feedback_parent'])
        if digest(parent/'config.json')!=self.c['parent_config_sha256']:
            raise ValueError('parent provenance changed')
        pilot=read_json(parent/'pilot_summary.json')['results']
        options=[name for name,r in pilot.items() if name.endswith('_turn_1.0') and r['timing_gate']]
        chosen=max(options,key=lambda name:pilot[name]['score'])
        parent_job=read_json(parent/'jobs'/f'{chosen}_e249000000_s0_pilot.json')
        spec={k:parent_job[k] for k in ('method','seconds','value_manifest','value_manifest_sha256')}
        spec['label']='search_value_turn'
        write_json(self.root/'selection.json',dict(parent_label=chosen,selection_split='pilot_development',
            behavior=spec,note='No evaluation trajectories enter the feedback training dataset.'))
        self.panel('feedback_train',[spec],range(149001000,149001064),split='train')
        self.phase='feedback_data';self.status()
        native=NativeEncoder.from_artifact(self.c['job']['native_artifact'])
        data=self.root/'data';build_feedback(self.root/'feedback_train_summary.json',data,native)
        model=read_json(spec['value_manifest'])
        values={'parent':dict(path=spec['value_manifest'],sha256=spec['value_manifest_sha256'])}
        for name,lr in [('replay',.00003),('replay_fast',.0001)]:
            self.phase='fit_'+name;self.status();out=self.root/'fit'/name
            cfg=dict(model=model['model'],model_sha256=model['model_sha256'],dataset=str(data/'manifest.json'),
                dataset_sha256=digest(data/'manifest.json'),mode='value',learning_rate=lr,anchor_weight=0.,
                value_weight=1.,minibatch=256,seed=149,steps=1200,evaluate_interval=100,
                deadline_unix=self.c['deadline_unix'],finish_reserve_seconds=30,source_id=self.c['source_id'])
            path=self.root/'configs'/(name+'.json');write_json(path,cfg)
            if name=='replay':
                self.execute(['-m','boardbench.v149.preflight','--config',str(path),'--out',str(self.root/'preflight'),
                    '--device','cuda'],self.root/'preflight',self.root/'preflight/result.json',self.c['trainer_cpu'],True)
            args=['-m','boardbench.v148.fit','--config',str(path),'--out',str(out),'--device','cuda']
            if (out/'latest.pt').exists() and not (out/'done.json').exists():args+=['--resume']
            self.execute(args,out,out/'done.json',self.c['trainer_cpu'],True)
            manifest=out/'value_manifest.json';values[name]=dict(path=str(manifest),sha256=digest(manifest))
            write_json(self.root/'costs.json',self.costs)
        for name,m in values.items():
            value=Value(m['path'],m['sha256'],native,self.c['job']['opponent_sha256'])
            prediction_report(data/'manifest.json',value,self.root/'diagnostics'/(name+'_prediction.json'))
        specs=[dict(label='model',method='model',seconds=.1),
               dict(label='rollout_0.1',method='search_rollout',seconds=.1),
               dict(label='rollout_1.0',method='search_rollout',seconds=1.)]
        for name,m in values.items():
            for cutoff,budget in [('turn',1.),('one',.1)]:
                specs.append(dict(label=f'{name}_{cutoff}_{budget}',method='search_value_'+cutoff,seconds=budget,
                    value_manifest=m['path'],value_manifest_sha256=m['sha256']))
        summary=self.panel('feedback_pilot',specs,range(249010000,249010004))
        # Keep original parents and equal-budget controls; new seeds check the
        # best update at each budget without changing the chosen model afterward.
        selected=[]
        for budget in (.1,1.):
            options=[s for s in specs if s['label'].startswith('replay') and s['seconds']==budget
                     and summary[s['label']]['timing_gate']]
            if options:selected.append(max(options,key=lambda s:summary[s['label']]['score']))
        expanded=None;comparisons={}
        if selected and time.time()<self.c['deadline_unix']-300:
            keep=[s for s in specs if s['label'].startswith(('model','rollout','parent'))]+selected
            expanded=self.panel('feedback_expanded',keep,range(249011000,249011016))
            rows=read_json(self.root/'feedback_expanded_summary.json')['rows']
            for spec in selected:
                label=spec['label'];base='parent_'+label.split('_',1)[1].removeprefix('fast_')
                comparisons[label]=paired([r for r in rows if r['method']==label],
                                           [r for r in rows if r['method']==base])
        self.phase='completed';self.status()
        write_json(self.root/'summary.json',dict(status='completed',parent_label=chosen,pilot=summary,
            expanded=expanded,vs_parent=comparisons,completed_unix=time.time(),
            costs=dict(allocated_cpu_seconds=sum(c['wall_seconds'] for c in self.costs)+time.time()-self.controller_started,
                       allocated_gpu_seconds=sum(c['wall_seconds']*c['gpu_slots'] for c in self.costs)),
            note='One feedback iteration. Old policies preserved; no automatic promotion. Development only.'))
