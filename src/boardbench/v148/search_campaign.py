"""Small paired search panels and training data, then search again with students."""
from pathlib import Path
import statistics
import time

from boardbench.artifacts.store import digest,read_json,write_json
from boardbench.v146.campaign import Runner as BaseRunner
from boardbench.v146.report import summarize,paired


class Runner(BaseRunner):
    WORKER_MODULE='boardbench.v148.worker'

    def jobs(self,methods,seeds,stage,student=None,value=None):
        jobs=super().jobs(methods,seeds,'expert' if stage.startswith('expert') else stage)
        for job in jobs:
            job.update(stage=stage,id=f'{stage}_{job["method"]}_e{job["environment_seed"]}_s{job["seat"]}',
                       mcts_config=self.config.get('mcts_config',{}))
            if student:job.update(student=student['model'],student_sha256=student['model_sha256'])
            if value:job.update(value_manifest=value['path'],value_manifest_sha256=value['sha256'])
        return jobs


def report(rows):
    result=summarize(rows);baseline=[r for r in rows if r['method']=='search_habitat']
    for method,metric in result.items():
        selected=[r for r in rows if r['method']==method]
        metric['vs_rollout']=paired(selected,baseline)
        for key in ('max_tree_depth','max_turn_depth','terminal_leaves','value_leaves','visited_actions'):
            metric[key]=statistics.mean(r.get(key,0) for r in selected)
    return result


def wait_for(r,path,phase):
    r.phase=phase
    began=time.time()
    while not path.exists():
        if r.stop or time.time()>=r.config['deadline_unix']-1800:
            raise InterruptedError('bounded wait; preserve completed search work')
        status=Path(r.config['learning_root'])/'status.json'
        if (time.time()-began>60 and status.exists()
                and read_json(status)['phase'] in ('failed','interrupted')):
            raise RuntimeError('learning peer stopped; no silent indefinite wait')
        r.report();time.sleep(5)
    return read_json(path)


def execute(root):
    r=Runner(root);root=r.root;c=r.config;completed=[]
    try:
        gate=r.jobs(['search_mcts','search_mcts_wide'],[148900001],'gate')
        for job in gate:job.update(split='calibration',seconds=3.)
        gates=r.run(gate,'execution_gate')
        write_json(root/'preflight.json',dict(passed=True,games=gates,source_id=r.launch['source_id'],
                   note='Short execution gate only; not a900s performance screen.'))
        for cycle in range(c['cycles']):
            ready=wait_for(r,Path(c['learning_root'])/f'ready_{cycle}.json',f'await_learning_{cycle}')
            if ready.get('stop'):break
            student=ready['student'];value=ready.get('value')
            if digest(student['model'])!=student['model_sha256']:raise ValueError('student hash changed')
            if value and digest(value['path'])!=value['sha256']:raise ValueError('value hash changed')
            methods=['model','search_habitat','search_mcts','search_mcts_wide']
            if value:methods.append('search_mcts_value')
            offset=cycle*10000
            rows=r.run(r.jobs(methods,range(248000000+offset,248000004+offset),f'c{cycle}_pilot',student,value),f'cycle{cycle}_paired4')
            pilot=report(rows);write_json(root/f'pilot_{cycle}.json',dict(results=pilot,uses_confirmation=False))
            candidates=sorted((m for m in methods if m.startswith('search_mcts')),key=lambda m:-pilot[m]['score'])[:2]
            # First screen always advances two tree variants, even if4 environments
            # are unfavorable. Nothing is screened on sub900s performance.
            rows=r.run(r.jobs(['model','search_habitat',*candidates],
                range(248001000+offset,248001000+offset+c['expanded_environments']),
                f'c{cycle}_expanded',student,value),f'cycle{cycle}_fresh_expansion')
            expanded=report(rows)
            teacher=max(candidates,key=lambda m:expanded[m]['score'])
            eligible=(expanded[teacher]['paired_score']['ci95'][0]>0
                      and expanded[teacher]['score']>=expanded['search_habitat']['score']-4)
            result=dict(results=expanded,tree_candidate=teacher,train_candidate=eligible,
                        improves_original_rollout=(expanded[teacher]['vs_rollout']['ci95'][0]>0
                        and expanded[teacher]['vs_rollout']['delta']>=1),uses_confirmation=False)
            write_json(root/f'expanded_{cycle}.json',result);completed.append(result)
            if not eligible:
                write_json(root/f'expert_{cycle}.json',dict(split='train',games=[],stop=True,
                    reason='tree candidate lacks sufficient full-game quality; retain original rollout and models'))
                break
            seeds=range(148000000+offset,148000000+offset+c['expert_environments'])
            games=r.run(r.jobs([teacher],seeds,f'expert_c{cycle}',student,value),f'cycle{cycle}_actual_train_games')
            write_json(root/f'expert_{cycle}.json',dict(format='v148_train_search_manifest',split='train',
                       teacher=teacher,games=games,student=student,value=value,stop=False))
        write_json(root/'summary.json',dict(status='completed',cycles=completed,completed_unix=time.time(),
                   note='Search development results only; pure model confirmation belongs to learning.'))
        r.phase='completed'
    except BaseException as exc:
        r.phase='interrupted' if isinstance(exc,(InterruptedError,TimeoutError)) else 'failed'
        write_json(root/(r.phase+'.json'),dict(error=repr(exc),time=time.time()));raise
    finally:r.close()
