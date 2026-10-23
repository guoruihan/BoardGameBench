"""Three evidence-driven CPU search panels, then retained-teacher training data."""
from pathlib import Path
import statistics
import time
from boardbench.artifacts.store import read_json,write_json
from boardbench.v146.campaign import Runner as BaseRunner
from boardbench.v146.report import summarize,paired


class Runner(BaseRunner):
    WORKER_MODULE='boardbench.v147.worker'

    def jobs(self,methods,seeds,stage):
        jobs=super().jobs(methods,seeds,'expert' if stage.startswith('expert') else stage)
        for job in jobs:
            job['stage']=stage;job['id']=f'{stage}_{job["method"]}_e{job["environment_seed"]}_s{job["seat"]}'
        return jobs


def report(rows):
    result=summarize(rows);baseline=[r for r in rows if r['method']=='search_habitat']
    for method,metric in result.items():
        selected=[r for r in rows if r['method']==method]
        metric['vs_v146_search']=paired(selected,baseline)
        metric['self_end_fraction']=statistics.mean(r['first_end_trigger']=='self_board' for r in selected)
    return result


def refine(first):
    contenders=sorted((m for m in first if m.startswith('search_') and m!='search_habitat'),key=lambda m:-first[m]['score'])
    winner=contenders[0]
    # If space protection raises terrain but loses animals, soften it; otherwise
    # combine the strongest terrain/animal balancing with a space intervention.
    space=first['search_space'];base=first['search_habitat']
    new='search_gentle' if space['animal']<base['animal']-4 else 'search_balanced_space'
    return list(dict.fromkeys(['model','search_habitat',winner,new,'search_stack_finish']))


def execute(root):
    r=Runner(root);root=r.root;c=r.config
    try:
        # Short checks test execution only, never rank900s strategies by3s scores.
        gate_jobs=r.jobs(['search_space','search_balanced','search_deadline'],[147900001],'gate')
        for job in gate_jobs:job.update(split='calibration',seconds=3.)
        gates=r.run(gate_jobs,'cpu_execution_gate')
        write_json(root/'preflight.json',dict(passed=True,games=gates,source_id=r.launch['source_id']))
        first=r.run(r.jobs(['model','search_habitat','search_model','search_space','search_balanced','search_deadline'],
                          range(247000000,247000004),'round1'),'round1_hypotheses4')
        s1=report(first);write_json(root/'round1.json',dict(results=s1,uses_confirmation=False))
        methods=refine(s1)
        write_json(root/'decision_round2.json',dict(methods=methods,rationale='soften animal loss or combine space and terrain; retain best first-round alternative',uses_confirmation=False))
        second=r.run(r.jobs(methods,range(247000100,247000104),'round2'),'round2_refinement4')
        s2=report(second);write_json(root/'round2.json',dict(results=s2,uses_confirmation=False))
        promising=sorted((m for m in s2 if m.startswith('search_') and m!='search_habitat'),key=lambda m:-s2[m]['score'])[:2]
        expanded=r.run(r.jobs(['model','search_habitat',*promising],range(247001000,247001016),'expanded'),'round3_fresh16')
        s3=report(expanded)
        # Retain the original if gains do not clear a fresh-panel descriptive CI.
        qualified=[m for m in promising if s3[m]['vs_v146_search']['ci95'][0]>0 and s3[m]['score']>s3['search_habitat']['score']+1]
        teacher=max(qualified,key=lambda m:s3[m]['score']) if qualified else 'search_habitat'
        complementary=max(promising,key=lambda m:s3[m]['score'])
        candidate=s3[complementary]
        if (candidate['paired_score']['ci95'][0]<=0 or candidate['score']<s3[teacher]['score']-4
                or candidate['animal']<=s3['model']['animal']):
            complementary=teacher  # Do not inject clearly weaker teachers into BC.
        write_json(root/'expanded.json',dict(results=s3,teacher=teacher,complementary=complementary,uses_confirmation=False,
            note='Development selection, not independent final confirmation; negative variants retained.'))
        # Main128 and complementary64 actual expert games. No hindsight score filter.
        expert=r.run(r.jobs([teacher],range(147000000,147000064),'expert_main'),'expert_main128')
        write_json(root/'expert_main_manifest.json',dict(format='v147_train_only_experts',split='train',teacher=teacher,games=expert))
        extra=r.run(r.jobs([complementary],range(147000100,147000132),'expert_complement'),'expert_complement64')
        write_json(root/'expert_manifest.json',dict(format='v147_train_only_experts',split='train',teacher=teacher,
            complementary=complementary,games=expert+extra))
        write_json(root/'summary.json',dict(status='completed',results=s3,teacher=teacher,complementary=complementary,
            expert_games=len(expert)+len(extra),uses_confirmation=False,completed_unix=time.time()))
        r.phase='completed'
    except BaseException as exc:
        r.phase='interrupted' if isinstance(exc,(InterruptedError,TimeoutError)) else 'failed'
        write_json(root/(r.phase+'.json'),dict(error=repr(exc),time=time.time()));raise
    finally:r.close()
