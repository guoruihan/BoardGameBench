"""Scores and paired uncertainty cluster both seats/opponents by environment."""
import numpy as np
from boardbench.v140.training import summarize


def summarize_panel(games):
    result=summarize(games)
    seeds=sorted({g['seed'] for g in games})
    for key in ('score','animal','opponent_score','turns','tokens','animals_placed'):
        result[key+'_mean']=float(np.mean([g[key] for g in games]))
        clusters=np.array([np.mean([g[key] for g in games if g['seed']==s]) for s in seeds])
        result['paired_'+key+'_sem']=float(clusters.std(ddof=1)/np.sqrt(len(clusters))) if len(clusters)>1 else None
    return result


def paired_delta(candidate,baseline):
    key=lambda g:(g['seed'],g['seat'],g['evaluation_opponent'])
    a={key(g):g for g in candidate['per_game']};b={key(g):g for g in baseline['per_game']}
    if len(a)!=len(candidate['per_game']) or a.keys()!=b.keys():raise ValueError('paired panels differ')
    result={};seeds=sorted({k[0] for k in a})
    for metric in ('score','animal','margin','win_credit'):
        values=np.array([np.mean([a[k][metric]-b[k][metric] for k in a if k[0]==s]) for s in seeds])
        mean=float(values.mean());sem=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
        result[metric]=dict(delta=mean,sem=sem,ci95=[mean-1.96*sem,mean+1.96*sem])
    return dict(environments=len(seeds),games=len(a),metrics=result)
