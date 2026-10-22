"""Paired environment-cluster summaries; development never becomes training data."""
import math
import statistics


def paired(rows,baseline,metric='score'):
    def index(data):
        values={(r['environment_seed'],r['seat'],r['search_seed']):r for r in data}
        if len(values)!=len(data):raise ValueError('duplicate evaluation keys')
        return values
    a,b=index(rows),index(baseline)
    if a.keys()!=b.keys():raise ValueError('unpaired panel')
    clusters={}
    for k in a:clusters.setdefault(k[0],[]).append(a[k][metric]-b[k][metric])
    if not clusters:raise ValueError('empty panel')
    values=[statistics.mean(v) for v in clusters.values()]
    delta=statistics.mean(values)
    sem=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0.
    return dict(delta=delta,ci95=[delta-1.96*sem,delta+1.96*sem],environments=len(values),
                wins=sum(v>0 for v in values),ties=values.count(0),losses=sum(v<0 for v in values))


def summarize(rows):
    methods={r['method'] for r in rows}
    base=[r for r in rows if r['method']=='model'];output={}
    for method in sorted(methods):
        selected=[r for r in rows if r['method']==method]
        summary={k:statistics.mean(r[k] for r in selected) for k in
                 ('score','animal','terrain','turns','animals_placed','decision_seconds','completed_rounds','fallback_decisions')}
        summary.update(games=len(selected),environments=len({r['environment_seed'] for r in selected}))
        if method!='model':
            summary['paired_score']=paired(selected,base)
            summary['paired_animal']=paired(selected,base,'animal')
        output[method]=summary
    return output


def pilot_choices(summary):
    # Four environments only screen hypotheses; never call this a reliable gain.
    return [m for m,r in sorted(summary.items(),key=lambda kv:-kv[1]['score'])
            if m.startswith('search_') and r['paired_score']['delta']>0
            and r['paired_score']['wins']>=2][:2]


def teacher_choice(summary):
    # Only NEW16 development environments supply this gate. It is not a formal
    # test and its descriptive intervals do not correct multiple comparisons.
    qualified=[m for m,r in summary.items() if m.startswith('search_')
        and r['paired_score']['delta']>=2 and r['paired_score']['ci95'][0]>0
        and r['paired_animal']['delta']>0]
    return max(qualified,key=lambda m:summary[m]['score']) if qualified else None
