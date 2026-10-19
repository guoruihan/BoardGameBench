"""Freeze finalized V1.2.4 references and compare exact environment/seed panels.

Historical mixed-load scores are contextual, not substitutes for same-batch
controls. Repetitions are averaged within environment, never pseudoreplicated.
"""
import argparse
import math
from pathlib import Path
from statistics import mean

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.v122.report import paired

REFERENCE_ARMS = ('original_900s', 'space_900s', 'habitat_900s')


def validate_reference(reference, plan):
    if (reference.get('schema') != 1 or reference['split'] != 'development'
            or plan['split'] != 'development'
            or reference['rules_version'] != plan['rules_version']
            or reference['split_sha256'] != plan['split_sha256']):
        raise ValueError('historical rules/split mismatch')
    if (reference['environment_seeds'] != plan['environment_seeds']
            or reference['search_seeds'] != plan['search_seeds']):
        raise ValueError('historical environment/search seeds must match exactly')
    arms = reference['arms']
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(REFERENCE_ARMS):
        raise ValueError('historical arms must be unique')
    if any(a['params'].get('mode') != 'timed' or a['params'].get('seconds') != 900 for a in plan['arms']):
        # Legacy plans also contain60s/300s/fixed/greedy arms; their freeze step
        # validates against a filtered plan below, never calls them900s controls.
        raise ValueError('historical comparison requires900-second current arms')
    expected = {(a,e,s) for a in arms for e in reference['environment_seeds']
                for s in reference['search_seeds']}
    found = set()
    for row in reference['rows']:
        key = (row['arm'],row['environment_seed'],row['search_seed'])
        if key in found or key not in expected:
            raise ValueError('duplicate or unexpected historical attempt')
        found.add(key)
        if (row['params'].get('mode') != 'timed' or row['params'].get('seconds') != 900
                or row['params'].get('continuation') != row['arm'].removesuffix('_900s')
                or row['status'] != 'completed' or not row['natural_terminal']
                or not math.isfinite(row['score']) or row['score'] != row['components']['total']
                or not math.isfinite(row['decision_seconds']) or row['decision_seconds'] <= 0):
            raise ValueError('historical reference needs valid true-terminal 900-second results')
    if found != expected:
        raise ValueError('historical reference is missing search repetitions')


def freeze(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('refusing to overwrite historical snapshot')
    plan, jobs = read_json(run/'plan.json'), read_json(run/'jobs.json')
    origin = read_json(run/'origin.json')
    parent = Path(origin['parent'])
    if source_id(parent/'source') != origin['policy_source_id']:
        raise ValueError('historical policy source hash mismatch')
    rows, files = {}, []
    expected_jobs = {j['id']:j for j in jobs if j['arm'] in REFERENCE_ARMS}
    for directory in (parent, run):
        state = read_json(directory/'status.json')
        if state.get('active_count') or state['phase'] == 'running' or not (directory/'watchdog_result.json').exists():
            raise ValueError('historical runs must be finalized and inactive')
        for path in sorted((directory/'resolved').glob('*.json')):
            if path.stem not in expected_jobs:
                continue
            row, job = read_json(path), expected_jobs[path.stem]
            if row['id'] in rows:
                raise ValueError('duplicate historical execution ownership')
            if any(row[k] != job[k] for k in ('id','arm','environment_seed','search_seed','params','method','multiplier')):
                raise ValueError('historical result differs from frozen job')
            rows[row['id']] = {k:row[k] for k in ('id','arm','environment_seed','search_seed','params',
                'status','natural_terminal','score','components','decision_seconds')}
            files.append({'path':str(path),'sha256':digest(path)})
    if set(rows) != set(expected_jobs):
        raise ValueError('incomplete historical ownership union')
    reference = {'schema':1,'run':str(run),'parent':str(parent),'source_id':origin['policy_source_id'],
        'origin_sha256':digest(run/'origin.json'),'plan_sha256':digest(run/'plan.json'),
        'jobs_sha256':digest(run/'jobs.json'),'resolved_files':files,
        'split':plan['split'],'rules_version':plan['rules_version'],'split_sha256':plan['split_sha256'],
        'environment_seeds':plan['environment_seeds'],'search_seeds':plan['search_seeds'],
        'arms':list(REFERENCE_ARMS),'rows':list(rows.values()),
        'caveat':'Historical expanded run mixed CPU loads; use same-batch rerun controls for primary conclusions.'}
    validate_reference(reference,{**plan,'arms':[a for a in plan['arms'] if a['id'] in REFERENCE_ARMS]})
    write_json(destination,reference)
    return reference


def comparisons(plan, summary):
    reference = plan['historical_reference']
    validate_reference(reference,plan)
    # The new report has already checked all job identities and complete panels.
    n = summary['balanced_panel_size']
    selected = plan['environment_seeds'][:n]
    old = {}
    for arm in reference['arms']:
        rows = [r for r in reference['rows'] if r['arm']==arm and r['environment_seed'] in selected]
        old[arm] = {'by_environment':{str(e):mean(r['score'] for r in rows if r['environment_seed']==e)
                                    for e in selected},
                    'mean':mean(r['score'] for r in rows) if rows else None,
                    'decision_seconds':mean(r['decision_seconds'] for r in rows) if rows else None}
    result = {}
    for arm, cell in summary['balanced_panel'].items():
        for previous, values in old.items():
            comparison = paired(cell['by_environment'],values['by_environment'])
            ratio = cell['decision_seconds']/values['decision_seconds'] if n else None
            comparison.update(current_mean=cell['mean'],historical_mean=values['mean'],
                actual_time_ratio=ratio,actual_mean_times_within_10_percent=ratio is not None and .9<=ratio<=1.1,
                same_batch_comparison=False)
            result[f'{arm}-vs-historical-{previous}'] = comparison
    return {'balanced_environments':n,'search_repetitions':len(plan['search_seeds']),
            'source_id':reference['source_id'],'caveat':reference['caveat'],
            'historical_cells':old,'comparisons':result}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();r=freeze(args.run,args.out)
    print({'snapshot':args.out,'sha256':digest(args.out),'rows':len(r['rows']),
           'environment_count':len(r['environment_seeds']),
           'means':{a:mean(x['score'] for x in r['rows'] if x['arm']==a) for a in r['arms']}})


if __name__=='__main__':
    main()
