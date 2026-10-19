"""Export auditable public-state search demonstrations, including target semantics."""
import argparse
import gzip
import json
from pathlib import Path

from boardbench.artifacts.store import digest, read_json, write_json


def export(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('refusing to overwrite an existing dataset')
    plan = read_json(run / 'plan.json')
    status = read_json(run / 'status.json')
    if status['phase'] == 'running' or status.get('active_count',0):
        raise ValueError('export requires a finalized task')
    if plan['split'] not in ('development','train_calibration'):
        raise ValueError('no confirmation/test training exports')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial')
    records = games = failed = 0
    with gzip.open(temporary, 'xt', encoding='utf-8') as stream:
        for path in sorted((run/'resolved').glob('*.json')):
            result=read_json(path)
            if result['status']!='completed' or not result['natural_terminal']:
                failed+=1
                continue
            episode_root=Path(result.get('inherited_from',str(run)))
            source_id=read_json(episode_root/'allocation.json')['source_id']
            trajectory=episode_root/'episodes'/result['id']/'decisions.jsonl'
            previous=None; total_delta=0
            with trajectory.open() as source:
                for line in source:
                    row=json.loads(line)
                    before,after=row['observation_before'],row['observation']
                    if (row['action'] not in before['legal_actions'] or
                            previous is not None and before!=previous):
                        raise ValueError('state/action alignment mismatch')
                    delta=after['score_breakdown']['total']-before['score_breakdown']['total']
                    if row['raw_score_delta']!=delta:
                        raise ValueError('raw score delta mismatch')
                    total_delta+=delta
                    target={
                        'run':str(episode_root),'job_id':result['id'],'source_id':source_id,
                        'split':plan['split'],'rules_version':before['rules_version'],
                        'environment_seed':result['environment_seed'],'search_seed':result['search_seed'],
                        'variant':result['params']['continuation'],'decision':row['decision'],
                        'observation':before,'action':row['action'],'next_observation':after,
                        'raw_score_delta':delta,'terminal_total_score':result['score'],
                        'environment_reward':result['score'] if after['terminated'] else 0,
                        'score_delta_return_to_go':result['score']-before['score_breakdown']['total'],
                        'candidates':row['search']['candidates'],
                        'candidate_target_semantics':'absolute terminal score under this continuation; unestimated is null, not zero; not optimal Q',
                        'solver_seconds':row['seconds'],'interrupted_sampling_round':row['search']['interrupted_round']}
                    stream.write(json.dumps(target,separators=(',',':'),allow_nan=False)+'\n')
                    records+=1;previous=after
            if previous is None or not previous['terminated'] or total_delta!=result['score']:
                raise ValueError('trajectory is not an aligned complete game')
            games+=1
    if destination.exists():
        raise ValueError('dataset target appeared during export')
    temporary.rename(destination)
    manifest={'run':str(run),'dataset':str(destination),'sha256':digest(destination),
              'records':records,'completed_games':games,'excluded_failed_games':failed,
              'split':plan['split'],'training_permission':'development adaptation only; no test claim'}
    write_json(destination.with_name(destination.name+'.manifest.json'),manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(export(a.run,a.out)))


if __name__=='__main__':
    main()
