"""Offline historical terminal-regret audit, never a policy feature or new test set."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean

from boardbench.artifacts.store import write_json, digest
from boardbench.environments.harmonies import Harmonies
from boardbench.v12.encoding import STOP


def terminal_animal_gains(env, final_action):
    baseline = deepcopy(env)
    if final_action == STOP:
        final = baseline.observe()
    else:
        final = baseline.step(final_action).observation
        if not final['terminated']:
            return []  # A general opportunity is not necessarily a missed reward.
    result=[]
    for action in env.observe()['legal_actions']:
        if action['type']!='place_animal':
            continue
        alternative=deepcopy(env)
        hidden=alternative._private_state()
        after=alternative.step(action).observation
        if alternative._private_state()!=hidden:
            raise ValueError('animal audit unexpectedly touched future randomness')
        if final_action!=STOP:
            if final_action not in after['legal_actions']:
                continue
            after=alternative.step(final_action).observation
            if not after['terminated']:
                continue
        gain=after['score_breakdown']['total']-final['score_breakdown']['total']
        if gain>0:
            result.append({'inserted_action':action,'total_score_gain':gain})
    return result


def audit_rows(rows):
    findings=[]
    for row in rows:
        if row['failed'] or row['reason'] not in ('natural','stopped'):
            continue
        env=Harmonies();env.reset(row['seed'])
        frames=row['frames']
        final_action=STOP if row['reason']=='stopped' else frames[-1]['action']
        prefix=frames if final_action==STOP else frames[:-1]
        for frame in prefix:
            obs=env.step(frame['action']).observation
            if obs!=frame['observation']:
                raise ValueError('historical replay observation mismatch')
        gains=terminal_animal_gains(env,final_action)
        settled=env.observe() if final_action==STOP else env.step(final_action).observation
        if settled['score_breakdown']['total']!=row['score']:
            raise ValueError('historical replay final score mismatch')
        if gains:
            findings.append({'seed':row['seed'],'reason':row['reason'],
                             'best_single_insertion_gain':max(x['total_score_gain'] for x in gains),
                             'counterfactuals':gains})
    return {'attempts':len(rows),'missed_episodes':len(findings),
            'mean_direct_loss_all_attempts':sum(x['best_single_insertion_gain'] for x in findings)/len(rows),
            'findings':findings,'scope':'historical offline audit; extra online computation cost not assessed; not training labels'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',nargs='+',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    rows=[json.loads(line) for path in args.episodes for line in Path(path).read_text().splitlines()]
    report=audit_rows(rows)
    report['sources']={p:digest(p) for p in args.episodes}
    if Path(args.out).exists():
        raise ValueError('new diagnostic output path required')
    write_json(args.out,report)
