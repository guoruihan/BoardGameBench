"""Bounded same-parent curriculum-prefix / official-self-play diagnostic.

No rules or rewards are changed. Prefix scores are observations in the original
full curriculum episode, not an artificially shortened evaluation environment.
"""
import argparse
from collections import Counter
from pathlib import Path
import time
import torch
from boardbench.artifacts.store import digest,source_id,write_json
from boardbench.environments.harmonies_market import HarmoniesMarket
from boardbench.environments.harmonies_duo import HarmoniesDuo
from boardbench.v142.policy import MarketSolver,load_policy
from boardbench.v143.policy import SelfplaySolver,initialize


def matched_prefix(games):
    by_seed={}
    for game in games:by_seed.setdefault(game['seed'],{})[game['setting']]=game
    matched=[panels for panels in by_seed.values()
        if set(panels)=={'curriculum','official_selfplay'}
        and all(p['prefix_reached'] for g in panels.values() for p in g['players'])]
    result=dict(environments=len(matched),turn=11,settings={})
    for setting in ('curriculum','official_selfplay'):
        rows=[p['prefix11'] for pair in matched for p in pair[setting]['players']]
        if rows:result['settings'][setting]={k:sum(p[k] for p in rows)/len(rows)
            for k in ('score','animal','visible_unique_cards','cards_taken')}
    return result


def run(parent,out,environments=16,seed_start=244200000,max_seconds=300):
    if not (1<=environments<=64 and 244200000<=seed_start<seed_start+environments<=244300000
            and 0<max_seconds<=300):raise ValueError('bounded development diagnostic only')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+max_seconds
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    solvers={'curriculum':MarketSolver(load_policy(parent)),'official_selfplay':SelfplaySolver(initialize(parent))}
    result=dict(parent=str(parent),parent_sha256=digest(parent),source_id=source_id(),audit_script_sha256=digest(__file__),
        status='running',note='Same seed labels and weights; trajectories/supply streams diverge between protocols. Not a causal decomposition or final confirmation.',games=[])
    try:
        for seed in range(seed_start,seed_start+environments):
            for setting,solver in solvers.items():
                duo=setting=='official_selfplay';env=HarmoniesDuo() if duo else HarmoniesMarket()
                obs,_=env.reset(seed);solver.start_episode(obs);history=[[],[]] if duo else [[]]
                seen=[set() for _ in history];taken=[0]*len(history);actions=[];trigger=None
                while not obs['terminated']:
                    if time.monotonic()>=deadline:raise TimeoutError('diagnostic wall cap')
                    seat=obs['active_player'] if duo else 0
                    seen[seat].update(c for c in obs['animal_market'] if c is not None)
                    action=solver.decide(obs);actions.append(dict(seat=seat,action=action))
                    taken[seat]+=action['type']=='take_card'
                    before=obs;obs=env.step(action).observation
                    if action['type']=='end_turn':
                        p=obs['players'][seat] if duo else obs;s=obs['score_breakdown'][seat] if duo else obs['score_breakdown']
                        history[seat].append(dict(turn=p['completed_turns'],score=s['total'],animal=s['animals'],
                            empty_cells=sum(c['stack_id']==0 for c in p['board']),visible_unique_cards=len(seen[seat]),cards_taken=taken[seat]))
                        if duo and not before['final_round'] and obs['final_round']:trigger=seat
                env.validate()
                rows=[]
                for seat,h in enumerate(history):
                    final=h[-1];prefix=next((p for p in h if p['turn']==11),final)
                    rows.append(dict(seat=seat,final=final,prefix11=prefix,prefix_reached=final['turn']>=11,turns=h))
                result['games'].append(dict(seed=seed,setting=setting,players=rows,end_reason=obs['end_reason'],
                    final_round_trigger_seat=trigger,actions=actions))
                write_json(out/'progress.json',dict(completed_games=len(result['games']),elapsed_seconds=time.monotonic()-started))
        result['status']='completed'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        result['wall_seconds']=time.monotonic()-started;result['summary']={}
        for setting in solvers:
            games=[g for g in result['games'] if g['setting']==setting]
            rows=[p for g in games for p in g['players']]
            if not rows:continue
            mean=lambda key,phase='final':sum(p[phase][key] for p in rows)/len(rows)
            result['summary'][setting]=dict(environments=len(games),player_episodes=len(rows),
                score=mean('score'),animal=mean('animal'),turns=mean('turn'),
                visible_unique_cards=mean('visible_unique_cards'),cards_taken=mean('cards_taken'),
                prefix11_score=mean('score','prefix11'),prefix11_animal=mean('animal','prefix11'),
                prefix11_reached=sum(p['prefix_reached'] for p in rows),end_reasons=dict(Counter(g['end_reason'] for g in games)))
        result['matched_prefix11']=matched_prefix(result['games'])
        write_json(out/'result.json',result)
    print(result['summary'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--out',required=True)
    p.add_argument('--environments',type=int,default=16);p.add_argument('--max-seconds',type=int,default=300)
    a=p.parse_args();run(a.parent,a.out,a.environments,max_seconds=a.max_seconds)
