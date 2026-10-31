"""Bounded CPU measurements on identical player-visible development states."""
import argparse,json,time
from pathlib import Path
import numpy as np

from .play import action_features,make_policy,write_json
from .reference_search import guarded_spec
from .registry import game_class
from .rich_observation import PublicTreeEncoder
from .simulation import clone,step_legal,terminated


def benchmark(game):
    cls=game_class(game);env=cls();env.reset(172090001)
    policies=[make_policy(cls,guarded_spec(game,{'kind':'heuristic'}),p) for p in (0,1)]
    states=[];observations=[];traces=[]
    for k in range(500):
        if terminated(env):break
        obs=env.observe();actions=env.legal_actions()
        if not actions:break
        chosen=policies[env.current_player].decide(obs,actions)
        if k%11==0 and len(states)<32:
            states.append(clone(env));observations.append((obs,actions));traces.append(actions.index(chosen))
        env.step(chosen)
    def median(fn):
        values=[]
        for _ in range(5):
            t=time.perf_counter();fn();values.append(time.perf_counter()-t)
        return sorted(values)[2]
    public=fast=0.
    for state,index in zip(states,traces):
        actions=state.legal_actions()
        for _ in range(3):
            a=clone(state);b=clone(state)
            t=time.perf_counter();a.step(actions[index]);public+=time.perf_counter()-t
            t=time.perf_counter();step_legal(b,actions,index);fast+=time.perf_counter()-t
            if a.snapshot()!=b.snapshot():raise AssertionError('Transition/RNG mismatch')
    cached,reference=PublicTreeEncoder(game),PublicTreeEncoder(game,cache=False)
    for obs,actions in observations:
        np.testing.assert_array_equal(cached.encode(obs),reference.encode(obs))
        a,x=cached.encode_actions(actions);b,y=reference.encode_actions(actions)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(x,y)
    def encoding(encoder):
        for obs,actions in observations:encoder.encode(obs);encoder.encode_actions(actions)
    old=median(lambda:[action_features(cls,o,a) for o,a in observations])
    uncached=median(lambda:encoding(reference));optimized=median(lambda:encoding(cached))
    n=len(states)
    return dict(game=game,states=n,trajectory_decisions=k+1,terminal=terminated(env),
        public_step_us=public/(n*3)*1e6,fast_step_us=fast/(n*3)*1e6,step_speedup=public/fast,
        old_lossy24_ms=old/n*1e3,rich_uncached_ms=uncached/n*1e3,rich_cached_ms=optimized/n*1e3,
        rich_encoding_speedup=uncached/optimized,
        mean_state_nodes=sum(len(cached.encode(o)) for o,a in observations)/n,
        mean_legal_actions=sum(len(a) for o,a in observations)/n,
        observation_metadata=cached.metadata)


def main():
    p=argparse.ArgumentParser();p.add_argument('--game',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();row=benchmark(a.game);write_json(a.output,row);print(json.dumps(row))

if __name__=='__main__':main()
