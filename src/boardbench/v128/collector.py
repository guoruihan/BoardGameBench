"""Same synchronized cohort and RNG ordering; faster environment-side work."""
from boardbench.v121.collector import EnvironmentPool as BasePool, _Shard
from boardbench.v12.encoding import STOP, action_map, feature_vector
from .environment import FastHarmonies


class FastShard(_Shard):
    def __init__(self,slots,arrays):
        super().__init__(slots,arrays);self.maps={}

    def _publish(self,slot):
        if self.arrays['done'][slot]:
            self.arrays['features'][slot].fill(0);self.arrays['masks'][slot].fill(False);self.maps[slot]={}
        else:
            obs=self.observations[slot]
            mapping=action_map(obs,'slots240_v1',True);self.maps[slot]=mapping
            self.arrays['features'][slot]=feature_vector(obs)
            self.arrays['masks'][slot].fill(False)
            self.arrays['masks'][slot,list(mapping)]=True

    def reset(self,seeds):
        for slot in self.slots:
            # reset initializes all state; avoid the redundant constructor reset(0).
            env=self.games.get(slot)
            if env is None:env=FastHarmonies.__new__(FastHarmonies)
            obs,_=env.reset(seeds[slot])
            self.games[slot],self.observations[slot],self.steps[slot]=env,obs,0
            self.arrays['done'][slot]=False;self.arrays['stats'][slot].fill(0);self._publish(slot)

    def step(self):
        for slot in self.slots:
            if self.arrays['done'][slot]:continue
            obs=self.observations[slot];action=self.maps[slot][int(self.arrays['actions'][slot])]
            stopped=action==STOP
            nxt=obs if stopped else self.games[slot].step(action).observation
            self.steps[slot]+=1;done=stopped or nxt['terminated']
            if self.steps[slot]>=256 and not done:raise RuntimeError('game action limit')
            score=nxt['score_breakdown']
            self.arrays['stats'][slot]=((score['total']-obs['score_breakdown']['total'])/150.,
                score['total'],score['animals'],len(nxt['completed_cards']),nxt['terminated'],stopped,
                action.get('type')=='place_animal')
            self.observations[slot]=nxt;self.arrays['done'][slot]=done;self._publish(slot)


class EnvironmentPool(BasePool):
    def __init__(self,count=16,workers=0,timeout=60.):
        super().__init__(count,workers,timeout,shard_class=FastShard)
