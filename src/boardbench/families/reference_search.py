"""Terminal rollout reference with public beliefs and cumulative game budgets.

This is approximate policy improvement, not optimal search. All root candidates
in a round share the same sampled future and continuation style; incomplete
rounds never contribute a fabricated terminal value. The real engine is only
accepted by the trusted sampled_world bridge, not by continuation policies.
"""
from copy import copy,deepcopy
import math
import random
import time

from .play import action_features,make_policy
from .public_world import sampled_world
from .simulation import clone,step_legal,terminated

# Additional emphasis on separately interpretable public features. These are
# candidate/continuation proposals, never alterations to official score/reward.
PROGRESS = {
 'cascadia':{1:1.5,3:.4,5:.8,6:1.},
 'calico':{2:.5,3:.5,4:.5},
 'patchwork':{3:1.5,8:-.7,9:.15,10:4.},
 'century_spice_road':{2:1.5,3:1.,4:.15},
 'jaipur':{2:.5,3:.4,4:.3},
 'splendor':{2:1.5,5:1.,7:.5},
 'gizmos':{2:1.2,4:.4,3:-.3},
 'machi_koro':{4:8.,2:.5},
 'clever':{2:2.,9:1.,10:1.,12:1.},
 'century_a_new_world':{6:8.,7:2.,9:3.,11:-2.},
 'stone_age':{2:1.,3:1.,4:.5,7:.5,8:.2},
 'targi':{6:.5,7:.7,12:1.5,14:2.},
 'dominion':{4:1.,6:2.},
 'el_dorado':{1:1.,6:.4,7:.4}, 'toy_battle':{1:3.,8:1.,9:1.,10:.7},
}
HORIZONS = dict(cascadia=65,calico=46,patchwork=28,harmonies=85,
 century_spice_road=100,jaipur=35,splendor=65,gizmos=100,machi_koro=100,
 clever=100,century_a_new_world=100,stone_age=160,targi=160,dominion=330,
 el_dorado=160,toy_battle=45)


def paired_selection(sums,delta_sums,delta_squares,rounds,z):
    """Heuristic uncertainty gate, not a simultaneous statistical guarantee."""
    if not rounds:return 0
    if z<=0:return max(range(len(sums)),key=lambda i:sums[i])
    if rounds<4:return 0
    eligible=[0]
    for i in range(1,len(sums)):
        error=math.sqrt(max(0.,delta_squares[i]-delta_sums[i]**2/rounds)/(rounds-1)/rounds)
        if delta_sums[i]/rounds>z*error+1e-8:eligible.append(i)
    return max(eligible,key=lambda i:sums[i])


def guarded_spec(game,spec):
    spec=dict(spec)
    if game=='targi':spec['decision_guard']='targi_trader_acyclic_v1'
    if game=='century_a_new_world':spec['decision_guard']='century_new_world_public_repeat_v1'
    return spec


def fresh_policy(policy):
    """Independent branch memory; immutable model weights can be shared."""
    if hasattr(policy,'policy'):
        return type(policy)(fresh_policy(policy.policy))
    result=copy(policy)
    if hasattr(policy,'rng'):result.rng=deepcopy(policy.rng)
    if hasattr(policy,'records'):result.records=[]
    if hasattr(policy,'solver'):
        result.solver=copy(policy.solver)
        result.solver.planners=deepcopy(policy.solver.planners)
    return result


class ProposalPolicy:
    def __init__(self,cls,weights=None,progress=False,habitat=None):
        self.cls=cls;self.weights=weights or cls.default_weights
        self.progress=progress;self.habitat=habitat

    def rank(self,obs,actions):
        if self.habitat is not None:
            rank=[a for _,a in self.habitat.rank(obs)]
            return [a for a in rank if a in actions]
        rows=action_features(self.cls,obs,actions)
        extra=PROGRESS.get(self.cls.game_id,{}) if self.progress else {}
        values=[sum(x*w for x,w in zip(row,self.weights))+sum(row[i]*w for i,w in extra.items()) for row in rows]
        return [actions[i] for i in sorted(range(len(actions)),key=lambda i:(-values[i],i))]

    def decide(self,obs,actions):return self.rank(obs,actions)[0]


def with_guard(cls,policy):
    if cls.game_id=='targi':
        from .targi_policy import TargiTraderAcyclicPolicy
        return TargiTraderAcyclicPolicy(policy)
    if cls.game_id=='century_a_new_world':
        from .century_new_world_policy import CenturyNewWorldRepeatPolicy
        return CenturyNewWorldRepeatPolicy(policy)
    return policy


class TerminalReference:
    def __init__(self,cls,rl_spec,*,seat,seconds=900.,seed=172,weights=None,width=6,
                 objective='win',max_rounds=None,max_simulation_steps=4000,clock=time.monotonic,
                 continuation='portfolio',improvement_z=0.,budget_reserve=2.):
        if seat not in (0,1) or not math.isfinite(seconds) or seconds<=0 or not 2<=width<=32:
            raise ValueError('Invalid bounded search configuration')
        if objective not in ('win','score'):raise ValueError('Invalid objective')
        if continuation not in ('portfolio','rl','expert'):raise ValueError('Invalid continuation')
        if not math.isfinite(improvement_z) or improvement_z<0:raise ValueError('Invalid uncertainty gate')
        if not math.isfinite(budget_reserve) or budget_reserve<0:raise ValueError('Invalid time reserve')
        if objective=='score' and cls.game_id in ('el_dorado','machi_koro','toy_battle'):
            raise ValueError('Race games do not have a scalar score objective')
        self.cls=cls;self.seat=seat;self.seconds=float(seconds);self.width=width
        self.rng=random.Random(seed);self.clock=clock;self.max_rounds=max_rounds
        self.max_simulation_steps=max_simulation_steps;self.objective=objective
        self.continuation=continuation
        self.improvement_z=float(improvement_z)
        self.budget_reserve=min(float(budget_reserve),self.seconds*.02)
        self.base=make_policy(cls,guarded_spec(cls.game_id,rl_spec),seed)
        habitat=None
        if cls.game_id=='harmonies':
            from boardbench.v146.policy import Habitat
            habitat=Habitat()
        self.expert=ProposalPolicy(cls,weights,habitat=habitat)
        self.progress=ProposalPolicy(cls,progress=True)
        self.spent=0.;self.horizon=float(HORIZONS[cls.game_id]);self.last={}
        self.total_steps=0;self.total_rounds=0;self.fallbacks=0;self.decisions=0

    def roots(self,obs,actions):
        primary=self.base.decide(obs,actions)
        expert=self.expert.rank(obs,actions);progress=self.progress.rank(obs,actions)
        # One best candidate per actual action family reduces premature loss of
        # acquire/rest/engine/score alternatives. Remaining slots preserve rank.
        groups=[];seen=set()
        for a in expert:
            kind=(a['type'],a.get('card') if a['type'] in ('buy','play_action','acquire') else a.get('patch'))
            if kind not in seen:groups.append(a);seen.add(kind)
        result=[]
        for stream in ([primary],expert[:1],progress[:1],groups[:2]):
            for a in stream:
                if a not in result:result.append(a)
        for rank in range(max(len(expert),len(progress))):
            for stream in (expert,progress):
                if rank<len(stream) and stream[rank] not in result:result.append(stream[rank])
            if len(result)>=self.width:break
        return result[:self.width]

    def _round(self,world,roots,round_index,deadline):
        # Alternate a domain heuristic, its long-term emphasis, and the retained
        # RL itself. Harmonies keeps its proven habitat rollout in two of three.
        mode=round_index%3 if self.continuation=='portfolio' else (2 if self.continuation=='rl' else 0)
        values=[];scores=[];own_counts=[];steps=0
        for action in roots:
            if self.clock()>=deadline:return None,steps,[]
            branch=clone(world)
            policies=[fresh_policy(self.base),fresh_policy(self.base)]
            if mode!=2:
                proposal=self.expert if mode==0 or self.cls.game_id=='harmonies' else self.progress
                policies[self.seat]=with_guard(self.cls,proposal)
            legal=branch.legal_actions();step_legal(branch,legal,legal.index(action));steps+=1;own=1
            for _ in range(self.max_simulation_steps):
                if terminated(branch):break
                if self.clock()>=deadline:return None,steps,[]
                seat=branch.current_player;obs=branch.observe(seat);legal=branch.legal_actions(seat)
                if not legal:return None,steps,[]
                choice=policies[seat].decide(obs,legal)
                step_legal(branch,legal,legal.index(choice));steps+=1;own+=seat==self.seat
            if not terminated(branch):return None,steps,[]
            out=branch.outcome();win=.5 if len(out['winners'])!=1 else float(self.seat in out['winners'])
            score=out['scores'][self.seat];other=out['scores'][1-self.seat]
            margin=0. if score is None or other is None else math.tanh((score-other)/max(1.,abs(score)+abs(other)))
            value=score if self.objective=='score' else win+.05*margin
            values.append(value);scores.append([score,other]);own_counts.append(own)
        return (values,scores),steps,own_counts

    def decide(self,env):
        if env.current_player!=self.seat:raise ValueError('Wrong search seat')
        began=self.clock();obs=env.observe(self.seat);actions=env.legal_actions(self.seat)
        if not actions:raise ValueError('No legal root actions')
        remaining=max(0.,self.seconds-self.budget_reserve-self.spent)
        # Leave enough time for later fallback policy calls. Once depleted,
        # selecting an already enumerated legal action requires no model call.
        roots=self.roots(obs,actions) if len(actions)>1 and remaining>.1 else actions[:1]
        allowance=max(0.,remaining-.002*self.horizon)*min(1.,2/max(2.,self.horizon))
        deadline=min(began+allowance,began+remaining)
        sums=[0.]*len(roots);squares=[0.]*len(roots);score_sums=[[0.,0.] for _ in roots]
        delta_sums=[0.]*len(roots);delta_squares=[0.]*len(roots)
        rounds=steps=0;own_counts=[]
        while len(roots)>1 and self.clock()<deadline and (self.max_rounds is None or rounds<self.max_rounds):
            world=sampled_world(env,self.seat,self.rng.getrandbits(63),verify=True)
            result,count,own=self._round(world,roots,rounds,deadline);steps+=count
            if result is None:break
            values,scores=result
            for i,value in enumerate(values):
                sums[i]+=value;squares[i]+=value*value
                delta=value-values[0];delta_sums[i]+=delta;delta_squares[i]+=delta*delta
                score_sums[i]=[x+y if y is not None else None for x,y in zip(score_sums[i],scores[i])]
            rounds+=1;own_counts+=own
        selected=paired_selection(sums,delta_sums,delta_squares,rounds,self.improvement_z)
        elapsed=self.clock()-began;self.spent+=elapsed
        fallback=not rounds and len(actions)>1
        self.decisions+=1;self.total_steps+=steps;self.total_rounds+=rounds;self.fallbacks+=int(fallback)
        self.last=dict(completed_rounds=rounds,simulation_steps=steps,selected_rank=selected,
            fallback=fallback,forced_action=len(actions)==1,decision_seconds=elapsed,
            cumulative_decision_seconds=self.spent,objective=self.objective,continuation=self.continuation,
            improvement_z=self.improvement_z,
            candidates=[dict(action=a,samples=rounds,mean=sums[i]/rounds if rounds else None,
                terminal_scores=[x/rounds if x is not None else None for x in score_sums[i]] if rounds else None,
                paired_delta=delta_sums[i]/rounds if rounds else None,
                paired_standard_error=math.sqrt(max(0.,delta_squares[i]-delta_sums[i]**2/rounds)/(rounds-1)/rounds) if rounds>1 else None,
                variance=max(0.,(squares[i]-sums[i]**2/rounds)/(rounds-1)) if rounds>1 else None)
                for i,a in enumerate(roots)])
        self.horizon=max(1.,.2*self.horizon+.8*sum(own_counts)/len(own_counts)-1) if own_counts else max(1.,self.horizon-1)
        return dict(roots[selected])
