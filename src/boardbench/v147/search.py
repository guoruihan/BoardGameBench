"""Alter search contents, never terminal rewards, legality or private information."""
from boardbench.environments.harmonies import STACKS
from boardbench.v123.policies import rank_actions, extension_count
from boardbench.v125.policies import action_key
from boardbench.v146.policy import Habitat, personal_view, terrain_rank
from boardbench.v146.search import TerminalSearch

VARIANTS=('habitat','model','space','balanced','deadline','gentle','stack_finish','balanced_space')


def public_horizon(obs):
    """Heuristic opponent board-fill time, not an oracle or rule transition."""
    other=obs['players'][1-obs['active_player']]
    empty=sum(c['stack_id']==0 for c in other['board'])
    turns=other['completed_turns']
    fill_per_turn=(23-empty)/max(1,turns)
    return min(sum(obs['remaining_tokens'].values())/6+1,
               max(0.,(empty-2)/max(1.,fill_per_turn)+.5))


class StrategicHabitat(Habitat):
    def __init__(self,variant):
        if variant not in VARIANTS:raise ValueError('unknown search variant')
        super().__init__();self.variant=variant

    def rank(self,obs):
        if self.variant=='habitat':return super().rank(obs)
        view=personal_view(obs);base=terrain_rank(view);empty=sum(c['stack_id']==0 for c in view['board'])
        horizon=public_horizon(obs)
        turns=int(horizon) if self.variant in ('deadline','stack_finish') else sum(view['remaining_tokens'].values())//6
        ranked=rank_actions(view,'habitat',goal_function=self.goals.goals,base_ranking=base,future_turns=turns)
        terrain={action_key(a):v for v,a in base};values=[]
        for value,action in ranked:
            kind=action['type'];v=value
            if self.variant in ('balanced','balanced_space','gentle') and kind in ('place_token','choose_bundle'):
                fraction=.65 if self.variant!='gentle' else .85
                v=terrain[action_key(action)]+fraction*(value-terrain[action_key(action)])
            if self.variant in ('space','balanced_space','gentle','stack_finish'):
                # Extra opening cost is only a rollout heuristic. Near an opponent
                # finish there is little reason to preserve cells for nonexistent turns.
                scale=(.65 if self.variant=='gentle' else 1.3)*min(1.,horizon/2)
                pressure=min(2.,max(0.,(10-empty)/4))
                if kind=='place_token':
                    cell=view['board'][action['cell_id']]
                    v+=(-scale*(1+pressure) if cell['stack_id']==0 else .3*scale)
                elif kind=='place_animal':
                    v-=.15*scale*extension_count(view['board'][action['anchor_cell_id']])
            if self.variant=='stack_finish' and kind=='place_token' and empty<=6:
                # Prefer actual immediate terrain increments, without forbidding
                # low-scoring setup actions or changing final score selection.
                v+=.4*terrain[action_key(action)]
            values.append((v,action))
        return sorted(values,key=lambda pair:-pair[0])


class Search(TerminalSearch):
    def __init__(self,focal,opponent,seat,continuation,seconds=900.,seed=0,**kwargs):
        if continuation not in VARIANTS:raise ValueError('unknown full-game continuation')
        super().__init__(focal,opponent,seat,'model' if continuation=='model' else 'habitat',seconds,seed,**kwargs)
        self.variant=continuation
        self.habitat=StrategicHabitat('habitat' if continuation=='model' else continuation)

    def decide(self,obs,planners):
        action=super().decide(obs,planners)
        self.last.update(variant=self.variant,opponent_horizon_estimate=public_horizon(obs))
        return action
