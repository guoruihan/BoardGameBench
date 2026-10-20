"""Fast public observations and compiled predicates; transitions remain original.

No private supply or RNG enters encoding. Each returned observation owns its
mutable containers, exactly like the reference deep-copy implementation.
"""
from copy import deepcopy
from boardbench.contracts import StepResult
from boardbench.environments.harmonies import Harmonies, PATTERNS


FORBIDDEN = {card: [[(orientation, sum(((32767 ^ sum(1 << v for v in allowed)) << (15*cell))
                                    for cell,allowed in pattern))
                    for orientation,pattern in orientations.items()]
                   for orientations in anchors] for card,anchors in PATTERNS.items()}


def public_copy(state):
    result={}
    for key,value in state.items():
        if key in ('board','active_cards','completed_cards'):
            result[key]=[v.copy() for v in value]
        elif key=='token_market':
            result[key]=[v.copy() if v is not None else None for v in value]
        elif key in ('pending_tokens','discarded_tokens','remaining_tokens'):
            result[key]=value.copy()
        elif key in ('animal_market','discarded_cards','remaining_cards'):
            result[key]=value.copy()
        else:
            # Scalars are immutable; fallback preserves future schema changes.
            result[key]=value if value is None or isinstance(value,(str,int,float,bool)) else deepcopy(value)
    return result


class FastHarmonies(Harmonies):
    def _observation(self,score):
        return {**public_copy(self.s),'task_id':self.task_id,'rules_version':self.rules_version,
                'legal_actions':self.legal_actions(),'score_breakdown':score.copy()}

    def observe(self):
        return self._observation(self.score_breakdown())

    def _result(self,reason=None):
        self.s['decision_index']+=1
        score=self.score_breakdown()
        info={'turn_index':self.s['turn_index'],'decision_index':self.s['decision_index']}
        if self.s['terminated']:
            self.s['score']=score['total'];info.update(score=score['total'],score_breakdown=score,reason=reason)
        return StepResult(self._observation(score),score['total'] if self.s['terminated'] else 0,
                          self.s['terminated'],info=info)

    def legal_actions(self):
        if self.s['terminated']:return []
        s=self.s;actions=[]
        if not s['bundle_taken']:
            actions += [{'type':'choose_bundle','bundle_id':i} for i in range(3)]
        for color,count in s['pending_tokens'].items():
            if count:
                actions += [{'type':'place_token','color':color,'cell_id':i}
                            for i in range(23) if self.can_place_token(color,i)]
        if not s['card_taken'] and len(s['active_cards'])<4:
            actions += [{'type':'take_card','market_slot':i} for i,card in enumerate(s['animal_market']) if card is not None]
        board=s['board']
        bits=sum(1 << (15*i+b['stack_id']) for i,b in enumerate(board))
        for card in s['active_cards']:
            for anchor,patterns in enumerate(FORBIDDEN[card['card_id']]):
                if board[anchor]['animal'] is not None:continue
                for orientation,forbidden in patterns:
                    if not bits & forbidden:
                        actions.append({'type':'place_animal','instance_id':card['instance_id'],
                                        'anchor_cell_id':anchor,'orientation':orientation})
                        break
        if s['bundle_taken'] and not any(s['pending_tokens'].values()):
            actions.append({'type':'end_turn'})
            if not s['card_taken']:
                actions += [{'type':'end_turn','discard_card_slot':i}
                            for i,card in enumerate(s['animal_market']) if card is not None]
        return actions
