"""Inventory-consistent controlled supply and an entirely legal animal-card trace."""
from collections import Counter
from copy import deepcopy

from boardbench.environments.harmonies import Harmonies, INVENTORY, COLORS, NEIGHBORS, STACKS


def fixture():
    env = Harmonies()
    chosen = [('green','blue','blue'), ('blue','blue','yellow'),
              ('yellow','red','brown'), ('brown','green','yellow'), ('blue','gray','gray')]
    remaining = Counter(INVENTORY)
    remaining.subtract(c for group in chosen for c in group)
    padding = [c for c in COLORS for _ in range(remaining[c])]
    order = []
    for group in chosen:
        order.extend(group)
        order.extend(padding[:6])
        padding = padding[6:]
    env._bag = order+padding
    env._deck = [5,1,2,3,4,6,7]+list(range(8,33))
    assert sorted(env._deck)==list(range(1,33))
    env.s['token_market'] = [None]*3
    env.s['animal_market'] = [None]*3
    env._refill_tokens()
    env._refill_cards()
    initial = env.save_state()
    center = next(i for i,n in enumerate(NEIGHBORS) if len(n)==6)
    anchors = NEIGHBORS[center][:5]
    protected = {center,*anchors}
    actions, milestones = [], {}
    def act(action):
        assert action in env.legal_actions(), action
        env.step(action)
        actions.append(deepcopy(action))
        count = Counter(env.s['remaining_tokens'])
        count.update(env.s['pending_tokens'])
        count.update(env.s['discarded_tokens'])
        for group in env.s['token_market']:
            count.update(group or [])
        for cell in env.s['board']:
            count.update(STACKS[cell['stack_id']])
        assert count==INVENTORY
    def animal(anchor):
        act(next(a for a in env.legal_actions() if a['type']=='place_animal'
                 and a['instance_id']==5 and a['anchor_cell_id']==anchor))
    def filler(color):
        act(next(a for a in env.legal_actions() if a['type']=='place_token'
                 and a['color']==color and a['cell_id'] not in protected))
    for turn, group in enumerate(chosen, 1):
        act(dict(type='choose_bundle',bundle_id=0))
        if turn==1:
            act(dict(type='place_token',color='green',cell_id=center))
            act(dict(type='take_card',market_slot=0))
            milestones['interleaved_take'] = len(actions)
            act(dict(type='place_token',color='blue',cell_id=anchors[0]))
            animal(anchors[0])
            milestones['partial_save'] = len(actions)
            act(dict(type='place_token',color='blue',cell_id=anchors[1]))
            animal(anchors[1])
        elif turn==2:
            act(dict(type='place_token',color='blue',cell_id=anchors[2]))
            animal(anchors[2])
            act(dict(type='take_card',market_slot=0))
            act(dict(type='place_token',color='blue',cell_id=anchors[3]))
            animal(anchors[3])
            filler('yellow')
        elif turn<5:
            for color in group:
                filler(color)
            act(dict(type='take_card',market_slot=0))
        else:
            assert len(env.s['active_cards'])==4 and not env.s['card_taken']
            act(dict(type='place_token',color='blue',cell_id=anchors[4]))
            animal(anchors[4])
            milestones['slot_released'] = len(actions)
            assert len(env.s['active_cards'])==3 and len(env.s['completed_cards'])==1
            act(dict(type='take_card',market_slot=0))
            milestones['completed_save'] = len(actions)
        if turn<5:
            act(dict(type='end_turn'))
    return {'snapshot':initial, 'actions':actions, 'milestones':milestones,
            'final_observation':env.observe(), 'description':'Controlled full supply; all actions legal; inventory checked at every step'}
