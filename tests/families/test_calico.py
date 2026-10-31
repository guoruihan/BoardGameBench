import itertools
import json
import random

import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.games.calico import (
    Calico, DATA, ADJ, INDEX, GOAL_INDICES, GOALS, BORDERS,
    component, goal_score, completion_probability,
)


def game(seed=0, first=0):
    g=Calico();g.reset(seed,first)
    return g


def start(g):
    while g.state['phase']=='goals':g.step(g.legal_actions()[0])


def test_authentic_board_boundaries_and_full_standard_setup_choices():
    assert len(INDEX)==47
    assert len(DATA['playable_cells'])==22
    assert len(DATA['goal_slots'])==3
    assert all(len(b['printed_cells'])==22 for b in DATA['boards'])
    assert all(len(ADJ[g])==6 for g in GOAL_INDICES)
    assert BORDERS[0][21]==4*6+4  # Primary image green flowers, not source typo leaves.
    assert BORDERS[0]!=BORDERS[1]
    g=game(first=1)
    assert g.current_player==1 and len(g.legal_actions())==24
    assert g.state['hands']==[[],[]] and g.state['market']==[]
    a=g.legal_actions()[-1];g.step(a)
    assert g.current_player==0 and g.state['goals'][1]==a['goals']
    g.step(g.legal_actions()[0])
    assert g.current_player==1 and g.state['phase']=='place'
    assert len(g.legal_actions())==44  # Either hand, every empty playable cell.
    assert len(g.state['deck'])==101
    g.validate()


def test_goal_multisets_and_double_is_not_additive():
    board=[-1]*47;center=GOAL_INDICES[0]
    colors=[0,0,1,1,2,2];patterns=[0,0,0,0,1,2]
    for n,c,p in zip(ADJ[center],colors,patterns):board[n]=6*c+p
    assert goal_score(board,center,'aa_bb_cc')==7
    for n,c,p in zip(ADJ[center],colors,[3,4,5,3,4,5]):board[n]=6*c+p
    assert goal_score(board,center,'aa_bb_cc')==11
    board[ADJ[center][0]]=-1
    assert goal_score(board,center,'aa_bb_cc')==0
    for goal,spec in GOALS.items():
        values=[i for i,count in enumerate(spec['counts']) for _ in range(count)]
        for n,c in zip(ADJ[center],values):board[n]=6*c+c
        assert goal_score(board,center,goal)==spec['double']


def test_goal_probability_matches_enumerated_small_completion():
    for goal in GOALS:
        counts=(2,1,1)
        success=0
        for remaining in itertools.product(range(6),repeat=2):
            values=[0,0,1,2]+list(remaining)
            got=sorted(values.count(i) for i in range(6) if values.count(i))
            success+=got==sorted(GOALS[goal]['counts'])
        assert completion_probability(counts,goal)==pytest.approx(success/36)


def test_merging_previously_awarded_groups_keeps_two_rewards():
    g=game();g.state['pattern_cats']=[0,0,1,1,2,2]
    board=[-2]*47
    g._public_board=lambda p:board  # Synthetic golden arrangement, no chance transitions.
    a=[(0,-2),(-1,-2),(-1,-1)]
    b=[(-1,1),(-1,2),(-2,2)]
    for group in (a,b):
        for coord in group:
            cell=INDEX[coord];board[cell]=0;g._award(0,cell)
    assert len(g.state['buttons'][0])==2
    assert len(g.state['cats'][0])==2
    cell=INDEX[(-1,0)];board[cell]=0;g._award(0,cell)
    assert len(component(board,cell,'color'))==7
    assert len(g.state['buttons'][0])==2
    assert len(g.state['cats'][0])==2


def test_cat_patterns_do_not_mix_and_border_counts_for_both_rewards():
    g=game();g.state['pattern_cats']=[0,0,1,1,2,2]
    board=[-2]*47;g._public_board=lambda p:board
    cells=[INDEX[c] for c in [(0,-2),(-1,-2),(-1,-1)]]
    for cell,code in zip(cells,[0,0,1]):
        board[cell]=code;g._award(0,cell)
    assert len(g.state['buttons'][0])==1
    assert len(g.state['cats'][0])==0  # Millie likes both patterns but cannot mix them.
    g=game();g.state['pattern_cats']=[0,0,1,1,2,2]
    board=[-2]*47;g._public_board=lambda p:board
    cell=INDEX[(-1,-2)]
    borders=[n for n in ADJ[cell] if n>=25]
    assert len(borders)>=2
    for n in borders[:2]:board[n]=0
    board[cell]=0;g._award(0,cell)
    assert len(g.state['buttons'][0])==len(g.state['cats'][0])==1


def test_rainbow_and_held_tiles_do_not_score():
    g=game();start(g)
    g.state['buttons'][0]=[dict(color=c,cell=c) for c in range(6)]
    g.state['rainbow'][0]=True
    assert g.score_breakdown()[0]['buttons']==21
    assert g.score_breakdown()[0]['cats']==0
    assert g.score_breakdown()[0]['goals']==[0,0,0]


def test_hidden_hand_future_deck_and_public_market_history():
    g=game();start(g)
    obs=g.observe(0)
    assert 'hands' not in obs and 'deck' not in obs and 'rng' not in obs
    g.state['hands'][1],g.state['deck'][:2]=g.state['deck'][:2],g.state['hands'][1]
    g.state['deck'].reverse()
    assert g.observe(0)==obs
    g.validate()
    place=g.legal_actions()[0];g.step(place)
    assert len(g.legal_actions())==3 and g.state['phase']=='draft'
    tile=g.observe()['market'][1]
    g.step(dict(type='draft',slot=1))
    assert g.state['public_history'][-1]==dict(type='draft',player=0,slot=1,tile=tile)


@pytest.mark.parametrize('seed',range(8))
def test_complete_game_replay_invariants_and_final_draft(seed):
    g=game(seed,seed%2);h=Calico();r=random.Random(seed)
    decisions=0
    while g.current_player is not None:
        g.validate();obs=g.observe();actions=g.legal_actions();assert actions
        for a in actions[::max(1,len(actions)//5)]:
            assert len(g.features(obs,a))==24
            assert isinstance(g.heuristic(obs,a),float)
        a=max(actions,key=lambda a:g.heuristic(obs,a)) if g.current_player==0 else r.choice(actions)
        if decisions%9==0:
            h.restore(json.loads(json.dumps(g.snapshot())))
            assert h.step(a)==g.step(a)
        else:g.step(a)
        decisions+=1
        assert decisions<=90
        if g.state['turns']==[22,22] and not g.state['terminated']:
            assert g.state['phase']=='draft'  # Final market draft is still required.
    assert decisions==90 and g.state['turns']==[22,22]
    assert len(g.state['deck'])==57
    assert [len(v) for v in g.state['hands']]==[2,2]
    assert g.outcome()['reason']=='both_quilts_filled'
    g.validate()


def test_illegal_action_and_corrupt_snapshot_rollback():
    g=game();start(g);before=g.snapshot()
    for action in [dict(type='place',hand=0,cell=22),dict(type='place',hand=False,cell=0),dict(type='draft',slot=0)]:
        with pytest.raises(InvalidAction):g.step(action)
        assert g.snapshot()==before
    bad=g.snapshot();bad['state']['deck'].pop()
    with pytest.raises(ValueError):g.restore(bad)
    assert g.snapshot()==before


@pytest.mark.parametrize('cats,buttons,rainbow,winners', [
    ([1,2],[3,0],[False,False],[1]),
    ([2,2],[3,2],[False,False],[0]),
    ([2,2],[2,2],[False,True],[1]),
    ([2,2],[2,2],[False,False],[0,1]),
])
def test_official_tiebreak_cats_then_buttons_then_shared(cats,buttons,rainbow,winners):
    g=game();start(g)
    # Golden terminal equality isolates the official tie hierarchy.
    g.state['turns']=[22,22];g.state['phase']='draft';g.state['hands'][0].pop()
    g.scores=lambda:[50,50]
    g.state['cats']=[[dict(cat=0)]*n for n in cats]
    g.state['buttons']=[[dict(color=0)]*n for n in buttons]
    g.state['rainbow']=rainbow
    result=g.step(dict(type='draft',slot=0))
    assert result['info']['winners']==winners
