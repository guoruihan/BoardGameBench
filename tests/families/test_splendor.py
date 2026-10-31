import json
import random
from collections import Counter
from copy import deepcopy
import pytest
from boardbench.contracts import InvalidAction
from boardbench.families.games.splendor import Splendor, CARDS, NOBLES, payments


def game(seed=0, first=0):
    g = Splendor(); g.reset(seed, first); return g


def tokens(g, p, values):
    for i, (old, new) in enumerate(zip(g.state['tokens'][p], values)):
        g.state['bank'][i] += old - new
    g.state['tokens'][p] = list(values)


def move_card(g, c, destination, p):
    for key in ('decks', 'market', 'bought', 'reserved'):
        for row in g.state[key]:
            if c in row:
                row.remove(c)
    g.state[destination][p].append(c)


def test_catalog_and_setup():
    assert Counter(c['level'] for c in CARDS.values()) == {1: 40, 2: 30, 3: 20}
    assert len(NOBLES) == 10 and all(n['points'] == 3 for n in NOBLES.values())
    # Photographed cost: not the conflicting legacy CSV's red-three cost.
    assert CARDS['white-L1-01']['cost'] == [0, 3, 0, 0, 0]
    g = game(); g.validate()
    assert g.state['bank'] == [4]*5+[5]
    assert len(g.state['nobles']) == 3
    assert [len(r) for r in g.state['market']] == [4]*3


def test_take_threshold_shortage_and_gold_exclusion():
    g = game()
    tokens(g, 1, [0, 0, 0, 1, 0, 0])
    assert {'type':'take_same','color':3} not in g.legal_actions()
    assert {'type':'take_same','color':0} in g.legal_actions()
    assert {'type':'take_same','color':5} not in g.legal_actions()
    g.state['bank'] = [0, 2, 0, 3, 0, 5]
    assert [a for a in g.legal_actions() if a['type']=='take_distinct'] == [dict(type='take_distinct',colors=[1,3])]


def test_ten_tokens_may_take_then_choose_return():
    g = game(); tokens(g, 0, [2, 2, 2, 2, 2, 0])
    g.step(dict(type='take_distinct',colors=[0,1,2]))
    assert g.current_player == 0 and g.state['phase'] == 'return'
    assert sum(g.state['tokens'][0]) == 13
    assert dict(type='return',tokens=[1,1,1,0,0,0]) in g.legal_actions()
    g.step(dict(type='return',tokens=[0,0,0,2,1,0]))
    assert sum(g.state['tokens'][0]) == 10 and g.current_player == 1
    g.validate()


def test_gold_alternatives_and_existing_bonuses():
    assert list(payments([0,0,0,1,0], [0]*5, [0,0,0,1,0,1])) == [[0,0,0,0,0,1], [0,0,0,1,0,0]]
    assert list(payments([0,2,1,0,0], [0,2,0,0,0], [0,0,1,0,0,0])) == [[0,0,1,0,0,0]]
    g=game(); tokens(g,0,[0,3,0,0,0,1])
    move_card(g,'white-L1-01','reserved',0)
    actions=[a for a in g.legal_actions() if a['type']=='buy' and a['card']=='white-L1-01']
    assert {tuple(a['pay']) for a in actions} == {(0,3,0,0,0,0),(0,2,0,0,0,1)}
    g.step(actions[0]);g.validate()
    assert g._bonuses(0) == [1,0,0,0,0]


def test_reservation_limit_no_gold_and_blind_information():
    g=game(); tokens(g,1,[0,0,0,0,0,5])
    a=dict(type='reserve',source='deck',level=1)
    g.step(a)
    assert len(g.state['reserved'][0]) == 1 and g.state['tokens'][0][5] == 0
    obs=g.observe(1)
    assert g.state['reserved'][0][0] not in str(obs['history'])
    # Swap a blind reserved identity with a same-tier hidden deck card.
    old=g.state['reserved'][0][0]
    g.state['reserved'][0][0]=g.state['decks'][0][-1]
    g.state['decks'][0][-1]=old
    assert g.observe(1) == obs
    g.state['current_player']=0
    for c in list(g.state['decks'][0])[:2]: move_card(g,c,'reserved',0)
    assert not any(a['type']=='reserve' for a in g.legal_actions())
    g.validate()


def test_noble_choice_one_per_turn_and_permanent_only():
    g=game()
    # Every four/four noble is eligible with four permanent bonuses in each color.
    for color in range(5):
        for c in [c for c,v in CARDS.items() if v['level']==1 and v['bonus']==color][:4]:
            move_card(g,c,'bought',0)
    g._after_main()
    assert g.state['phase']=='noble' and len(g.legal_actions())==3
    g.step(g.legal_actions()[0])
    assert len(g.state['acquired_nobles'][0]) == 1 and len(g.state['nobles'])==2
    assert g.current_player == 1
    g.validate()
    g=game(); tokens(g,0,[4,4,0,0,0,0])
    assert not g._eligible_nobles(0)


def test_equal_rounds_and_fewer_bought_tiebreak():
    g=game(first=1)
    g.scores=lambda:[17,15]
    g.state['current_player']=1
    g._end_turn()
    assert not g.state['terminated'] and g.current_player==0
    g._end_turn()
    assert g.state['terminated'] and g.state['winners']==[0]
    g=game()
    g.scores=lambda:[16,16]
    g.state['bought']=[['a']*9,['b']*10]
    g.state['current_player']=1
    g.state['turns']=[1,0]
    g._end_turn()
    assert g.state['winners']==[0]


def test_atomic_invalid_action_and_json_replay():
    g=game();before=g.snapshot()
    with pytest.raises(InvalidAction):g.step(dict(type='take_same',color=True))
    assert g.snapshot()==before
    clone=Splendor();clone.restore(json.loads(json.dumps(before)))
    for _ in range(20):
        obs=g.observe();a=max(g.legal_actions(),key=lambda a:g.heuristic(obs,a))
        assert g.step(a)==clone.step(a)
        assert g.snapshot()==clone.snapshot()
        g.validate()


def double_blocked_game():
    """Reach a genuine mutual block using only legal actions, without purchases."""
    g = game(1)
    for _ in range(6):
        g.step(dict(type='reserve', source='deck', level=3))
    for colors in ([0, 1, 2], [0, 1, 2], [0, 3, 4], [0, 3, 4], [1, 2, 3]):
        g.step(dict(type='take_distinct', colors=colors))
    g.step(dict(type='return', tokens=[0, 0, 0, 0, 0, 2]))
    g.step(dict(type='take_distinct', colors=[1, 2, 4]))
    g.step(dict(type='return', tokens=[0, 0, 0, 0, 0, 1]))
    g.step(dict(type='take_distinct', colors=[3, 4]))
    g.step(dict(type='return', tokens=[0, 0, 0, 1, 0, 1]))
    g.step(dict(type='take_distinct', colors=[3]))
    g.step(dict(type='return', tokens=[0, 0, 0, 0, 0, 1]))
    g.validate()
    assert g.state['tokens'] == [[2] * 5 + [0]] * 2
    return g


def test_pass_only_when_all_normal_actions_impossible_and_atomic():
    for g in (game(), double_blocked_game()):
        if not any(g.state['bank'][:5]):
            if g.state['reserved'][0][0] == 'red-L3-01':
                # A free reserve slot is a normal action even without ordinary gems.
                move_card(g, 'red-L3-01', 'decks', 2)
        before = g.snapshot()
        assert dict(type='pass') not in g.legal_actions()
        with pytest.raises(InvalidAction):
            g.step(dict(type='pass'))
        assert g.snapshot() == before
    g = double_blocked_game()
    move_card(g, 'red-L3-01', 'decks', 2)
    move_card(g, 'white-L1-03', 'reserved', 0)
    assert any(a['type'] == 'buy' for a in g.legal_actions())
    before = g.snapshot()
    with pytest.raises(InvalidAction):
        g.step(dict(type='pass'))
    assert g.snapshot() == before


def test_forced_pass_reveals_hand_and_replays_without_rng_or_score_change():
    g = double_blocked_game()
    before_obs = g.observe(1)
    old = g.state['reserved'][0][0]
    g.state['reserved'][0][0] = g.state['decks'][2][-1]
    g.state['decks'][2][-1] = old
    assert g.observe(1) == before_obs  # Blind hand is private before passing.
    before = g.snapshot()
    assert g.legal_actions() == [dict(type='pass')]
    assert len(g.features(g.observe(), dict(type='pass'))) == 24
    result = g.step(dict(type='pass'))
    assert g.observe(1)['revealed_reserved'][0] == g.state['reserved'][0]
    assert g.observe(1)['history'][-1]['revealed_reserved'] == g.state['reserved'][0]
    assert g.state['revealed_reserved'][1] == []
    assert g.current_player == 1 and g.state['turns'] == [8, 7]
    assert g.snapshot()['rng'] == before['rng']
    assert g.state['tokens'] == before['state']['tokens'] and g.scores() == [0, 0]
    assert not result['terminated'] and result['rewards'] == [0., 0.]
    clone = Splendor(); clone.restore(json.loads(json.dumps(g.snapshot())))
    assert clone.observe(0) == g.observe(0) and clone.observe(1) == g.observe(1)
    assert clone.step(dict(type='pass')) == g.step(dict(type='pass'))
    assert clone.snapshot() == g.snapshot()
    with pytest.raises(ValueError):
        clone.restore(dict(g.snapshot(), rules_version='1'))


def test_revealed_reserved_persists_but_new_blind_reservation_is_private():
    g = double_blocked_game()
    g.step(dict(type='pass'))
    g.step(dict(type='pass'))
    revealed = list(g.state['reserved'][0])
    # A controlled fixture lets seat 0 buy its seven-green reserved card.
    for c in [c for c in g.state['decks'][0]
              if CARDS[c]['bonus'] == 2 and CARDS[c]['points'] == 0][:5]:
        move_card(g, c, 'bought', 0)
    g.step(dict(type='buy', source='reserved', card='red-L3-01', pay=[0, 0, 2, 0, 0, 0]))
    g.step(dict(type='take_distinct', colors=[2]))
    g.step(dict(type='return', tokens=[0, 0, 1, 0, 0, 0]))
    hidden = g.state['decks'][2][-1]
    g.step(dict(type='reserve', source='deck', level=3))
    assert g.observe(1)['revealed_reserved'][0] == revealed[1:]
    assert hidden in g.state['reserved'][0] and hidden not in str(g.observe(1))
    # Earlier revealed identities remain in the public event history after buying.
    assert g.observe(1)['history'][18]['revealed_reserved'] == revealed
    g.validate()


def test_mutual_forced_pass_is_not_an_invented_terminal_or_draw():
    g = double_blocked_game()
    before = g.snapshot()
    for _ in range(6):
        assert g.legal_actions() == [dict(type='pass')]
        g.step(dict(type='pass')); g.validate()
        assert not g.outcome()['terminated'] and not g.outcome()['winners']
    assert g.scores() == [0, 0] and g.snapshot()['rng'] == before['rng']
    # The existing runner owns the limit and must keep this in the denominator.
    from boardbench.families.play import run_match
    from unittest.mock import patch
    class BlockedSplendor(Splendor):
        def reset(self, seed=0, first_player=0):
            self.restore(double_blocked_game().snapshot())
            return self.observe()
    with patch('boardbench.families.play.game_class', return_value=BlockedSplendor):
        result = run_match('splendor', [{'kind': 'heuristic'}] * 2, max_decisions=4)
    assert result['status'] == 'truncated' and result['rewards'] is None
    assert result['decisions'] == 4 and not result['outcome']['terminated']


@pytest.mark.parametrize('seed',range(8))
def test_complete_seeded_matches(seed):
    g=game(seed,seed%2)
    rng=random.Random(seed)
    for step in range(500):
        obs=g.observe();actions=g.legal_actions()
        assert actions
        # Random opponent tests substantial non-greedy legal paths; random/random
        # may legitimately cycle, so a progressing baseline anchors termination.
        a=rng.choice(actions) if g.current_player==seed%2 and seed%2 else max(actions,key=lambda a:g.heuristic(obs,a))
        assert len(g.features(obs,a))==24
        g.step(a);g.validate()
        if g.state['terminated']:break
    assert g.state['terminated'],(seed,step,g.scores())
