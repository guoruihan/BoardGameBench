import json
import random
from copy import deepcopy
import pytest
from boardbench.contracts import InvalidAction
from boardbench.families.games.jaipur import Jaipur, GOODS, DATA


def game(seed=0):
    g = Jaipur()
    g.reset(seed)
    return g


def hand(g, values, herd=0):
    """Create a test hand while preserving the card multiset."""
    p = g.current_player
    pool = g.state['deck'] + g.state['discard'] + g.state['market']
    for i, n in enumerate(g.state['hands'][p]):
        pool.extend([GOODS[i]] * n)
    pool.extend(['camel'] * g.state['herds'][p])
    for i, n in enumerate(values):
        for _ in range(n):
            pool.remove(GOODS[i])
    for _ in range(herd):
        pool.remove('camel')
    g.state['hands'][p] = values
    g.state['herds'][p] = herd
    g.state['market'] = pool[-5:]
    g.state['deck'] = pool[:-5]
    g.state['discard'] = []


def test_component_setup_and_information():
    g = game()
    g.validate()
    assert g.state['market'][:3] == ['camel'] * 3
    obs = g.observe(0)
    assert 'deck' not in obs and 'herd' not in obs['opponent']
    before = g.observe(0)
    g.state['hands'][1].reverse()  # Equal-size hidden hand identity doesn't leak.
    g.state['deck'].reverse()
    g.state['herds'][1] += 2
    g.state['bonus_tokens']['3'].reverse()
    assert g.observe(0) == before


def test_cap_exchange_and_sale_sizes():
    g = game(4)
    hand(g, [0, 0, 0, 4, 3, 0], 2)
    assert not any(a['type'] == 'take' for a in g.legal_actions())
    for a in g.legal_actions():
        if a['type'] == 'exchange':
            assert sum(a['take']) >= 2
            assert not any(t and x for t, x in zip(a['take'], a['give']))
            assert a['give'][6] == 0
    hand(g, [0, 0, 0, 0, 0, 6])
    g.step(dict(type='sell', good='leather', count=6))
    assert len(g.state['bonus_won'][0]) == 1
    assert g.state['bonus_won'][0][0]['kind'] == '5_or_more'
    g.validate()


def test_sale_gets_bonus_despite_goods_shortage():
    g = game(5)
    hand(g, [3, 0, 0, 0, 0, 0])
    pile = g.state['goods_tokens']['diamond']
    g.state['goods_won'][1] += [dict(good='diamond', value=v) for v in pile[:-1]]
    g.state['goods_tokens']['diamond'] = pile[-1:]
    g.step(dict(type='sell', good='diamond', count=3))
    assert len(g.state['goods_won'][0]) == 1
    assert g.state['bonus_won'][0][0]['kind'] == '3'
    g.validate()


def test_deck_exhaustion_not_itself_end():
    g = game()
    g.state['market'] = ['leather'] * 5
    g.state['hands'][0] = [0] * 6
    g.state['deck'] = ['cloth']
    g.step(dict(type='take', good='leather'))
    assert g.state['round_number'] == 1 and len(g.state['deck']) == 0
    # Avoid a fully tied empty-score round.
    g.state['herds'] = [2, 0]
    g.state['hands'][1] = [0] * 6
    g.step(dict(type='take', good='leather'))
    assert g.state['round_number'] == 2 and g.state['seals'] == [1, 0]
    assert g.current_player == 1


def test_round_ties_and_match_end():
    g = game()
    g.state['herds'] = [1, 1]
    g._end_round('test')
    assert g.state['camel_award'] == [0, 0]
    assert g.outcome()['unresolved']
    assert not g.outcome()['terminated'] and not g.legal_actions()
    g = game()
    g.state['herds'] = [2, 0]
    g.state['seals'] = [1, 0]
    g._end_round('test')
    assert g.outcome()['winners'] == [0] and g.scores() == [2, 0]
    g = game()
    g.state['herds'] = [1, 1]
    g.state['goods_won'] = [[dict(good='silver', value=5)], [dict(good='cloth', value=3)]]
    g.state['bonus_won'][1] = [dict(kind='3', value=2)]
    g._end_round('test')
    assert g.state['seals'] == [0, 1] and g.current_player == 0


@pytest.mark.parametrize('points,goods,bonus,herds,seals,reason', [
    (74, [
        {'gold': [6, 6, 5], 'silver': [5] * 5, 'spice': [5, 3, 3, 2], 'diamond': [5, 5]},
        {'leather': [4, 3, 2, 1, 1], 'cloth': [5, 3, 3, 2], 'diamond': [7, 7, 5], 'gold': [5, 5]},
    ], [[('3', 2), ('4', 4), ('3', 3)], [('5_or_more', 8), ('4', 5), ('3', 3)]],
     [3, 6], [1, 1], 'three_goods_depleted'),
    (75, [
        {'silver': [5] * 3, 'leather': [4, 3, 2], 'gold': [6, 6, 5, 5],
         'cloth': [2, 2, 1, 1], 'diamond': [5, 5]},
        {'spice': [5, 3, 3, 2], 'diamond': [7, 7], 'cloth': [5, 3, 3],
         'leather': [1] * 5, 'silver': [5, 5]},
    ], [[('3', 2), ('4', 6), ('4', 5)], [('4', 4), ('3', 3), ('5_or_more', 10)]],
     [3, 7], [0, 0], 'market_cannot_refill'),
])
def test_exact_scoring_tie_preserves_unresolved_state(points, goods, bonus, herds, seals, reason):
    """Token allocations from the two league cases; no invented fourth tiebreak.

    The fixture reconstructs scoring inventory, not the original action trajectory.
    The camel holder has five extra rupees, but no extra bonus/goods token.
    """
    g = game()
    s = g.state
    s['seals'], s['herds'] = list(seals), list(herds)
    s['round_number'] = sum(seals) + 1
    s['hands'] = [[0] * 6 for _ in range(2)]
    cards = [kind for kind, count in DATA['cards'].items()
             for _ in range(count - (sum(herds) if kind == 'camel' else 0))]
    market_size = 4 if reason == 'market_cannot_refill' else 5
    s['market'], s['discard'], s['deck'] = cards[:market_size], cards[market_size:], []
    for p in (0, 1):
        s['goods_won'][p] = [dict(good=kind, value=v)
                             for kind, values in goods[p].items() for v in values]
        for token in s['goods_won'][p]:
            s['goods_tokens'][token['good']].remove(token['value'])
        s['bonus_won'][p] = [dict(kind=kind, value=v) for kind, v in bonus[p]]
        for token in s['bonus_won'][p]:
            s['bonus_tokens'][token['kind']].remove(token['value'])
    before_rng, before_round = g.snapshot()['rng'], s['round_number']
    g._end_round(reason)
    g.validate()
    expected = [points, 3, 14 if points == 74 else 16]
    assert s['rounds'][-1]['criteria'] == [expected, expected]
    assert s['camel_award'] == [0, 5]
    assert s['seals'] == seals and s['round_number'] == before_round
    assert g.snapshot()['rng'] == before_rng  # No implicit restart/shuffle.
    assert g.outcome() == dict(terminated=False, scores=seals, winners=[],
                               reason='unresolved_round_tie', unresolved=True)
    assert not g.legal_actions()
    clone = Jaipur()
    clone.restore(json.loads(json.dumps(g.snapshot())))
    assert clone.outcome() == g.outcome() and not clone.legal_actions()
    before = clone.snapshot()
    with pytest.raises(InvalidAction):
        clone.step(dict(type='camels'))
    assert clone.snapshot() == before


def test_atomic_rejection_json_replay():
    g = game(2)
    before = g.snapshot()
    with pytest.raises(InvalidAction):
        g.step(dict(type='sell', good='gold', count=True))
    assert g.snapshot() == before
    clone = Jaipur()
    clone.restore(json.loads(json.dumps(before)))
    for _ in range(15):
        a = max(g.legal_actions(), key=lambda a: g.heuristic(g.observe(), a))
        assert g.step(a) == clone.step(a)
        assert g.snapshot() == clone.snapshot()
        g.validate()


@pytest.mark.parametrize('seed', range(8))
def test_complete_seeded_matches(seed):
    g = game(seed)
    rng = random.Random(seed)
    for n in range(600):
        actions = g.legal_actions()
        if not actions:
            break
        obs = g.observe()
        for a in actions[:5]:
            assert len(g.features(obs, a)) == 24
        a = rng.choice(actions) if seed % 2 else max(actions, key=lambda a: g.heuristic(obs, a))
        g.step(a)
        g.validate()
        if g.state['terminated']:
            break
    assert g.state['terminated'] or g.state['blocked_reason'], (seed, n)
