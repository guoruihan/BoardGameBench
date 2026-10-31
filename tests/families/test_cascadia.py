import itertools
import json
import random
from collections import Counter

import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.games.cascadia import (
    Cascadia, DATA, TILES, HABITATS, ANIMALS, DIRECTIONS, edges, neighbors,
    components, geometry, habitat_scores, animal_scores, elk_score, fringe,
)


def game(seed=0, first=0):
    g = Cascadia(); g.reset(seed, first)
    return g


def choose(g):
    if g.state['phase'] == 'overpopulation': g.step(dict(type='keep'))


def set_market(g, animals):
    for a in g.state['market_animals']: g.state['bag'][a] += 1
    for a in animals: g.state['bag'][a] -= 1
    g.state['market_animals'] = list(animals)


def earn_nature(g, p=0):
    c = next(c for c in g.state['boards'][p] if TILES[c['tile']]['keystone'])
    c['animal'] = TILES[c['tile']]['supported_animals'][0]
    g.state['bag'][c['animal']] -= 1
    g.state['nature'][p] += 1


def scripted_draw(g, script, recorded=None):
    todo = iter(script)
    def draw():
        if recorded is not None: recorded.append(dict(g.state['bag']))
        a = next(todo); assert g.state['bag'][a] > 0
        g.state['bag'][a] -= 1
        return a
    return draw


def test_authentic_complete_component_counts_and_independent_census():
    normal = [TILES[i] for i in DATA['normal_tile_ids']]
    assert len(normal) == 85 and len(TILES) == 100 and len(DATA['starters']) == 5
    assert sum(t['keystone'] for t in normal) == 25
    assert Counter(tuple(sorted(t['habitats'])) for t in normal) == Counter({
        **{(h,): 5 for h in HABITATS},
        **{tuple(sorted(h)): 6 for h in itertools.combinations(HABITATS, 2)}})
    for a in ANIMALS:
        assert sum(a in t['supported_animals'] for t in normal) == 32
        assert [sum(a in t['supported_animals'] and h in t['habitats'] for t in normal)
                for h in HABITATS] == DATA['animal_habitat_audit'][a]
    g = game(first=1); g.validate()
    assert g.current_player == 1 and len(g.state['deck']) == 39
    assert all(len(b) == 3 for b in g.state['boards'])
    assert g.state['nature'] == [0, 0]


def test_edges_clockwise_and_matching_is_scoring_not_placement():
    tile = next(t['id'] for t in TILES.values() if t['habitats'] == ['forest', 'wetland'])
    assert edges(tile, 0) == ('wetland',)*3 + ('forest',)*3
    assert edges(tile, 1) == ('forest', 'wetland', 'wetland', 'wetland', 'forest', 'forest')
    assert habitat_scores(((0, 0, tile, 0), (1, 0, tile, 0))) == (0, 1, 0, 1, 0)
    assert habitat_scores(((0, 0, tile, 0), (1, 0, tile, 3))) == (0, 1, 0, 2, 0)
    g = game(); choose(g); g.step(g.legal_actions()[0])
    actions = [a for a in g.legal_actions() if a['type'] == 'place_tile']
    assert {(a['q'], a['r']) for a in actions} == set(fringe(g.state['boards'][0]))
    before = g.snapshot()
    with pytest.raises(InvalidAction): g.step(dict(type='place_tile', q=99, r=99, rotation=0))
    assert g.snapshot() == before


def test_bear_exact_components_and_four_pair_cap():
    assert animal_scores(((0,0,'bear'),(1,0,'bear'),(2,0,'bear')))[0] == 0
    for pairs, score in [(1,4),(2,11),(3,19),(4,27),(5,27)]:
        placed = tuple((4*i+j,0,'bear') for i in range(pairs) for j in (0,1))
        assert animal_scores(placed)[0] == score


def brute_elk(coords):
    """Independent exhaustive set partition, checking every subset's collinearity."""
    points = sorted(coords)
    if not points: return 0
    first, rest = points[0], points[1:]
    best = 0
    for size in range(1, min(4, len(points))+1):
        for tail in itertools.combinations(rest, size-1):
            group = (first,) + tail
            if size > 1:
                collinear = any(len({c[axis] for c in group}) == 1 for axis in (0,1))
                collinear |= len({q+r for q,r in group}) == 1
                if not collinear or len(components(group)) != 1: continue
            best = max(best, (0,2,5,9,13)[size]+brute_elk(set(points)-set(group)))
    return best


def test_elk_optimal_partition_matches_independent_exhaustive_oracle():
    assert elk_score(tuple((i,0) for i in range(8))) == 26
    assert elk_score(tuple((i,0) for i in range(5))) == 15
    grid = [(q,r) for q in range(3) for r in range(3)]
    rng = random.Random(81)
    for _ in range(50):
        coords = tuple(sorted(rng.sample(grid, rng.randrange(1,9))))
        assert elk_score(coords) == brute_elk(coords)


def test_salmon_exclusion_once_triangle_valid_and_invalid_still_counts_for_fox():
    assert animal_scores(((0,0,'salmon'),(1,0,'salmon'),(0,1,'salmon')))[2] == 8
    star = ((0,0,'salmon'),(1,-1,'salmon'),(0,1,'salmon'),(-1,0,'salmon'))
    assert animal_scores(star)[2] == 6
    assert animal_scores(star+((1,0,'fox'),))[4] == 1
    # Two adjacent degree-three centers are both excluded based on original board.
    fish = ((0,0),(1,0),(-1,0),(0,-1),(2,0),(1,1))
    valid = {c for c in fish if sum(n in fish for n in neighbors(c)) <= 2}
    expected = sum((0,2,5,8,12,16,20,25)[min(len(c),7)] for c in components(valid))
    assert animal_scores(tuple((q,r,'salmon') for q,r in fish))[2] == expected


def test_hawk_isolation_fox_unique_species_and_printed_icons_do_not_score():
    assert animal_scores(((0,0,'hawk'),(1,0,'hawk'),(5,0,'hawk')))[3] == 2
    center = ((0,0,'fox'),)
    around = tuple((q,r,a) for (q,r),a in zip(neighbors((0,0)), ANIMALS+('bear',)))
    assert animal_scores(center+around)[4] >= 5
    assert animal_scores(center+((1,0,'bear'),(0,1,'bear')))[4] == 1
    assert animal_scores(()) == (0,0,0,0,0)


def test_designer_wipe_returns_each_batch_before_subsequent_replacement(monkeypatch):
    g = game(); set_market(g, ['bear']*4)
    recorded = []
    monkeypatch.setattr(g, '_draw_animal', scripted_draw(g, ['elk']*4+['hawk','salmon','fox','hawk'], recorded))
    g._overpopulation()
    assert recorded[0]['bear'] == 16 and recorded[4]['bear'] == 20
    assert [h['reason'] for h in g.state['public_history'][-2:]] == ['mandatory_four']*2
    assert g.state['phase'] == 'choose'
    g.validate()


def test_free_wipe_once_and_nature_does_not_reset_it(monkeypatch):
    g = game(); earn_nature(g); set_market(g, ['bear']*3+['elk']); g._overpopulation()
    assert g.state['phase'] == 'overpopulation'
    monkeypatch.setattr(g, '_draw_animal', scripted_draw(g, ['hawk','fox','salmon','bear','bear','bear']))
    g.step(dict(type='free_wipe'))
    assert g.state['free_wipe_used'] and g.state['phase'] == 'choose'
    g.step(dict(type='nature_wipe', slots=[0,1,2]))
    assert g.state['market_animals'] == ['bear']*3+['elk']
    assert g.state['phase'] == 'choose' and g.state['free_wipe_used']
    g.validate()


def test_nature_wipe_can_offer_unused_three_wipe(monkeypatch):
    g = game(); choose(g); earn_nature(g); set_market(g, ['bear','elk','hawk','fox'])
    monkeypatch.setattr(g, '_draw_animal', scripted_draw(g, ['bear','bear']))
    g.step(dict(type='nature_wipe', slots=[1,2]))
    assert g.state['phase'] == 'overpopulation' and not g.state['free_wipe_used']
    g.validate()


def test_nature_split_pair_only_refills_the_two_selected_slots():
    g = game(); choose(g); earn_nature(g)
    tiles = g.state['market_tiles'][:]; animals = g.state['market_animals'][:]
    g.step(dict(type='select', tile_slot=0, animal_slot=2, nature=True))
    assert g.state['market_tiles'] == [None]+tiles[1:]
    assert g.state['market_animals'] == animals[:2]+[None,animals[3]]
    g.step(dict(type='discard_animal'))
    assert g.state['phase'] == 'place' and g.state['current_player'] == 0
    g.step(next(a for a in g.legal_actions() if a['type'] == 'place_tile'))
    assert g.state['market_tiles'][1:] == tiles[1:]
    refill = next(h for h in reversed(g.state['public_history']) if h['type'] == 'refill')
    assert refill['tile_slot'] == 0 and refill['animal_slot'] == 2
    assert g.state['nature'][0] == 0 and g.state['nature_spent'][0] == 1
    g.validate()


def test_animal_before_tile_keystone_award_and_voluntary_discard():
    g = game(); choose(g)
    c = next(c for c in g.state['boards'][0] if TILES[c['tile']]['keystone'])
    a = TILES[c['tile']]['supported_animals'][0]
    set_market(g, [a]+[x for x in ANIMALS if x != a][:3])
    g.step(dict(type='select', tile_slot=0, animal_slot=0, nature=False))
    assert dict(type='discard_animal') in g.legal_actions()
    g.step(dict(type='place_animal', q=c['q'], r=c['r']))
    assert g.state['nature'][0] == 1 and g.state['phase'] == 'place'
    assert all(x['type'] == 'place_tile' for x in g.legal_actions())
    assert g.state['market_tiles'][0] is None
    g.step(g.legal_actions()[0]); g.validate()


def test_hidden_tile_order_removed_set_and_bag_order_do_not_leak():
    g = game(); obs = g.observe()
    g.state['deck'].reverse(); g.state['removed'].reverse()
    g.state['deck'][0], g.state['removed'][0] = g.state['removed'][0], g.state['deck'][0]
    assert g.observe() == obs
    assert not {'deck','removed','bag','rng'} & obs.keys()
    g.validate()


@pytest.mark.parametrize('seed',range(6))
def test_complete_games_conservation_json_replay_and_natural_end(seed):
    g = game(seed, seed%2); other = Cascadia(); rng = random.Random(seed+92)
    n = 0
    while g.current_player is not None:
        g.validate(); obs = g.observe(); actions = g.legal_actions(); assert actions
        assert g.legal_actions(1-g.current_player) == []
        for a in actions[::max(1,len(actions)//4)]:
            assert len(g.features(obs,a)) == 24
            assert isinstance(g.heuristic(obs,a),float)
        a = max(actions,key=lambda a:g.heuristic(obs,a)) if g.current_player == 0 else rng.choice(actions)
        if n%11 == 0:
            other.restore(json.loads(json.dumps(g.snapshot())))
            assert other.step(a) == g.step(a)
        else: g.step(a)
        n += 1; assert n < 300
    g.validate()
    assert g.state['turns'] == [20,20] and [len(b) for b in g.state['boards']] == [23,23]
    assert len(g.state['deck']) == 0 and g.state['market_tiles'].count(None) == 1
    assert g.outcome()['reason'] == 'forty_turns_complete' and not g.legal_actions()


def test_invalid_action_and_corrupt_snapshot_are_atomic():
    g = game(); before = g.snapshot()
    with pytest.raises(InvalidAction): g.step({'type':'select','tile_slot':False,'animal_slot':0,'nature':False})
    assert g.snapshot() == before
    corrupt = json.loads(json.dumps(before)); corrupt['state']['bag']['bear'] += 1
    with pytest.raises(AssertionError): g.restore(corrupt)
    assert g.snapshot() == before


@pytest.mark.parametrize('scores,nature,winners', [([90,91],[4,1],[1]),([90,90],[4,1],[0]),([90,90],[2,2],[0,1])])
def test_official_final_tie_break(scores,nature,winners,monkeypatch):
    g = game(); g.state['deck'] = []; g.state['turns'] = [19,20]; g.state['nature'] = nature
    monkeypatch.setattr(g,'scores',lambda:scores)
    g._end_turn()
    assert g.outcome()['winners'] == winners
