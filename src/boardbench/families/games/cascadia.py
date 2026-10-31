"""Cascadia, standard two-player game with the five official A scoring cards."""
from collections import Counter
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path

from boardbench.families.base import DuoGame


DATA = json.loads((Path(__file__).parents[1] / 'data/cascadia.json').read_text())
TILES = {t['id']: t for t in DATA['tiles']}
HABITATS = tuple(DATA['habitats'])
ANIMALS = tuple(DATA['animals'])
DIRECTIONS = tuple(tuple(v) for v in DATA['directions_clockwise'])


def neighbors(coord):
    q, r = coord
    return tuple((q + dq, r + dr) for dq, dr in DIRECTIONS)


def components(coords):
    unseen = set(coords)
    result = []
    while unseen:
        todo = [min(unseen)]; unseen.remove(todo[0]); group = set(todo)
        while todo:
            for n in neighbors(todo.pop()):
                if n in unseen:
                    unseen.remove(n); group.add(n); todo.append(n)
        result.append(group)
    return result


@lru_cache(maxsize=600)
def edges(tile, rotation):
    e = TILES[tile]['edge_habitats']
    return tuple(e[(i - rotation) % 6] for i in range(6))


def geometry(board):
    return tuple(sorted((c['q'], c['r'], c['tile'], c['rotation']) for c in board))


def wildlife(board):
    return tuple(sorted((c['q'], c['r'], c['animal']) for c in board if c['animal']))


@lru_cache(maxsize=32768)
def habitat_scores(layout):
    cells = {(q, r): edges(t, rot) for q, r, t, rot in layout}
    scores = []
    for habitat in HABITATS:
        unseen = {c for c, e in cells.items() if habitat in e}
        largest = 0
        while unseen:
            start = min(unseen); unseen.remove(start); todo = [start]; size = 1
            while todo:
                c = todo.pop()
                for side, n in enumerate(neighbors(c)):
                    if (n in unseen and cells[c][side] == habitat
                            and cells[n][(side + 3) % 6] == habitat):
                        unseen.remove(n); todo.append(n); size += 1
            largest = max(largest, size)
        scores.append(largest)
    return tuple(scores)


@lru_cache(maxsize=32768)
def elk_score(coords):
    """Optimal disjoint straight-line partition; adjacent groups are allowed."""
    total = 0
    for group in components(coords):
        ordered = sorted(group); index = {c: i for i, c in enumerate(ordered)}
        options = [[] for _ in ordered]
        patterns = {(1 << i, 2) for i in range(len(ordered))}
        for q, r in ordered:
            for dq, dr in DIRECTIONS[:3]:
                line = [(q, r)]
                for length in range(2, 5):
                    line.append((q + (length-1)*dq, r + (length-1)*dr))
                    if line[-1] not in group:
                        break
                    patterns.add((sum(1 << index[c] for c in line), (0, 2, 5, 9, 13)[length]))
        for mask, value in patterns:
            for i in range(len(ordered)):
                if mask & (1 << i):
                    options[i].append((mask, value))

        @lru_cache(None)
        def solve(mask):
            if not mask:
                return 0
            i = (mask & -mask).bit_length() - 1
            return max(value + solve(mask ^ part) for part, value in options[i]
                       if part & mask == part)
        total += solve((1 << len(ordered)) - 1)
    return total


@lru_cache(maxsize=32768)
def animal_scores(placed):
    by_type = {a: {(q, r) for q, r, x in placed if x == a} for a in ANIMALS}
    pairs = sum(len(g) == 2 for g in components(by_type['bear']))
    bear = (0, 4, 11, 19, 27)[min(pairs, 4)]
    elk = elk_score(tuple(sorted(by_type['elk'])))
    fish = by_type['salmon']
    valid_fish = {c for c in fish if sum(n in fish for n in neighbors(c)) <= 2}
    salmon = sum((0, 2, 5, 8, 12, 16, 20, 25)[min(len(g), 7)]
                 for g in components(valid_fish))
    hawks = by_type['hawk']
    isolated = sum(not any(n in hawks for n in neighbors(c)) for c in hawks)
    hawk = (0, 2, 5, 8, 11, 14, 18, 22, 26)[min(isolated, 8)]
    lookup = {(q, r): a for q, r, a in placed}
    fox = sum(len({lookup[n] for n in neighbors(c) if n in lookup}) for c in by_type['fox'])
    return bear, elk, salmon, hawk, fox


def fringe(board):
    occupied = {(c['q'], c['r']) for c in board}
    return sorted({n for c in occupied for n in neighbors(c)} - occupied)


def animal_value(board, coord, animal):
    placed = wildlife(board)
    gain = sum(animal_scores(tuple(sorted(placed + ((coord[0], coord[1], animal),)))))
    gain -= sum(animal_scores(placed))
    same = {(q, r) for q, r, a in placed if a == animal}
    # Preparation value is a policy feature only; official score stays exact.
    if animal == 'bear' and not any(n in same for n in neighbors(coord)):
        gain += 2.2
    if animal == 'fox':
        gain += .15 * sum(n not in {(q, r) for q, r, _ in placed} for n in neighbors(coord))
    return gain


def best_animal_value(board, animal):
    return max([0.] + [animal_value(board, (c['q'], c['r']), animal)
                      + int(TILES[c['tile']]['keystone']) for c in board
                      if c['animal'] is None and animal in TILES[c['tile']]['supported_animals']])


class Cascadia(DuoGame):
    game_id = 'cascadia'
    rules_version = '2021_base_designer_2023'
    config_id = 'standard_all_a_duo'

    def _reset(self, first_player):
        deck = list(DATA['normal_tile_ids']); self.rng.shuffle(deck)
        selected, removed = deck[:43], deck[43:]
        starters = self.rng.sample(DATA['starters'], 2)
        self.state.update(boards=[[dict(**c, animal=None) for c in st['cells']] for st in starters],
                          starter_ids=[st['id'] for st in starters], deck=selected, removed=removed,
                          market_tiles=[selected.pop() for _ in range(4)], market_animals=[],
                          bag={a: 20 for a in ANIMALS}, nature=[0, 0], nature_spent=[0, 0],
                          turns=[0, 0], phase='choose', free_wipe_used=False,
                          pending_tile=None, pending_animal=None, tile_slot=None, animal_slot=None,
                          public_history=[])
        self.state['market_animals'] = [self._draw_animal() for _ in range(4)]
        self._overpopulation()

    def _draw_animal(self):
        bag = self.state['bag']; draw = self.rng.randrange(sum(bag.values()))
        for a in ANIMALS:
            if draw < bag[a]:
                bag[a] -= 1
                return a
            draw -= bag[a]
        raise AssertionError('empty animal bag')

    def _wipe(self, slots, reason):
        s = self.state
        old = [s['market_animals'][i] for i in slots]
        new = [self._draw_animal() for _ in slots]
        for i, a in zip(slots, new):
            s['market_animals'][i] = a
        # Designer 2021-10-04: return after EACH replacement draw, before checking again.
        for a in old:
            s['bag'][a] += 1
        s['public_history'].append(dict(type='wipe', player=s['current_player'], reason=reason,
                                        slots=list(slots), removed=old, drawn=new))

    def _overpopulation(self):
        s = self.state
        while len(set(s['market_animals'])) == 1:
            self._wipe(list(range(4)), 'mandatory_four')
        s['phase'] = ('overpopulation' if not s['free_wipe_used']
                      and 3 in Counter(s['market_animals']).values() else 'choose')

    def _legal_actions(self):
        s = self.state; p = s['current_player']
        if s['phase'] == 'overpopulation':
            return [dict(type='keep'), dict(type='free_wipe')]
        if s['phase'] == 'choose':
            result = [dict(type='select', tile_slot=i, animal_slot=i, nature=False) for i in range(4)]
            if s['nature'][p]:
                result += [dict(type='select', tile_slot=i, animal_slot=j, nature=True)
                           for i in range(4) for j in range(4)]
                result += [dict(type='nature_wipe', slots=list(c)) for n in range(1, 5)
                           for c in combinations(range(4), n)]
            return result
        result = []; board = s['boards'][p]
        if s['pending_tile'] is not None:
            rotations = range(1 if TILES[s['pending_tile']]['keystone'] else 6)
            result += [dict(type='place_tile', q=q, r=r, rotation=rot)
                       for q, r in fringe(board) for rot in rotations]
        if s['pending_animal'] is not None:
            result += [dict(type='place_animal', q=c['q'], r=c['r']) for c in board
                       if c['animal'] is None and s['pending_animal'] in TILES[c['tile']]['supported_animals']]
            result.append(dict(type='discard_animal'))
        return result

    def _step(self, action):
        s = self.state; p = s['current_player']; typ = action['type']
        s['public_history'].append(dict(action, player=p))
        if typ == 'keep':
            s['phase'] = 'choose'
        elif typ == 'free_wipe':
            s['free_wipe_used'] = True
            a = next(a for a, n in Counter(s['market_animals']).items() if n == 3)
            self._wipe([i for i, x in enumerate(s['market_animals']) if x == a], 'free_three')
            self._overpopulation()
        elif typ == 'nature_wipe':
            s['nature'][p] -= 1; s['nature_spent'][p] += 1
            self._wipe(action['slots'], 'nature')
            self._overpopulation()
        elif typ == 'select':
            i, j = action['tile_slot'], action['animal_slot']
            if action['nature']:
                s['nature'][p] -= 1; s['nature_spent'][p] += 1
            s['tile_slot'], s['animal_slot'] = i, j
            s['pending_tile'], s['pending_animal'] = s['market_tiles'][i], s['market_animals'][j]
            s['market_tiles'][i] = s['market_animals'][j] = None
            s['phase'] = 'place'
        elif typ == 'place_tile':
            s['boards'][p].append(dict(q=action['q'], r=action['r'], rotation=action['rotation'],
                                       tile=s['pending_tile'], animal=None))
            s['pending_tile'] = None
        elif typ == 'place_animal':
            c = next(c for c in s['boards'][p] if (c['q'], c['r']) == (action['q'], action['r']))
            c['animal'] = s['pending_animal']; s['pending_animal'] = None
            s['nature'][p] += int(TILES[c['tile']]['keystone'])
        elif typ == 'discard_animal':
            s['bag'][s['pending_animal']] += 1; s['pending_animal'] = None
        if s['phase'] == 'place' and s['pending_tile'] is None and s['pending_animal'] is None:
            self._end_turn()

    def _end_turn(self):
        s = self.state; p = s['current_player']; s['turns'][p] += 1
        if not s['deck']:
            s['phase'] = 'finished'
            scores = self.scores(); best = max(scores)
            winners = [i for i in (0, 1) if scores[i] == best]
            if len(winners) == 2 and s['nature'][0] != s['nature'][1]:
                winners = [int(s['nature'][1] > s['nature'][0])]
            self.finish(winners, 'forty_turns_complete')
            return
        tile, animal = s['deck'].pop(), self._draw_animal()
        s['market_tiles'][s['tile_slot']] = tile
        s['market_animals'][s['animal_slot']] = animal
        s['public_history'].append(dict(type='refill', tile_slot=s['tile_slot'], tile=tile,
                                        animal_slot=s['animal_slot'], animal=animal))
        s['tile_slot'] = s['animal_slot'] = None
        s['current_player'] = 1-p; s['free_wipe_used'] = False
        self._overpopulation()

    def score_breakdown(self):
        habitats = [habitat_scores(geometry(b)) for b in self.state['boards']]
        result = []
        for p in (0, 1):
            animals = animal_scores(wildlife(self.state['boards'][p]))
            majority = [2 if a > b else int(a == b) for a, b in zip(habitats[p], habitats[1-p])]
            nature = self.state['nature'][p]
            result.append(dict(animals=dict(zip(ANIMALS, animals)),
                               habitats=dict(zip(HABITATS, habitats[p])),
                               majority=dict(zip(HABITATS, majority)), nature=nature,
                               total=sum(animals)+sum(habitats[p])+sum(majority)+nature))
        return result

    def scores(self):
        return [v['total'] for v in self.score_breakdown()]

    def _observe(self, player):
        s = self.state
        return {**{k: s[k] for k in ('boards', 'starter_ids', 'nature', 'turns', 'phase',
                                    'market_tiles', 'market_animals', 'pending_tile', 'pending_animal',
                                    'tile_slot', 'animal_slot', 'free_wipe_used', 'public_history')},
                'deck_size': len(s['deck']), 'bag_size': sum(s['bag'].values()),
                'scoring_cards': {a: 'A' for a in ANIMALS}, 'score_breakdown': self.score_breakdown()}

    def validate(self):
        super().validate(); s = self.state
        assert len(s['boards']) == 2 and len(set(s['starter_ids'])) == 2
        assert len(s['market_tiles']) == len(s['market_animals']) == 4
        assert s['phase'] in ('choose', 'overpopulation', 'place', 'finished')
        normal = s['deck'] + s['removed'] + [t for t in s['market_tiles'] if t is not None]
        if s['pending_tile'] is not None: normal.append(s['pending_tile'])
        animal_counts = Counter(s['bag'])
        animal_counts.update(a for a in s['market_animals'] if a is not None)
        if s['pending_animal'] is not None: animal_counts[s['pending_animal']] += 1
        assert set(s['bag']) == set(ANIMALS) and all(type(n) is int and n >= 0 for n in s['bag'].values())
        for p, board in enumerate(s['boards']):
            coords = {(c['q'], c['r']) for c in board}
            assert len(coords) == len(board) and len(components(coords)) == 1
            starter = DATA['starters'][s['starter_ids'][p]]['cells']
            assert all(any(all(c[k] == v for k, v in original.items()) for c in board) for original in starter)
            normal += [c['tile'] for c in board if c['tile'] >= 16]
            animal_counts.update(c['animal'] for c in board if c['animal'] is not None)
            for c in board:
                assert c['tile'] in TILES and type(c['rotation']) is int and 0 <= c['rotation'] < 6
                assert c['animal'] is None or c['animal'] in TILES[c['tile']]['supported_animals']
            earned = sum(c['animal'] is not None and TILES[c['tile']]['keystone'] for c in board)
            assert s['nature'][p] >= 0 and s['nature_spent'][p] >= 0
            assert earned == s['nature'][p] + s['nature_spent'][p]
            extra = int(s['phase'] == 'place' and p == s['current_player'] and s['pending_tile'] is None)
            assert len(board) == 3 + s['turns'][p] + extra
        assert sorted(normal) == list(range(16, 101)) and len(s['removed']) == 42
        assert animal_counts == Counter({a: 20 for a in ANIMALS})
        assert all(0 <= n <= 20 for n in s['turns'])
        assert abs(s['turns'][0]-s['turns'][1]) <= 1
        if s['phase'] in ('choose', 'overpopulation'):
            assert all(t is not None for t in s['market_tiles']+s['market_animals'])
            assert s['pending_tile'] is s['pending_animal'] is None
            assert len(set(s['market_animals'])) > 1
        elif s['phase'] == 'place':
            assert s['market_tiles'].count(None) == s['market_animals'].count(None) == 1
            assert s['pending_tile'] is not None or s['pending_animal'] is not None
        if s['terminated']:
            assert s['phase'] == 'finished' and s['turns'] == [20, 20] and not s['deck']

    @staticmethod
    def features(observation, action):
        p = observation['current_player']; board = observation['boards'][p]
        typ = action['type']; f = [0.] * 24
        f[20] = observation['turns'][p] / 20
        f[21] = observation['nature'][p] / 10
        if typ == 'place_animal':
            a = observation['pending_animal']; coord = (action['q'], action['r'])
            c = next(c for c in board if (c['q'], c['r']) == coord)
            f[1] = animal_value(board, coord, a); f[2] = int(TILES[c['tile']]['keystone'])
            f[0] = f[1] + 1.3*f[2]
        elif typ == 'place_tile':
            tile = observation['pending_tile']; coord = (action['q'], action['r'])
            before = habitat_scores(geometry(board))
            new = board + [dict(q=coord[0], r=coord[1], tile=tile, rotation=action['rotation'], animal=None)]
            after = habitat_scores(geometry(new))
            f[3] = sum(after)-sum(before)
            f[4] = sum(n in {(c['q'], c['r']) for c in board} for n in neighbors(coord))
            values = [animal_value(board, coord, a)+int(TILES[tile]['keystone']) for a in TILES[tile]['supported_animals']]
            f[5] = max(values)
            a = observation['pending_animal']
            if a in TILES[tile]['supported_animals']:
                f[6] = max(0., animal_value(board, coord, a)+int(TILES[tile]['keystone'])-best_animal_value(board, a))
            for i, v in enumerate(after): f[10+i] = v-before[i]
            f[0] = f[3] + .15*f[4] + .3*f[5] + .9*f[6] + .2
        elif typ == 'select':
            a = observation['market_animals'][action['animal_slot']]
            t = TILES[observation['market_tiles'][action['tile_slot']]]
            f[7] = max(best_animal_value(board, a), 2.5 if a in t['supported_animals'] else 0.)
            f[8] = len(t['habitats']) + .3*len(t['supported_animals'])
            f[9] = int(action['nature'])
            f[0] = f[7] + .8*f[8] - 1.5*f[9]
        elif typ == 'free_wipe':
            best = max(best_animal_value(board, a) for a in observation['market_animals'])
            f[0] = 2.4-best
        elif typ == 'keep':
            f[0] = 0.
        elif typ == 'nature_wipe':
            f[0] = -2.0
            f[9] = 1.
        elif typ == 'discard_animal':
            f[0] = -1.0
        return f


GAME_CLASS = Cascadia
