"""Classic Patchwork, two players, official corrected leather track.

Complete placement enumeration uses 81-bit integer boards. Distinct rotations
and reflections are canonicalized only when they occupy identical cells. All
ordinary purchases are atomic choices; special patches are immediate phases.
"""
from functools import lru_cache
import json
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / 'data/patchwork.json').read_text())
PATCHES = {p['id']: p for p in DATA['patches']}
INCOME = tuple(DATA['income_events'])
LEATHER = tuple(DATA['leather_events'])
END = DATA['end']
FULL = (1 << 81) - 1
LEFT = sum(1 << (9*y) for y in range(9))
RIGHT = LEFT << 8
WINDOWS = tuple(sum(1 << (9*(y+dy)+x+dx) for dy in range(7) for dx in range(7))
                for y in range(3) for x in range(3))


def orientations(cells):
    """Unique normalized dihedral shapes, with stable ordering."""
    shapes = set()
    for reflect in (1, -1):
        pts = [(reflect*x, y) for x, y in cells]
        for _ in range(4):
            xmin, ymin = min(x for x, y in pts), min(y for x, y in pts)
            shapes.add(tuple(sorted((x-xmin, y-ymin) for x, y in pts)))
            pts = [(-y, x) for x, y in pts]
    return tuple(sorted(shapes))


ORIENTATIONS = {p: orientations(v['cells']) for p, v in PATCHES.items()}
PLACEMENTS = {}
MASKS = {}
for _p, _shapes in ORIENTATIONS.items():
    _placements = []
    for _o, _cells in enumerate(_shapes):
        for _y in range(9-max(y for x, y in _cells)):
            for _x in range(9-max(x for x, y in _cells)):
                _mask = sum(1 << (9*(_y+dy)+_x+dx) for dx, dy in _cells)
                _placements.append((_o, _x, _y, _mask))
                MASKS[(_p, _o, _x, _y)] = _mask
    PLACEMENTS[_p] = tuple(_placements)


def neighbors(mask):
    return (((mask & ~RIGHT) << 1) | ((mask & ~LEFT) >> 1)
            | (mask << 9) | (mask >> 9)) & FULL


@lru_cache(maxsize=16384)
def board_metrics(mask):
    empty = FULL ^ mask
    isolated = (empty & ~neighbors(empty)).bit_count()
    closest = max((mask & window).bit_count() for window in WINDOWS)
    return isolated, closest


class Patchwork(DuoGame):
    game_id = 'patchwork'
    config_id = 'classic_corrected_duo'

    def _reset(self, first_player):
        ring = list(PATCHES)
        self.rng.shuffle(ring)
        self.state.update(boards=[0, 0], buttons=[5, 5], income=[0, 0],
                          positions=[0, 0], top=first_player, ring=ring,
                          neutral=(ring.index('A')+1) % len(ring),
                          placements=[[], []], leather=list(LEATHER),
                          discarded_leather=[], bonus_owner=None, arrivals=[],
                          phase='turn', pending_events=[])

    def _observe(self, player):
        s = self.state
        return dict(phase=s['phase'], board_masks=[hex(v) for v in s['boards']],
                    boards=[[''.join('#' if m & (1 << (9*y+x)) else '.'
                                     for x in range(9)) for y in range(9)]
                            for m in s['boards']], buttons=s['buttons'],
                    income=s['income'], positions=s['positions'], top=s['top'],
                    ring=s['ring'], neutral=s['neutral'], patches=DATA['patches'],
                    placements=s['placements'], leather=s['leather'],
                    discarded_leather=s['discarded_leather'],
                    bonus_owner=s['bonus_owner'], arrivals=s['arrivals'],
                    pending_events=s['pending_events'], income_events=list(INCOME),
                    end=END, score_breakdown=self.score_breakdown())

    def _legal_actions(self):
        s = self.state
        p = s['current_player']
        board = s['boards'][p]
        if s['phase'] == 'leather':
            return [dict(type='leather', x=x, y=y) for y in range(9) for x in range(9)
                    if not board & (1 << (9*y+x))]
        actions = [dict(type='advance')]
        ring = s['ring']
        for i in range(min(3, len(ring))):
            patch = ring[(s['neutral']+i) % len(ring)]
            if PATCHES[patch]['cost'] > s['buttons'][p]:
                continue
            actions.extend(dict(type='buy', patch=patch, orientation=o, x=x, y=y)
                           for o, x, y, mask in PLACEMENTS[patch] if not mask & board)
        return actions

    def _bonus(self, player):
        if self.state['bonus_owner'] is None:
            board = self.state['boards'][player]
            if any(board & window == window for window in WINDOWS):
                self.state['bonus_owner'] = player

    def _step(self, action):
        s, p = self.state, self.state['current_player']
        if action['type'] == 'leather':
            event = s['pending_events'].pop(0)
            mask = 1 << (9*action['y']+action['x'])
            s['boards'][p] |= mask
            s['placements'][p].append(dict(patch='leather', event=event,
                                           x=action['x'], y=action['y']))
            self._bonus(p)
            self._resolve_events()
            return
        old = s['positions'][p]
        if action['type'] == 'advance':
            new = min(s['positions'][1-p]+1, END)
            s['buttons'][p] += new-old
        else:
            patch = action['patch']
            spec = PATCHES[patch]
            index = s['ring'].index(patch)
            s['ring'].pop(index)
            s['neutral'] = index % len(s['ring']) if s['ring'] else 0
            s['buttons'][p] -= spec['cost']
            s['income'][p] += spec['income']
            s['boards'][p] |= MASKS[(patch, action['orientation'], action['x'], action['y'])]
            s['placements'][p].append({k: v for k, v in action.items() if k != 'type'})
            self._bonus(p)
            new = min(old+spec['time'], END)
        s['positions'][p] = new
        s['top'] = p
        if new == END:
            s['arrivals'].append(p)
        s['pending_events'] = sorted(e for e in (*INCOME, *LEATHER) if old < e <= new)
        self._resolve_events()

    def _resolve_events(self):
        s, p = self.state, self.state['current_player']
        while s['pending_events']:
            event = s['pending_events'][0]
            if event in INCOME:
                s['buttons'][p] += s['income'][p]
            elif event in s['leather']:
                s['leather'].remove(event)
                if s['boards'][p] != FULL:
                    s['phase'] = 'leather'
                    return
                s['discarded_leather'].append(event)
            s['pending_events'].pop(0)
        s['phase'] = 'turn'
        if s['positions'] == [END, END]:
            scores = self.scores()
            winner = s['arrivals'][0] if scores[0] == scores[1] else int(scores[1] > scores[0])
            self.finish([winner], reason='both_time_tokens_at_end')
        elif s['positions'][0] == s['positions'][1]:
            s['current_player'] = s['top']
        else:
            s['current_player'] = int(s['positions'][1] < s['positions'][0])

    def score_breakdown(self):
        s = self.state
        return [dict(buttons=s['buttons'][p], bonus=7*int(s['bonus_owner'] == p),
                     empty=81-s['boards'][p].bit_count(),
                     total=s['buttons'][p]+7*int(s['bonus_owner'] == p)
                           -2*(81-s['boards'][p].bit_count())) for p in (0, 1)]

    def scores(self):
        return [entry['total'] for entry in self.score_breakdown()]

    def validate(self):
        super().validate()
        s = self.state
        ordinary, leather = list(s['ring']), list(s['leather'])+list(s['discarded_leather'])
        if s['phase'] not in ('turn', 'leather'):
            raise ValueError('invalid phase')
        for p in (0, 1):
            board, income = 0, 0
            for placement in s['placements'][p]:
                patch = placement['patch']
                if patch == 'leather':
                    leather.append(placement['event'])
                    x, y = placement['x'], placement['y']
                    if not (type(x) is int and type(y) is int and 0 <= x < 9 and 0 <= y < 9):
                        raise ValueError('invalid leather placement')
                    mask = 1 << (9*y+x)
                else:
                    ordinary.append(patch)
                    income += PATCHES[patch]['income']
                    mask = MASKS[(patch, placement['orientation'], placement['x'], placement['y'])]
                if board & mask:
                    raise ValueError('overlapping patches')
                board |= mask
            if board != s['boards'][p] or income != s['income'][p]:
                raise ValueError('board/income inconsistent with placements')
            if not (0 <= s['positions'][p] <= END and s['buttons'][p] >= 0):
                raise ValueError('invalid time/buttons')
        if s['phase'] == 'leather':
            if not s['pending_events'] or s['pending_events'][0] not in LEATHER:
                raise ValueError('missing pending leather')
            leather.append(s['pending_events'][0])
            if s['boards'][s['current_player']] == FULL:
                raise ValueError('full board cannot await leather placement')
        if sorted(ordinary) != sorted(PATCHES) or sorted(leather) != sorted(LEATHER):
            raise ValueError('component conservation failure')
        if not 0 <= s['neutral'] < max(1, len(s['ring'])):
            raise ValueError('invalid neutral gap')
        if s['top'] not in (0, 1) or s['bonus_owner'] not in (None, 0, 1):
            raise ValueError('invalid marker owner')
        if len(s['arrivals']) != len(set(s['arrivals'])):
            raise ValueError('duplicate arrival')
        if sorted(s['arrivals']) != [p for p in (0, 1) if s['positions'][p] == END]:
            raise ValueError('invalid arrival list')
        if s['terminated'] and (s['positions'] != [END, END] or s['pending_events']):
            raise ValueError('premature terminal')

    @staticmethod
    def features(observation, action):
        p = observation['current_player']
        board = int(observation['board_masks'][p], 16)
        old = observation['positions'][p]
        oldholes, oldwindow = board_metrics(board)
        area = cost = income = contacts = holes = window = bonus = leather = 0
        remaining = sum(event > old for event in INCOME)
        if action['type'] == 'advance':
            time = min(observation['positions'][1-p]+1, END)-old
            net = time
            score = 1.0
        else:
            if action['type'] == 'buy':
                patch = PATCHES[action['patch']]
                mask = MASKS[(action['patch'], action['orientation'], action['x'], action['y'])]
                area, cost, income = len(patch['cells']), patch['cost'], patch['income']
                time = min(patch['time'], END-old)
            else:
                mask = 1 << (9*action['y']+action['x'])
                area, time = 1, 0
            filled = board | mask
            newholes, newwindow = board_metrics(filled)
            holes, window = newholes-oldholes, newwindow-oldwindow
            contacts = (neighbors(board) & mask).bit_count()
            contacts += (mask & (LEFT | RIGHT | 511 | (511 << 72))).bit_count()
            bonus = int(observation['bonus_owner'] is None and newwindow == 49)
            net = 2*area-cost+income*remaining
            score = net/max(1, time) + 0.16*contacts - 0.85*holes + 0.08*window+7*bonus
        new = old+time
        leather = sum(old < event <= new for event in observation['leather'])
        score += 1.1*leather
        features = [score, area, cost, income, time, remaining, net, contacts,
                    holes, window, bonus, leather, old/END,
                    observation['buttons'][p], observation['income'][p],
                    board.bit_count()/81, observation['positions'][1-p]-old,
                    float(action['type'] == 'advance'), float(action['type'] == 'leather'),
                    float(new == END), sum(old < event <= new for event in INCOME),
                    len(observation['ring']), oldholes, oldwindow/49]
        return [float(v) for v in features]


GAME_CLASS = Patchwork
