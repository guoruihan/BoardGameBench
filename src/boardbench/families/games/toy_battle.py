"""Toy Battle on official Castle Field first-game setup (rulebook p3).

All troop effects apply. Printed special bases act as ordinary bases, an
explicit official introductory configuration, not the standard advanced setup.
"""
from collections import Counter, deque
import json
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / 'data/toy_battle.json').read_text())
NODES = tuple(n['id'] for n in DATA['nodes'])
HQ = tuple(DATA['headquarters'])
ADJ = {n: [] for n in NODES}
for a, b in DATA['edges']:
    ADJ[a].append(b)
    ADJ[b].append(a)
REGIONS = tuple(DATA['regions'])


def connected(board, player):
    result, todo = {HQ[player]}, [HQ[player]]
    while todo:
        n = todo.pop()
        for other in ADJ[n]:
            if other not in result and board[other] and board[other][-1]['owner'] == player:
                result.add(other)
                todo.append(other)
    return result


def targets(board, player, troop):
    reach = connected(board, player)
    frontier = {n for v in reach for n in ADJ[v]} | reach
    result = []
    for node in NODES:
        if node == HQ[player]:
            continue
        if node not in frontier and not (troop == 4 and node not in HQ):
            continue
        stack = board[node]
        if stack and stack[-1]['owner'] != player:
            opposing = stack[-1]['troop']
            if troop != 0 and opposing != 0 and troop <= opposing:
                continue
        result.append(node)
    return result


def distance(board, player):
    """Public minimum new positions to connect to enemy HQ; heuristic only."""
    dist = {HQ[player]: 0}
    todo = deque([HQ[player]])
    while todo:
        n = todo.popleft()
        for other in ADJ[n]:
            own = bool(board[other] and board[other][-1]['owner'] == player)
            d = dist[n] + int(not own)
            if d < dist.get(other, 100):
                dist[other] = d
                if own:
                    todo.appendleft(other)
                else:
                    todo.append(other)
    return dist[HQ[1-player]]


class ToyBattle(DuoGame):
    game_id = 'toy_battle'
    config_id = 'castle_field_intro_duo'

    def _reset(self, first_player):
        s = self.state
        s.update(board={n: [] for n in NODES}, hands=[[], []], reserves=[[], []],
                 removed=[[], []], discard=[[], []], medals=[0, 0],
                 claimed=[None]*len(REGIONS), phase='turn', effect_source=None,
                 public_history=[])
        for p in (0, 1):
            deck = list(range(24))
            self.rng.shuffle(deck)
            s['removed'][p], s['reserves'][p] = deck[:4], deck[4:]
            self._draw(p, 3 if p == first_player else 4)

    def _draw(self, player, count):
        s = self.state
        count = min(count, 8-len(s['hands'][player]), len(s['reserves'][player]))
        for _ in range(count):
            s['hands'][player].append(s['reserves'][player].pop())
        s['public_history'].append(dict(type='draw', player=player, count=count))

    def _observe(self, player):
        s = self.state
        return dict(board=s['board'], hand=[dict(uid=u, troop=u//3) for u in s['hands'][player]],
                    hand_sizes=[len(h) for h in s['hands']],
                    reserve_sizes=[len(d) for d in s['reserves']],
                    removed_counts=[4, 4],
                    discard=[[dict(uid=u, troop=u//3) for u in d] for d in s['discard']],
                    medals=s['medals'], claimed=s['claimed'], phase=s['phase'],
                    effect_source=s['effect_source'], public_history=s['public_history'],
                    map=DATA, special_base_effects=False)

    def _placements(self, player):
        return [dict(type='place', troop=t, node=n)
                for t in sorted({u//3 for u in self.state['hands'][player]})
                for n in targets(self.state['board'], player, t)]

    def _legal_actions(self):
        s, p = self.state, self.state['current_player']
        if s['phase'] in ('turn', 'extra_place'):
            result = self._placements(p)
            if s['phase'] == 'extra_place':
                result.append(dict(type='skip'))
            elif s['reserves'][p] and len(s['hands'][p]) < 8:
                result.append(dict(type='draw'))
            return result
        result = [dict(type='skip')]
        if s['phase'] == 'bonus_draw':
            if s['reserves'][p] and len(s['hands'][p]) < 8:
                result.append(dict(type='use_effect'))
        elif s['phase'] == 'random_discard':
            if s['hands'][1-p]:
                result.append(dict(type='use_effect'))
        elif s['phase'] == 'remove':
            for node in ADJ[s['effect_source']['node']]:
                if s['board'][node] and s['board'][node][-1]['owner'] != p:
                    result.append(dict(type='remove', node=node))
        return result

    def _step(self, action):
        s, p = self.state, self.state['current_player']
        kind = action['type']
        if kind == 'skip':
            self._end_turn()
        elif kind == 'draw':
            self._draw(p, 2)
            self._end_turn()
        elif kind == 'place':
            troop, node = action['troop'], action['node']
            uid = next(u for u in s['hands'][p] if u//3 == troop)
            s['hands'][p].remove(uid)
            piece = dict(owner=p, uid=uid, troop=troop)
            s['board'][node].append(piece)
            s['public_history'].append(dict(type='place', node=node, **piece))
            if node == HQ[1-p]:
                self.finish([p], reason='enemy_headquarters_captured')
                return
            if self._claim_regions():
                return
            s['effect_source'] = dict(node=node, **piece)
            s['phase'] = {1: 'bonus_draw', 2: 'extra_place', 3: 'remove',
                          5: 'random_discard', 6: 'bonus_draw'}.get(troop, 'turn')
            if s['phase'] == 'turn':
                self._end_turn()
        elif kind == 'remove':
            piece = s['board'][action['node']].pop()
            s['discard'][piece['owner']].append(piece['uid'])
            s['public_history'].append(dict(type='remove', node=action['node'], **piece))
            if not self._claim_regions():
                self._end_turn()
        elif kind == 'use_effect':
            if s['phase'] == 'bonus_draw':
                self._draw(p, 2 if s['effect_source']['troop'] == 1 else 1)
            else:
                uid = s['hands'][1-p].pop(self.rng.randrange(len(s['hands'][1-p])))
                s['discard'][1-p].append(uid)
                s['public_history'].append(dict(type='random_discard', owner=1-p,
                                               uid=uid, troop=uid//3))
            self._end_turn()

    def _claim_regions(self):
        s = self.state
        for i, region in enumerate(REGIONS):
            if s['claimed'][i] is not None:
                continue
            owners = {s['board'][n][-1]['owner'] if s['board'][n] else None
                      for n in region['boundary']}
            if len(owners) == 1 and None not in owners:
                p = next(iter(owners))
                s['claimed'][i] = p
                s['medals'][p] += region['medals']
                s['public_history'].append(dict(type='claim', player=p, region=i,
                                               medals=region['medals']))
                if s['medals'][p] >= DATA['medal_targets'][p]:
                    self.finish([p], reason='medal_objective_reached')
                    return True
        return False

    def _end_turn(self):
        s = self.state
        s['phase'], s['effect_source'] = 'turn', None
        s['current_player'] = 1-s['current_player']
        if not self._legal_actions():
            p = s['current_player']
            winner = p if s['medals'][p] > s['medals'][1-p] else 1-p
            self.finish([winner], reason='active_player_has_no_draw_or_placement')

    def scores(self):
        return list(self.state['medals'])

    def score_breakdown(self):
        return [dict(medals=self.state['medals'][p], total=self.state['medals'][p],
                     claimed_regions=sum(v == p for v in self.state['claimed'])) for p in (0, 1)]

    def validate(self):
        super().validate()
        s = self.state
        if set(s['board']) != set(NODES):
            raise ValueError('board nodes mismatch')
        seen = [list(s['hands'][p])+s['reserves'][p]+s['removed'][p]+s['discard'][p]
                for p in (0, 1)]
        for node, stack in s['board'].items():
            for piece in stack:
                p, uid = piece['owner'], piece['uid']
                if p not in (0, 1) or type(uid) is not int or piece['troop'] != uid//3:
                    raise ValueError('invalid troop')
                if node == HQ[p]:
                    raise ValueError('troop on own headquarters')
                seen[p].append(uid)
        for p in (0, 1):
            if sorted(seen[p]) != list(range(24)) or len(s['removed'][p]) != 4:
                raise ValueError('troop conservation failure')
            if not 0 <= len(s['hands'][p]) <= 8:
                raise ValueError('rack overflow')
            expected = sum(r['medals'] for i, r in enumerate(REGIONS) if s['claimed'][i] == p)
            if s['medals'][p] != expected:
                raise ValueError('medals mismatch')
        if len(s['claimed']) != len(REGIONS) or any(v not in (None, 0, 1) for v in s['claimed']):
            raise ValueError('invalid region owner')
        if s['phase'] not in ('turn', 'extra_place', 'bonus_draw', 'random_discard', 'remove'):
            raise ValueError('invalid phase')
        if not s['terminated'] and s['phase'] != 'turn' and not s['effect_source']:
            raise ValueError('missing effect source')

    @staticmethod
    def features(observation, action):
        p = observation['current_player']
        board = observation['board']
        kind = action['type']
        values = dict(gain=0, capture=0, threat=0, control=0, strength=0,
                      draw=0, discard=0, connection=0, progress=0, enemy_progress=0,
                      region_progress=0, own_cover=0, flexibility=0, effect=0)
        if kind in ('place', 'remove'):
            node = action['node']
            before = board[node][-1] if board[node] else None
            projected = dict(board)
            projected[node] = list(board[node])
            if kind == 'place':
                troop = action['troop']
                values['strength'] = troop
                values['capture'] = int(before is not None and before['owner'] != p)
                values['own_cover'] = int(before is not None and before['owner'] == p)
                values['control'] = int(before is None or before['owner'] != p)
                values['effect'] = float(troop in (1, 2, 3, 5, 6))
                values['threat'] = int(node == HQ[1-p])
                projected[node].append(dict(owner=p, troop=troop))
                values['draw'] = min(2 if troop == 1 else int(troop == 6),
                                     observation['reserve_sizes'][p],
                                     9-observation['hand_sizes'][p])
                values['discard'] = int(troop == 5 and observation['hand_sizes'][1-p] > 0)
            else:
                projected[node].pop()
                values['capture'] = 1
                values['strength'] = before['troop']
                values['control'] = int(bool(projected[node]) and projected[node][-1]['owner'] == p)
            for i, r in enumerate(REGIONS):
                if observation['claimed'][i] is not None:
                    continue
                old = sum(bool(board[n]) and board[n][-1]['owner'] == p for n in r['boundary'])
                new = sum(bool(projected[n]) and projected[n][-1]['owner'] == p for n in r['boundary'])
                values['gain'] += r['medals']*int(new == len(r['boundary']))
                values['region_progress'] += r['medals']*(new*new-old*old)/len(r['boundary'])**2
            values['connection'] = len(connected(projected, p))-len(connected(board, p))
            values['progress'] = distance(board, p)-distance(projected, p)
            values['enemy_progress'] = distance(projected, 1-p)-distance(board, 1-p)
            values['flexibility'] = len({n for v in connected(projected,p) for n in ADJ[v]})
        elif kind in ('draw', 'use_effect'):
            if observation['phase'] == 'random_discard':
                values['discard'] = 1
            else:
                amount = 2 if kind == 'draw' or observation['effect_source']['troop'] == 1 else 1
                values['draw'] = min(amount, 8-observation['hand_sizes'][p], observation['reserve_sizes'][p])
        score = (100*values['threat'] + 12*values['gain'] + 2.4*values['capture']
                 + 0.8*values['control'] + 0.08*values['strength'] + 1.3*values['draw']
                 + 1.8*values['discard'] + 0.35*values['connection'] + 1.8*values['progress']
                 + 1.5*values['enemy_progress'] + 2.2*values['region_progress']
                 - 0.6*values['own_cover'] + 0.05*values['flexibility'] + 0.2*values['effect'])
        if kind == 'skip':
            score = -0.2
        f = [score]+list(values.values())+[observation['medals'][p], observation['medals'][1-p],
              observation['hand_sizes'][p], observation['hand_sizes'][1-p],
              observation['reserve_sizes'][p], observation['reserve_sizes'][1-p],
              float(kind == 'skip'), float(kind == 'draw'), float(observation['phase'] == 'extra_place')]
        return [float(v) for v in f]


GAME_CLASS = ToyBattle
