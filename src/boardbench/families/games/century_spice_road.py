"""Century: Spice Road, original standalone two-player configuration."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / 'data' / 'century_spice_road.json').read_text())
MERCHANTS = {c['id']: c for c in DATA['merchants']}
for _p in (0, 1):
    MERCHANTS[f'start-{_p}-produce'] = dict(id=f'start-{_p}-produce', kind='produce', gain=[2, 0, 0, 0])
    MERCHANTS[f'start-{_p}-upgrade'] = dict(id=f'start-{_p}-upgrade', kind='upgrade', steps=2)
ORDERS = {c['id']: c for c in DATA['orders']}


def bounded_vectors(limits, total):
    if len(limits) == 1:
        if 0 <= total <= limits[0]:
            yield [total]
        return
    for n in range(max(0, total - sum(limits[1:])), min(limits[0], total) + 1):
        for tail in bounded_vectors(limits[1:], total - n):
            yield [n] + tail


def payment_sequences(resources, count):
    if count == 0:
        yield []
        return
    for i, n in enumerate(resources):
        if n:
            remaining = list(resources); remaining[i] -= 1
            for tail in payment_sequences(remaining, count - 1):
                yield [i] + tail


def upgrade_sequences(resources, steps):
    # Each sequence is an actual choice of upgrades, including choosing none.
    yield []
    if steps:
        for i, n in enumerate(resources[:3]):
            if n:
                after = list(resources); after[i] -= 1; after[i + 1] += 1
                for tail in upgrade_sequences(after, steps - 1):
                    yield [i] + tail


def play_actions(card, resources):
    c = MERCHANTS[card]
    if c['kind'] == 'produce':
        return [dict(type='play', card=card)]
    if c['kind'] == 'upgrade':
        return [dict(type='play', card=card, upgrades=u) for u in upgrade_sequences(resources, c['steps'])]
    repeats = min(r // n for r, n in zip(resources, c['input']) if n)
    return [dict(type='play', card=card, repeats=n) for n in range(1, repeats + 1)]


def after_play(resources, action):
    c = MERCHANTS[action['card']]
    out = list(resources)
    if c['kind'] == 'produce':
        out = [r + n for r, n in zip(out, c['gain'])]
    elif c['kind'] == 'trade':
        out = [r + action['repeats'] * (b - a) for r, a, b in zip(out, c['input'], c['output'])]
    else:
        for i in action['upgrades']:
            out[i] -= 1; out[i + 1] += 1
    return out


class CenturySpiceRoad(DuoGame):
    game_id = 'century_spice_road'
    config_id = 'base_2017_duo'

    def _reset(self, first_player):
        merchants = [c['id'] for c in DATA['merchants']]
        orders = list(ORDERS)
        self.rng.shuffle(merchants); self.rng.shuffle(orders)
        resources = [[4, 0, 0, 0] for _ in range(2)]
        resources[first_player][0] = 3
        self.state.update(merchant_deck=merchants, order_deck=orders,
                          merchant_market=[], order_market=[], resources=resources,
                          hands=[[f'start-{p}-produce', f'start-{p}-upgrade'] for p in (0, 1)],
                          played=[[], []], claimed=[[], []], coins=[[0, 0], [0, 0]],
                          gold=4, silver=4, phase='main', turns=[0, 0],
                          final_round=False, history=[])
        for _ in range(6):
            self.state['merchant_market'].append(dict(card=merchants.pop(), resources=[0] * 4))
        for _ in range(5):
            self.state['order_market'].append(orders.pop())

    def _legal_actions(self):
        s, p = self.state, self.current_player
        resources = s['resources'][p]
        if s['phase'] == 'discard':
            return [dict(type='discard', resources=v) for v in bounded_vectors(resources, sum(resources) - 10)]
        actions = [dict(type='rest')]
        for c in s['hands'][p]:
            actions.extend(play_actions(c, resources))
        for i, entry in enumerate(s['merchant_market']):
            if i <= sum(resources):
                actions.extend(dict(type='acquire', card=entry['card'], payment=pay)
                               for pay in payment_sequences(resources, i))
        actions.extend(dict(type='claim', card=c) for c in s['order_market']
                       if all(r >= n for r, n in zip(resources, ORDERS[c]['cost'])))
        return actions

    def _step(self, action):
        s, p = self.state, self.current_player
        resources, typ = s['resources'][p], action['type']
        if typ == 'discard':
            s['resources'][p] = [r - n for r, n in zip(resources, action['resources'])]
            self._end_turn()
        else:
            if typ == 'rest':
                s['hands'][p].extend(s['played'][p]); s['played'][p] = []
            elif typ == 'play':
                s['hands'][p].remove(action['card']); s['played'][p].append(action['card'])
                s['resources'][p] = after_play(resources, action)
            elif typ == 'acquire':
                pos = next(i for i, entry in enumerate(s['merchant_market']) if entry['card'] == action['card'])
                for i, color in enumerate(action['payment']):
                    resources[color] -= 1
                    s['merchant_market'][i]['resources'][color] += 1
                entry = s['merchant_market'].pop(pos)
                s['hands'][p].append(entry['card'])
                s['resources'][p] = [r + n for r, n in zip(resources, entry['resources'])]
                if s['merchant_deck']:
                    s['merchant_market'].append(dict(card=s['merchant_deck'].pop(), resources=[0] * 4))
            elif typ == 'claim':
                c = action['card']; pos = s['order_market'].index(c)
                s['resources'][p] = [r - n for r, n in zip(resources, ORDERS[c]['cost'])]
                if pos == 0 and s['gold']:
                    s['gold'] -= 1; s['coins'][p][0] += 1
                elif s['silver'] and pos == (1 if s['gold'] else 0):
                    s['silver'] -= 1; s['coins'][p][1] += 1
                s['order_market'].pop(pos)
                s['claimed'][p].append(c)
                if s['order_deck']:
                    s['order_market'].append(s['order_deck'].pop())
                if len(s['claimed'][p]) >= 6:
                    s['final_round'] = True
            if sum(s['resources'][p]) > 10:
                s['phase'] = 'discard'
            else:
                self._end_turn()
        s['history'].append(dict(player=p, action=deepcopy(action),
                                 merchant_market=deepcopy(s['merchant_market']),
                                 order_market=list(s['order_market'])))

    def _end_turn(self):
        s, p = self.state, self.current_player
        s['phase'] = 'main'; s['turns'][p] += 1
        if s['final_round'] and p != s['first_player']:
            scores = self.scores()
            # Last in seat order wins an exact point tie.
            ranks = [(scores[q], int(q != s['first_player'])) for q in (0, 1)]
            self.finish([max((0, 1), key=lambda q: ranks[q])], 'equal_turns_after_six_orders')
        else:
            s['current_player'] = 1 - p

    def scores(self):
        s = self.state
        return [sum(ORDERS[c]['points'] for c in s['claimed'][p]) + 3*s['coins'][p][0]
                + s['coins'][p][1] + sum(s['resources'][p][1:]) for p in (0, 1)]

    def score_breakdown(self):
        s = self.state
        return [dict(total=self.scores()[p], orders=sum(ORDERS[c]['points'] for c in s['claimed'][p]),
                     gold=3*s['coins'][p][0], silver=s['coins'][p][1],
                     remaining_spices=sum(s['resources'][p][1:])) for p in (0, 1)]

    def _observe(self, player):
        s = self.state
        return dict(phase=s['phase'], first_player=s['first_player'],
                    resources=deepcopy(s['resources']), hand=list(s['hands'][player]),
                    hand_counts=[len(h) for h in s['hands']], played=deepcopy(s['played']),
                    claimed=list(s['claimed'][player]), claimed_counts=[len(c) for c in s['claimed']],
                    coins=deepcopy(s['coins']), own_score=self.scores()[player],
                    merchant_market=deepcopy(s['merchant_market']), order_market=list(s['order_market']),
                    merchant_deck_count=len(s['merchant_deck']), order_deck_count=len(s['order_deck']),
                    gold=s['gold'], silver=s['silver'], turns=list(s['turns']),
                    final_round=s['final_round'], history=deepcopy(s['history']))

    def validate(self):
        super().validate()
        s = self.state
        merchants = s['merchant_deck'] + [e['card'] for e in s['merchant_market']]
        merchants += [c for key in ('hands', 'played') for row in s[key] for c in row]
        assert Counter(merchants) == Counter(MERCHANTS.keys())
        orders = s['order_deck'] + s['order_market'] + [c for row in s['claimed'] for c in row]
        assert Counter(orders) == Counter(ORDERS.keys())
        assert s['gold'] + sum(c[0] for c in s['coins']) == 4
        assert s['silver'] + sum(c[1] for c in s['coins']) == 4
        assert min([s['gold'], s['silver']] + [n for c in s['coins'] for n in c]) >= 0
        for p in (0, 1):
            assert all(type(n) is int and n >= 0 for n in s['resources'][p])
            assert sum(s['resources'][p]) <= 10 or (s['phase'] == 'discard' and p == self.current_player)
        assert s['phase'] in ('main', 'discard')
        assert len(s['merchant_market']) <= 6 and len(s['order_market']) <= 5
        for e in s['merchant_market']:
            assert all(type(n) is int and n >= 0 for n in e['resources'])
        if s['terminated']:
            assert s['turns'][0] == s['turns'][1]

    @staticmethod
    def features(obs, action):
        p, typ = obs['player'], action['type']
        resources = list(obs['resources'][p])
        def utility(r):
            values = []
            for j, cid in enumerate(obs['order_market']):
                c = ORDERS[cid]
                missing = sum(max(0, n-x)*(i+1) for i, (n,x) in enumerate(zip(c['cost'], r)))
                coin = 3 if j == 0 and obs['gold'] else 1 if obs['silver'] and j == (1 if obs['gold'] else 0) else 0
                values.append((c['points'] + coin + 4) / (1 + missing * .5))
            return max(values, default=0) + .10 * sum((i+1)*n for i,n in enumerate(r)) - .65 * max(0, sum(r)-10)
        before = utility(resources)
        points, acquired, rest, delta = 0., 0., 0., 0.
        if typ == 'claim':
            points = ORDERS[action['card']]['points']
            value = 30 + points
        elif typ == 'play':
            after = after_play(resources, action)
            delta = utility(after)-before
            value = delta + .10
        elif typ == 'discard':
            after = [r-n for r,n in zip(resources,action['resources'])]
            delta = utility(after)-before
            value = delta
        elif typ == 'acquire':
            entry = next(e for e in obs['merchant_market'] if e['card'] == action['card'])
            after = list(resources)
            for color in action['payment']: after[color] -= 1
            after = [r+n for r,n in zip(after,entry['resources'])]
            possibilities = play_actions(action['card'], after)
            gain = max([utility(after_play(after,a))-utility(after) for a in possibilities] + [0.])
            c = MERCHANTS[action['card']]
            total_cards = len(obs['hand']) + len(obs['played'][p])
            acquired = .65*gain + (1.0 if c['kind']=='produce' else .25) - .18*total_cards
            delta = utility(after)-before
            value = acquired + delta
        else:
            rest = len(obs['played'][p])
            gain = max([utility(after_play(resources,a))-before for c in obs['played'][p] for a in play_actions(c,resources)] + [0.])
            value = .65*gain + .12*rest - .3
        f = [value, points, delta, acquired, rest, sum(resources),
             sum((i+1)*n for i,n in enumerate(resources)), float(typ=='play'),
             float(typ=='claim'), float(typ=='acquire'), float(typ=='discard'),
             obs['claimed_counts'][p], float(obs['final_round'])]
        return f + [0.] * (24-len(f))


GAME_CLASS = CenturySpiceRoad
