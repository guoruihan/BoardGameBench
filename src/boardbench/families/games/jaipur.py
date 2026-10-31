"""Space Cowboys 2019 Jaipur, complete match (first to two seals).

Cards with identical faces are interchangeable; exchange vectors enumerate every
legal multiset. A round tied after both printed tiebreaks is explicitly unresolved.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / 'data' / 'jaipur.json').read_text())
GOODS = tuple(DATA['goods_tokens_descending'])
KINDS = GOODS + ('camel',)


def multisets(limits, total, prefix=()):
    """All bounded count vectors with a prescribed total, without permutations."""
    if len(limits) == 1:
        if 0 <= total <= limits[0]:
            yield list(prefix + (total,))
        return
    for n in range(max(0, total - sum(limits[1:])), min(limits[0], total) + 1):
        yield from multisets(limits[1:], total - n, prefix + (n,))


class Jaipur(DuoGame):
    game_id = 'jaipur'
    config_id = 'space_cowboys_2019_duo_match'

    def _reset(self, first_player):
        self.state.update(seals=[0, 0], round_number=0, rounds=[], history=[],
                          blocked_reason=None)
        self._new_round(first_player)

    def _new_round(self, first_player):
        s = self.state
        deck = [g for g, n in DATA['cards'].items() for _ in range(n - (3 if g == 'camel' else 0))]
        self.rng.shuffle(deck)
        hands, herds = [[0] * 6 for _ in range(2)], [0, 0]
        for p in (0, 1):
            for _ in range(5):
                g = deck.pop()
                if g == 'camel':
                    herds[p] += 1
                else:
                    hands[p][GOODS.index(g)] += 1
        bonus = {k: list(v) for k, v in DATA['bonus_token_multisets'].items()}
        for pile in bonus.values():
            self.rng.shuffle(pile)
        s.update(round_number=s['round_number'] + 1, current_player=first_player,
                 market=['camel'] * 3 + [deck.pop(), deck.pop()], deck=deck,
                 hands=hands, herds=herds, discard=[],
                 goods_tokens=deepcopy(DATA['goods_tokens_descending']), bonus_tokens=bonus,
                 goods_won=[[], []], bonus_won=[[], []], camel_award=[0, 0])
        s['history'].append(dict(type='round_start', round=s['round_number'],
                                 first_player=first_player, market=list(s['market'])))

    def _legal_actions(self):
        s, p = self.state, self.current_player
        if s['blocked_reason']:
            return []
        h = s['hands'][p]
        market_counts = [s['market'].count(g) for g in GOODS]
        actions = []
        if sum(h) < 7:
            actions += [dict(type='take', good=g) for g, n in zip(GOODS, market_counts) if n]
        if 'camel' in s['market']:
            actions.append(dict(type='camels'))
        for g, n in zip(GOODS, h):
            actions += [dict(type='sell', good=g, count=q)
                        for q in range(DATA['minimum_sale'][g], n + 1)]
        for n in range(2, sum(market_counts) + 1):
            for take in multisets(market_counts, n):
                limits = [0 if take[i] else h[i] for i in range(6)] + [s['herds'][p]]
                for give in multisets(limits, n):
                    if sum(h) + give[6] <= 7:
                        actions.append(dict(type='exchange', take=take, give=give))
        return actions

    def _step(self, action):
        s, p = self.state, self.current_player
        h = s['hands'][p]
        typ = action['type']
        event = dict(player=p, round=s['round_number'], action=deepcopy(action))
        refill = False
        if typ == 'take':
            g = action['good']
            h[GOODS.index(g)] += 1
            s['market'].remove(g)
            refill = True
        elif typ == 'camels':
            n = s['market'].count('camel')
            s['herds'][p] += n
            s['market'] = [g for g in s['market'] if g != 'camel']
            refill = True
        elif typ == 'exchange':
            for i, n in enumerate(action['take']):
                h[i] += n
                for _ in range(n):
                    s['market'].remove(GOODS[i])
            for i, n in enumerate(action['give']):
                if i == 6:
                    s['herds'][p] -= n
                else:
                    h[i] -= n
                s['market'].extend([KINDS[i]] * n)
        elif typ == 'sell':
            g, n = action['good'], action['count']
            h[GOODS.index(g)] -= n
            s['discard'].extend([g] * n)
            pile = s['goods_tokens'][g]
            won = pile[:n]
            del pile[:n]
            s['goods_won'][p].extend([dict(good=g, value=v) for v in won])
            event['goods_values'] = won
            if n >= 3:
                key = str(n) if n < 5 else '5_or_more'
                if s['bonus_tokens'][key]:
                    value = s['bonus_tokens'][key].pop()
                    s['bonus_won'][p].append(dict(kind=key, value=value))
                    event['bonus_kind'] = key  # Opponent never sees its value.
        if refill:
            while len(s['market']) < 5 and s['deck']:
                s['market'].append(s['deck'].pop())
        event['market_after'] = list(s['market'])
        s['history'].append(event)
        if sum(not x for x in s['goods_tokens'].values()) >= 3:
            self._end_round('three_goods_depleted')
        elif refill and len(s['market']) < 5:
            self._end_round('market_cannot_refill')
        else:
            s['current_player'] = 1 - p

    def _round_points(self):
        s = self.state
        return [sum(t['value'] for t in s['goods_won'][p] + s['bonus_won'][p])
                + s['camel_award'][p] for p in (0, 1)]

    def _end_round(self, reason):
        s = self.state
        s['camel_award'] = [5 if s['herds'][p] > s['herds'][1-p] else 0 for p in (0, 1)]
        points = self._round_points()
        criteria = [(points[p], len(s['bonus_won'][p]), len(s['goods_won'][p])) for p in (0, 1)]
        winner = None if criteria[0] == criteria[1] else max((0, 1), key=lambda p: criteria[p])
        s['rounds'].append(dict(round=s['round_number'], scores=points, winner=winner,
                                reason=reason, criteria=[list(c) for c in criteria],
                                herds=list(s['herds'])))
        if winner is None:
            s['blocked_reason'] = 'unresolved_round_tie'
            return
        s['seals'][winner] += 1
        if s['seals'][winner] == 2:
            self.finish([winner], 'two_seals')
        else:
            self._new_round(1 - winner)

    def _observe(self, player):
        s, other = self.state, 1 - player
        return dict(seals=list(s['seals']), round_number=s['round_number'],
                    market=list(s['market']), deck_count=len(s['deck']),
                    goods_tokens=deepcopy(s['goods_tokens']),
                    bonus_counts={k: len(v) for k, v in s['bonus_tokens'].items()},
                    hand=list(s['hands'][player]), herd=s['herds'][player],
                    goods_won=deepcopy(s['goods_won'][player]),
                    bonus_won=deepcopy(s['bonus_won'][player]),
                    opponent=dict(hand_size=sum(s['hands'][other]),
                                  goods_won=deepcopy(s['goods_won'][other]),
                                  bonus_kinds=[t['kind'] for t in s['bonus_won'][other]]),
                    discard=list(s['discard']), history=deepcopy(s['history']),
                    rounds=deepcopy(s['rounds']), blocked_reason=s['blocked_reason'])

    def scores(self):
        # Official match score is seals, not a sum of rupees from separate rounds.
        return list(self.state['seals'])

    def score_breakdown(self):
        points = self._round_points()
        return [dict(total=self.state['seals'][p], current_round_rupees=points[p],
                     round_scores=[r['scores'][p] for r in self.state['rounds']]) for p in (0, 1)]

    def outcome(self):
        out = super().outcome()
        if self.state['blocked_reason']:
            out.update(unresolved=True, reason=self.state['blocked_reason'])
        return out

    def validate(self):
        super().validate()
        s = self.state
        counts = Counter(s['deck'] + s['market'] + s['discard'])
        for p in (0, 1):
            assert all(type(n) is int and n >= 0 for n in s['hands'][p])
            assert sum(s['hands'][p]) <= 7
            counts.update({g: n for g, n in zip(GOODS, s['hands'][p])})
        counts['camel'] += sum(s['herds'])
        assert counts == Counter(DATA['cards'])
        for g in GOODS:
            tokens = s['goods_tokens'][g] + [t['value'] for w in s['goods_won'] for t in w if t['good'] == g]
            assert Counter(tokens) == Counter(DATA['goods_tokens_descending'][g])
        for k, values in DATA['bonus_token_multisets'].items():
            tokens = s['bonus_tokens'][k] + [t['value'] for w in s['bonus_won'] for t in w if t['kind'] == k]
            assert Counter(tokens) == Counter(values)
        assert len(s['market']) == 5 or s['terminated'] or s['blocked_reason']
        assert all(0 <= n <= 2 for n in s['seals'])

    @staticmethod
    def features(obs, action):
        h = list(obs['hand'])
        typ = action['type']
        def hand_value(hand):
            value = 0.
            for i, n in enumerate(hand):
                pile = obs['goods_tokens'][GOODS[i]]
                value += sum(pile[:n])
                if n >= 3 and obs['bonus_counts']['3' if n == 3 else '4' if n == 4 else '5_or_more']:
                    value += 2 if n == 3 else 5 if n == 4 else 9
                elif n and i < 3 and n == 1:
                    value -= 2.5
            return value
        before = hand_value(h)
        immediate, bonus, camel_gain = 0., 0., 0.
        if typ == 'sell':
            i, n = GOODS.index(action['good']), action['count']
            immediate = sum(obs['goods_tokens'][action['good']][:n])
            if n >= 3:
                k = str(n) if n < 5 else '5_or_more'
                bonus = (2 if n == 3 else 5 if n == 4 else 9) if obs['bonus_counts'][k] else 0
            h[i] -= n
        elif typ == 'take':
            h[GOODS.index(action['good'])] += 1
        elif typ == 'exchange':
            h = [n + t - g for n, t, g in zip(h, action['take'], action['give'])]
            camel_gain = -action['give'][6]
        else:
            camel_gain = obs['market'].count('camel')
        delta = hand_value(h) - before
        pressure = (7 - min(obs['deck_count'], 7)) / 7
        score = immediate + bonus + .68 * delta + .55 * camel_gain
        if typ == 'sell':
            score += pressure * (immediate + bonus) * .3 + (sum(obs['hand']) >= 6) * 1.5
        f = [score, immediate, bonus, delta, camel_gain, sum(h), pressure,
             float(typ == 'sell'), float(typ == 'take'), float(typ == 'exchange'), float(typ == 'camels')]
        return f + [0.] * (24 - len(f))


GAME_CLASS = Jaipur
