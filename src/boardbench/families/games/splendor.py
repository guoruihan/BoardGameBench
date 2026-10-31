"""Classic Splendor, original two-player setup and all 90 development cards."""
from collections import Counter
from copy import deepcopy
from itertools import combinations, product
import json
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / 'data' / 'splendor.json').read_text())
CARDS = {c['id']: c for c in DATA['cards']}
NOBLES = {n['id']: n for n in DATA['nobles']}


def bounded_vectors(limits, total):
    if len(limits) == 1:
        if 0 <= total <= limits[0]:
            yield [total]
        return
    for n in range(max(0, total - sum(limits[1:])), min(limits[0], total) + 1):
        for tail in bounded_vectors(limits[1:], total - n):
            yield [n] + tail


def payments(cost, bonuses, tokens):
    need = [max(0, c - b) for c, b in zip(cost, bonuses)]
    if sum(max(0, n - t) for n, t in zip(need, tokens)) > tokens[5]:
        return
    for ordinary in product(*(range(max(0, n - tokens[5]), min(n, tokens[i]) + 1) for i, n in enumerate(need))):
        gold = sum(need) - sum(ordinary)
        if gold <= tokens[5]:
            yield list(ordinary) + [gold]


class Splendor(DuoGame):
    game_id = 'splendor'
    rules_version = '2'
    config_id = 'classic_2014_duo'
    coverage_status = 'ready_with_rulings'
    known_unresolved_edges = ('mutual_forced_pass_has_no_verified_terminal_ruling',)

    def _reset(self, first_player):
        decks, market = [], []
        for level in (1, 2, 3):
            deck = [c['id'] for c in DATA['cards'] if c['level'] == level]
            self.rng.shuffle(deck)
            market.append([deck.pop() for _ in range(4)])
            decks.append(deck)
        nobles = list(NOBLES)
        self.rng.shuffle(nobles)
        self.state.update(decks=decks, market=market, nobles=nobles[:3],
                          removed_nobles=nobles[3:], bank=[4, 4, 4, 4, 4, 5],
                          tokens=[[0] * 6 for _ in range(2)], bought=[[], []],
                          reserved=[[], []], revealed_reserved=[[], []], acquired_nobles=[[], []],
                          phase='main', final_round=False, turns=[0, 0], history=[])

    def _bonuses(self, player):
        count = Counter(CARDS[c]['bonus'] for c in self.state['bought'][player])
        return [count[i] for i in range(5)]

    def _eligible_nobles(self, player):
        bonuses = self._bonuses(player)
        return [n for n in self.state['nobles'] if all(b >= r for b, r in zip(bonuses, NOBLES[n]['required']))]

    def _legal_actions(self):
        s, p = self.state, self.current_player
        if s['phase'] == 'return':
            return [dict(type='return', tokens=v) for v in bounded_vectors(s['tokens'][p], sum(s['tokens'][p]) - 10)]
        if s['phase'] == 'noble':
            return [dict(type='noble', noble=n) for n in self._eligible_nobles(p)]
        actions = []
        colors = [i for i, n in enumerate(s['bank'][:5]) if n]
        if colors:
            actions.extend(dict(type='take_distinct', colors=list(cs)) for cs in combinations(colors, min(3, len(colors))))
        actions.extend(dict(type='take_same', color=i) for i in colors if s['bank'][i] >= 4)
        if len(s['reserved'][p]) < 3:
            actions.extend(dict(type='reserve', source='market', card=c) for row in s['market'] for c in row)
            actions.extend(dict(type='reserve', source='deck', level=i + 1) for i, d in enumerate(s['decks']) if d)
        bonuses = self._bonuses(p)
        for source, cards in [('market', [c for row in s['market'] for c in row]), ('reserved', s['reserved'][p])]:
            for c in cards:
                actions.extend(dict(type='buy', source=source, card=c, pay=pay)
                               for pay in payments(CARDS[c]['cost'], bonuses, s['tokens'][p]))
        # Asmodee's attributed ruling permits passing only when every normal
        # action is impossible, with the reserved hand revealed as proof.
        # See the evidence grades and original article link in the game docs.
        return actions or [dict(type='pass')]

    def _remove_market(self, card):
        level = CARDS[card]['level'] - 1
        row = self.state['market'][level]
        pos = row.index(card)
        if self.state['decks'][level]:
            row[pos] = self.state['decks'][level].pop()
        else:
            row.pop(pos)

    def _step(self, action):
        s, p = self.state, self.current_player
        typ = action['type']
        event = dict(player=p, action=deepcopy(action))
        if typ == 'return':
            for i, n in enumerate(action['tokens']):
                s['tokens'][p][i] -= n
                s['bank'][i] += n
            self._after_main()
        elif typ == 'noble':
            n = action['noble']
            s['nobles'].remove(n)
            s['acquired_nobles'][p].append(n)
            self._end_turn()
        else:
            if typ == 'pass':
                s['revealed_reserved'][p] = list(s['reserved'][p])
                event['revealed_reserved'] = list(s['reserved'][p])
            elif typ in ('take_distinct', 'take_same'):
                taken = [0] * 6
                if typ == 'take_distinct':
                    for i in action['colors']:
                        taken[i] = 1
                else:
                    taken[action['color']] = 2
                for i, n in enumerate(taken):
                    s['tokens'][p][i] += n
                    s['bank'][i] -= n
            elif typ == 'reserve':
                if action['source'] == 'deck':
                    c = s['decks'][action['level'] - 1].pop()
                else:
                    c = action['card']
                    self._remove_market(c)
                s['reserved'][p].append(c)
                if s['bank'][5]:
                    s['bank'][5] -= 1
                    s['tokens'][p][5] += 1
            elif typ == 'buy':
                c = action['card']
                if action['source'] == 'market':
                    self._remove_market(c)
                else:
                    s['reserved'][p].remove(c)
                    if c in s['revealed_reserved'][p]:
                        s['revealed_reserved'][p].remove(c)
                for i, n in enumerate(action['pay']):
                    s['tokens'][p][i] -= n
                    s['bank'][i] += n
                s['bought'][p].append(c)
            self._after_main()
        # Public observations show replacement cards, so their identities are
        # retained in public history; blind reservation never discloses its draw.
        event['market_after'] = deepcopy(s['market'])
        s['history'].append(event)

    def _after_main(self):
        s, p = self.state, self.current_player
        if sum(s['tokens'][p]) > 10:
            s['phase'] = 'return'
        elif self._eligible_nobles(p):
            s['phase'] = 'noble'
        else:
            self._end_turn()

    def _end_turn(self):
        s, p = self.state, self.current_player
        s['turns'][p] += 1
        s['phase'] = 'main'
        if self.scores()[p] >= 15:
            s['final_round'] = True
        if s['final_round'] and p != s['first_player']:
            ranks = [(self.scores()[q], -len(s['bought'][q])) for q in (0, 1)]
            self.finish([q for q in (0, 1) if ranks[q] == max(ranks)], 'equal_turns_after_15')
        else:
            s['current_player'] = 1 - p

    def scores(self):
        s = self.state
        return [sum(CARDS[c]['points'] for c in s['bought'][p]) + 3 * len(s['acquired_nobles'][p]) for p in (0, 1)]

    def _observe(self, player):
        s = self.state
        return dict(phase=s['phase'], first_player=s['first_player'],
                    market=deepcopy(s['market']), deck_counts=[len(d) for d in s['decks']],
                    nobles=list(s['nobles']), bank=list(s['bank']),
                    tokens=deepcopy(s['tokens']), bought=deepcopy(s['bought']),
                    bonuses=[self._bonuses(p) for p in (0, 1)],
                    reserved=list(s['reserved'][player]),
                    revealed_reserved=deepcopy(s['revealed_reserved']),
                    reserved_counts=[len(r) for r in s['reserved']],
                    acquired_nobles=deepcopy(s['acquired_nobles']), scores=self.scores(),
                    turns=list(s['turns']), final_round=s['final_round'],
                    history=deepcopy(s['history']))

    def validate(self):
        super().validate()
        s = self.state
        all_cards = [c for collection in ('decks', 'market', 'bought', 'reserved') for row in s[collection] for c in row]
        assert Counter(all_cards) == Counter(CARDS.keys())
        all_nobles = s['nobles'] + s['removed_nobles'] + [n for ns in s['acquired_nobles'] for n in ns]
        assert Counter(all_nobles) == Counter(NOBLES.keys())
        for i in range(6):
            assert s['bank'][i] + sum(t[i] for t in s['tokens']) == (5 if i == 5 else 4)
            assert all(type(v) is int and v >= 0 for v in [s['bank'][i]] + [t[i] for t in s['tokens']])
        for p in (0, 1):
            assert len(s['reserved'][p]) <= 3
            assert len(set(s['revealed_reserved'][p])) == len(s['revealed_reserved'][p])
            assert set(s['revealed_reserved'][p]) <= set(s['reserved'][p])
            assert sum(s['tokens'][p]) <= 10 or (s['phase'] == 'return' and self.current_player == p)
        assert s['phase'] in ('main', 'return', 'noble')
        assert all(len(row) <= 4 for row in s['market'])
        if s['phase'] == 'noble':
            assert self._eligible_nobles(self.current_player)
        if s['terminated']:
            assert s['turns'][0] == s['turns'][1]

    @staticmethod
    def features(obs, action):
        p, typ = obs['player'], action['type']
        tokens, bonuses = list(obs['tokens'][p]), list(obs['bonuses'][p])
        candidates = [c for row in obs['market'] for c in row] + obs['reserved']
        def distances(tok, bon):
            return [(c, max(0, sum(max(0, x-b-t) for x, b, t in zip(CARDS[c]['cost'], bon, tok)) - tok[5])) for c in candidates]
        def potential(tok, bon):
            values = [(2.3 + CARDS[c]['points'] * 1.5 + .4 / (1 + bon[CARDS[c]['bonus']])) / (1 + d) for c, d in distances(tok, bon)]
            return sum(sorted(values, reverse=True)[:3])
        before = potential(tokens, bonuses)
        points, engine, gold_spent, spent, noble_progress = 0., 0., 0., 0., 0.
        if typ == 'buy':
            card = CARDS[action['card']]
            points = card['points']
            engine = 1 / (1 + bonuses[card['bonus']] * .3)
            bonuses[card['bonus']] += 1
            spent = sum(action['pay'])
            gold_spent = action['pay'][5]
            tokens = [t - v for t, v in zip(tokens, action['pay'])]
            for n in obs['nobles']:
                req = NOBLES[n]['required']
                missing = sum(max(0, r-b) for r, b in zip(req, bonuses))
                if req[card['bonus']] >= bonuses[card['bonus']]:
                    noble_progress += 1.2 / (1 + missing)
                if missing == 0:
                    noble_progress += 2
            value = 2.4 * points + 2.4 * engine - .18 * spent - .16 * gold_spent + noble_progress
        elif typ == 'noble':
            value = 3.
        else:
            if typ == 'take_distinct':
                for i in action['colors']:
                    tokens[i] += 1
            elif typ == 'take_same':
                tokens[action['color']] += 2
            elif typ == 'return':
                tokens = [t - v for t, v in zip(tokens, action['tokens'])]
            elif typ == 'reserve':
                tokens[5] += int(obs['bank'][5] > 0)
            change = potential(tokens, bonuses) - before
            value = change * 1.8
            if typ == 'reserve':
                value -= 1.2 + .3 * obs['reserved_counts'][p]
                if action['source'] == 'market':
                    c = action['card']
                    distance = next(d for target, d in distances(tokens, bonuses) if target == c)
                    value += .35 * CARDS[c]['points'] / (1 + distance)
            if typ != 'return':
                value -= .7 * max(0, sum(tokens) - 10)
        f = [value, points, engine, spent, gold_spent, noble_progress,
             sum(tokens), potential(tokens, bonuses) - before,
             float(typ == 'buy'), float(typ == 'reserve'), float(typ == 'return'),
             float(typ == 'noble'), float(obs['final_round'])]
        return f + [0.] * (24 - len(f))


GAME_CLASS = Splendor
