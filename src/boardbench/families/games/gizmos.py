"""CMON Gizmos (2018): exact two-player base game, original photographed cards.

Payment reachability runs backwards through converter effects. It enumerates
physical payment vectors and used Gizmos, rather than duplicate equivalent
conversion orders. Research return order and reaction order remain player choices.
"""
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / 'data' / 'gizmos.json').read_text())
CARDS = {c['id']: c for c in DATA['cards']}


def compositions(total, size=4):
    if size == 1:
        yield (total,)
    else:
        for n in range(total + 1):
            for tail in compositions(total - n, size - 1):
                yield (n,) + tail


@lru_cache(maxsize=2048)
def payment_options(energy, converter_ids, color, cost):
    """Return every distinct (physical payment, used converter IDs) outcome.

    A reverse wildcard changes one required output into its required input. A
    reverse doubler merges two output requirements into one input. Sub-effects
    on advanced converters may interleave with other converters in this build;
    using either sub-effect consumes the whole Gizmo for the rest of the turn.
    Wildcard identity conversions and partly unused doubled output are allowed.
    """
    if cost == 0:
        return (((0, 0, 0, 0), ()),)
    units = [(cid, u['type'], u['source']) for cid in converter_ids for u in CARDS[cid]['converters']]
    needs = list(compositions(cost)) if color is None else [tuple(cost if i == color else 0 for i in range(4))]
    stack = [(need, 0) for need in needs]
    seen = set(stack)
    answers = set()
    while stack:
        need, mask = stack.pop()
        if all(n <= e for n, e in zip(need, energy)):
            used = tuple(sorted({units[j][0] for j in range(len(units)) if mask & (1 << j)}))
            answers.add((need, used))
        for j, (_, kind, source) in enumerate(units):
            if mask & (1 << j):
                continue
            next_needs = []
            if kind == 'double':
                if need[source]:
                    n = list(need)
                    n[source] -= int(n[source] >= 2)
                    next_needs.append(tuple(n))
            else:
                for out_color, count in enumerate(need):
                    if count:
                        for in_color in range(4) if source is None else (source,):
                            n = list(need); n[out_color] -= 1; n[in_color] += 1
                            next_needs.append(tuple(n))
            for n in next_needs:
                key = (n, mask | (1 << j))
                if key not in seen:
                    seen.add(key); stack.append(key)
    return tuple(sorted(answers))


class Gizmos(DuoGame):
    game_id = 'gizmos'
    config_id = 'cmon_2018_duo'
    rules_version = '2'
    coverage_status = 'ready_with_rulings'
    known_unresolved_edges = ('all_players_blocked_no_stalemate_ruling',)

    def _reset(self, first_player):
        decks, market, removed = [], [], ['g0-2', 'g0-3']
        for level, shown in [(1, 4), (2, 3), (3, 2)]:
            deck = [c['id'] for c in DATA['cards'] if c['level'] == level]
            self.rng.shuffle(deck)
            if level == 3:
                removed.extend(deck[16:]); deck = deck[:16]
            market.append([deck.pop() for _ in range(shown)])
            decks.append(deck)
        self.state.update(decks=decks, market=market, removed=removed,
                          built=[['g0-0'], ['g0-1']], archives=[[], []],
                          energy=[[0]*4, [0]*4], internal=[13]*4, energy_row=[],
                          vp=[0, 0], phase='main', used=[], pending=[], tasks=[],
                          task_counter=0, research=None, final_round=False,
                          turns=[0, 0], history=[])
        self._refill_energy()

    def _limits(self, player):
        limits = dict(storage=5, file=1, research=3, no_file=False, no_research=False,
                      discount_archive=0, discount_research=0, discount_level2=0)
        for cid in self.state['built'][player]:
            c = CARDS[cid]
            if c['kind'] == 'upgrade':
                for key in limits:
                    if key in c:
                        limits[key] = (limits[key] or c[key]) if key.startswith('no_') else limits[key] + c[key]
        return limits

    def _draw_internal(self):
        total = sum(self.state['internal'])
        if not total:
            return None
        number = self.rng.randrange(total)
        for color, n in enumerate(self.state['internal']):
            if number < n:
                self.state['internal'][color] -= 1
                return color
            number -= n
        raise AssertionError('energy draw')

    def _refill_energy(self):
        while len(self.state['energy_row']) < 6:
            color = self._draw_internal()
            if color is None:
                break
            self.state['energy_row'].append(color)

    def _discount(self, card, source, player):
        limits = self._limits(player)
        return (limits.get('discount_' + source, 0)
                + (limits['discount_level2'] if CARDS[card]['level'] == 2 else 0))

    def _payments(self, card, source, free=False):
        p = self.current_player
        c = CARDS[card]
        converters = tuple(sorted(cid for cid in self.state['built'][p]
                                  if CARDS[cid]['kind'] == 'converter' and cid not in self.state['used']))
        discounts = [c['cost']] if free else range(min(c['cost'], self._discount(card, source, p)) + 1)
        for discount in discounts:
            for pay, used in payment_options(tuple(self.state['energy'][p]), converters, c['color'], c['cost'] - discount):
                yield dict(type='build', source=source, card=card, pay=list(pay),
                           converters=list(used), discount=discount)

    def _base_actions(self, allowed=None, free=False):
        s, p = self.state, self.current_player
        limits = self._limits(p)
        actions = []
        if allowed in (None, 'pick') and sum(s['energy'][p]) < limits['storage']:
            actions.extend(dict(type='pick', color=c) for c in sorted(set(s['energy_row'])))
        if allowed in (None, 'file') and not limits['no_file'] and len(s['archives'][p]) < limits['file']:
            actions.extend(dict(type='file', source='market', card=c) for row in s['market'] for c in row)
        if allowed in (None, 'free_build'):
            for source, cards in [('market', [c for row in s['market'] for c in row]), ('archive', s['archives'][p])]:
                for c in cards:
                    if not free or CARDS[c]['level'] == 1:
                        actions.extend(self._payments(c, source, free))
        if allowed in (None, 'research') and not limits['no_research']:
            actions.extend(dict(type='research', level=i+1) for i, d in enumerate(s['decks']) if d)
        return actions

    def _legal_actions(self):
        s, p = self.state, self.current_player
        if s['phase'] == 'main':
            # The licensed BGA adaptation added "pass if no legal actions"
            # (Fletcheese/gizmos bc8e07a, 2023-05-20). Its UI does not test
            # Build affordability; here a pass requires all four to be absent.
            # This explicit digital ruling is documented, not a publisher FAQ.
            return self._base_actions() or [dict(type='forced_pass')]
        if s['phase'] == 'research_choose':
            actions = [dict(type='research_pass')]
            limits = self._limits(p)
            for c in s['research']['cards']:
                if not limits['no_file'] and len(s['archives'][p]) < limits['file']:
                    actions.append(dict(type='file', source='research', card=c))
                actions.extend(self._payments(c, 'research'))
            return actions
        if s['phase'] == 'research_return':
            return [dict(type='return_research', card=c) for c in s['research']['cards']]
        actions = [dict(type='end_turn')]
        actions.extend(dict(type='activate', card=c) for c in s['pending'])
        for task in s['tasks']:
            for a in self._base_actions(task['kind'], task['kind'] == 'free_build'):
                a['bonus'] = task['id']; actions.append(a)
        return actions

    def _remove_card(self, card, source):
        s, p = self.state, self.current_player
        if source == 'archive':
            s['archives'][p].remove(card)
        elif source == 'research':
            s['research']['cards'].remove(card)
        else:
            level = CARDS[card]['level'] - 1
            row = s['market'][level]; index = row.index(card)
            if s['decks'][level]:
                row[index] = s['decks'][level].pop()
            else:
                row.pop(index)

    def _event(self, kind, eligible, color=None, card=None, source=None):
        s = self.state
        for cid in eligible:
            c = CARDS[cid]
            if c['kind'] != 'trigger' or c['trigger'] != kind or cid in s['used'] or cid in s['pending']:
                continue
            if 'colors' in c and color is not None and color not in c['colors']:
                continue
            if kind == 'build':
                if 'source' in c and c['source'] != source:
                    continue
                if 'level_filter' in c and c['level_filter'] != CARDS[card]['level']:
                    continue
            s['pending'].append(cid)

    def _start_research_return(self):
        if self.state['research']['cards']:
            self.state['phase'] = 'research_return'
        else:
            self.state['research'] = None
            self.state['phase'] = 'effects'

    def _perform(self, action):
        s, p = self.state, self.current_player
        typ = action['type']; eligible = list(s['built'][p])
        if typ == 'research':
            level = action['level'] - 1
            amount = min(self._limits(p)['research'], len(s['decks'][level]))
            s['research'] = dict(level=level, cards=[s['decks'][level].pop() for _ in range(amount)])
            s['phase'] = 'research_choose'
            return
        from_research = action.get('source') == 'research'
        if typ == 'pick':
            color = action['color']
            s['energy_row'].remove(color); s['energy'][p][color] += 1
            self._refill_energy()
            self._event('pick', eligible, color=color)
        elif typ == 'file':
            self._remove_card(action['card'], action['source'])
            s['archives'][p].append(action['card'])
            self._event('file', eligible)
        elif typ == 'build':
            card, source = action['card'], action['source']
            self._remove_card(card, source)
            for i, n in enumerate(action['pay']):
                s['energy'][p][i] -= n; s['internal'][i] += n
            self._refill_energy()
            s['used'].extend(action['converters'])
            s['built'][p].append(card)
            self._event('build', eligible, color=CARDS[card]['color'], card=card, source=source)
            if len(s['built'][p]) >= 16 or sum(CARDS[c]['level'] == 3 for c in s['built'][p]) >= 4:
                s['final_round'] = True
        if from_research:
            self._start_research_return()
        else:
            s['phase'] = 'effects'

    def _activate(self, cid):
        s, p = self.state, self.current_player
        s['pending'].remove(cid); s['used'].append(cid)
        c = CARDS[cid]
        if c['effect'] == 'vp':
            s['vp'][p] += c['amount']
        elif c['effect'] == 'random':
            for _ in range(c['amount']):
                if sum(s['energy'][p]) >= self._limits(p)['storage']:
                    break
                color = self._draw_internal()
                if color is None:
                    break
                s['energy'][p][color] += 1
        else:
            s['task_counter'] += 1
            s['tasks'].append(dict(id=s['task_counter'], kind=c['effect'], remaining=c['amount']))

    def _step(self, action):
        s, p = self.state, self.current_player
        typ = action['type']
        public = deepcopy(action)
        if typ in ('end_turn', 'forced_pass'):
            self._end_turn()
        elif typ == 'activate':
            self._activate(action['card'])
        elif typ == 'research_pass':
            self._start_research_return()
        elif typ == 'return_research':
            s['research']['cards'].remove(action['card'])
            s['decks'][s['research']['level']].insert(0, action['card'])
            public = dict(type='return_research')  # Hidden bottom-card identity.
            self._start_research_return()
        else:
            if 'bonus' in action:
                task = next(t for t in s['tasks'] if t['id'] == action['bonus'])
                task['remaining'] -= 1
                if task['remaining'] == 0:
                    s['tasks'].remove(task)
            self._perform(action)
        if s['phase'] == 'effects' and not s['pending'] and not s['tasks']:
            self._end_turn()
        s['history'].append(dict(player=p, action=public, market_after=deepcopy(s['market']),
                                 energy_row_after=list(s['energy_row'])))

    def _end_turn(self):
        s, p = self.state, self.current_player
        s['turns'][p] += 1
        s.update(phase='main', used=[], pending=[], tasks=[], research=None)
        if s['final_round'] and p != s['first_player']:
            scores = self.scores()
            ranks = [(scores[q], len(s['built'][q]), sum(s['energy'][q]), int(q != s['first_player'])) for q in (0, 1)]
            self.finish([max((0, 1), key=lambda q: ranks[q])], 'equal_turns_after_build_threshold')
        else:
            s['current_player'] = 1-p

    def scores(self):
        s = self.state
        return [sum((sum(s['energy'][p]) if CARDS[c].get('award') == 'energy'
                     else s['vp'][p] if CARDS[c].get('award') == 'vp'
                     else CARDS[c]['points']) for c in s['built'][p]) + s['vp'][p] for p in (0, 1)]

    def score_breakdown(self):
        s = self.state
        return [dict(total=self.scores()[p], printed=sum(CARDS[c]['points'] for c in s['built'][p]),
                     tokens=s['vp'][p], energy_awards=sum(CARDS[c].get('award') == 'energy' for c in s['built'][p])*sum(s['energy'][p]),
                     token_awards=sum(CARDS[c].get('award') == 'vp' for c in s['built'][p])*s['vp'][p]) for p in (0, 1)]

    def _observe(self, player):
        s = self.state
        research = None
        if s['research']:
            research = dict(level=s['research']['level'] + 1, count=len(s['research']['cards']))
            if player == self.current_player:
                research['cards'] = list(s['research']['cards'])
        return dict(phase=s['phase'], first_player=s['first_player'], market=deepcopy(s['market']),
                    deck_counts=[len(d) for d in s['decks']], built=deepcopy(s['built']),
                    archives=deepcopy(s['archives']), energy=deepcopy(s['energy']),
                    energy_row=list(s['energy_row']), internal_count=sum(s['internal']),
                    vp=list(s['vp']), scores=self.scores(), limits=[self._limits(p) for p in (0, 1)],
                    used=list(s['used']), pending=list(s['pending']), tasks=deepcopy(s['tasks']),
                    research=research, turns=list(s['turns']), final_round=s['final_round'],
                    history=deepcopy(s['history']))

    def validate(self):
        super().validate()
        s = self.state
        cards = s['removed'] + [c for key in ('decks', 'market', 'built', 'archives') for row in s[key] for c in row]
        if s['research']:
            cards += s['research']['cards']
        assert Counter(cards) == Counter(CARDS.keys())
        for i in range(4):
            assert s['internal'][i] + s['energy_row'].count(i) + sum(e[i] for e in s['energy']) == 13
        assert len(s['energy_row']) <= 6
        assert len(s['used']) == len(set(s['used'])) and len(s['pending']) == len(set(s['pending']))
        assert not set(s['used']) & set(s['pending'])
        for p in (0, 1):
            assert all(type(n) is int and n >= 0 for n in s['energy'][p])
            assert sum(s['energy'][p]) <= self._limits(p)['storage']
            assert len(s['archives'][p]) <= self._limits(p)['file']
        assert s['phase'] in ('main', 'effects', 'research_choose', 'research_return')
        assert (s['research'] is not None) == s['phase'].startswith('research_')
        if s['terminated']:
            assert s['turns'][0] == s['turns'][1]

    @staticmethod
    def features(obs, action):
        p, typ = obs['player'], action['type']
        energy = obs['energy'][p]; built = [CARDS[c] for c in obs['built'][p]]
        room = obs['limits'][p]['storage'] - sum(energy)
        def card_value(c):
            engine = 0.
            if c['kind'] == 'trigger':
                engine = {'pick':2.7, 'random':2.5, 'vp':1.2, 'file':1.1, 'research':2.5, 'free_build':3.1}[c['effect']]
                engine *= c.get('amount', 1)
                if c['trigger'] == 'file' or c.get('source') == 'archive': engine *= .65
                if len(c.get('colors', [])) == 2: engine *= 1.3
            elif c['kind'] == 'converter':
                engine = 1.4 * len(c['converters'])
            elif c['kind'] == 'upgrade':
                engine = .5*c.get('storage',0) + .3*c.get('file',0) + .2*c.get('research',0)
                engine += .6*(c.get('discount_archive',0)+c.get('discount_research',0)+c.get('discount_level2',0))
                if c.get('no_file'):engine-=.7
                if c.get('no_research'):engine-=.7
            points = sum(energy) if c.get('award')=='energy' else obs['vp'][p] if c.get('award')=='vp' else c['points']
            return points, engine
        points, engine, spent, gain = 0., 0., 0., 0.
        if typ == 'build':
            c=CARDS[action['card']];points,engine=card_value(c);spent=sum(action['pay'])
            value=5+points+engine-.45*spent-.12*len(action['converters'])
        elif typ == 'pick':
            color=action['color']
            potentials=[]
            for cid in [c for row in obs['market'] for c in row]+obs['archives'][p]:
                c=CARDS[cid]
                if c['color'] in (color,None):
                    missing=c['cost']-(sum(energy) if c['color'] is None else energy[color])
                    if missing>0:potentials.append(sum(card_value(c))/(missing+1))
            gain=sum(c.get('amount',1) for c in built if c['kind']=='trigger' and c['trigger']=='pick' and color in c.get('colors',[]))
            value=1+max(potentials,default=.1)+.8*min(gain,max(0,room-1))
        elif typ == 'file':
            c=CARDS[action['card']];points,engine=card_value(c)
            missing=max(0,c['cost']-(sum(energy) if c['color'] is None else energy[c['color']]))
            gain=sum(c.get('amount',1) for c in built if c['kind']=='trigger' and c['trigger']=='file')
            value=(points+engine)/(missing+2)+.5*min(room,gain)-.2*len(obs['archives'][p])
        elif typ == 'activate':
            c=CARDS[action['card']]
            gain=c['amount']
            if c['effect'] in ('random','pick'):gain=min(gain,room)
            value=2+gain
        elif typ == 'research':
            value=.35+.05*obs['limits'][p]['research']
        elif typ == 'research_pass':
            value=-1.
        elif typ == 'return_research':
            value=-sum(card_value(CARDS[action['card']]))*.01
        else:
            value=-2.
        f=[value,points,engine,spent,gain,room,sum(energy),len(built),obs['scores'][p],
           obs['scores'][1-p],float(typ=='build'),float(typ=='pick'),float(typ=='file'),
           float(typ=='activate'),float(typ=='research'),float(obs['final_round'])]
        return f+[0.]*(24-len(f))


GAME_CLASS = Gizmos
