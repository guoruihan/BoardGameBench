"""Five Tribes base Fakir edition, four turn markers / two players.

Sowing is committed sequentially with exact completion feasibility. No strategic
candidate pruning and no ability can interrupt an unfinished sow. See the game
coverage document for source-tiered interpretations and unresolved boundaries.
"""
from collections import Counter
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / 'data/five_tribes_base.json').read_text())
COLORS = tuple(DATA['meeples'])
GOODS = tuple(c for c in DATA['resources'] if c != 'fakir')
VP = DATA['djinn_vp']
SALE = DATA['sale_values']
BIDS = (0, 1, 3, 5, 8, 12, 18)
PAYMENTS = {'elder': (1, 0), 'fakir': (0, 1), 'two_elders': (2, 0), 'elder_fakir': (1, 1)}
SINGLE_COST = ('anun_nak', 'bouraq', 'enki', 'hagis', 'iblis', 'lamia')
DOUBLE_COST = ('echidna', 'sibittis', 'utug', 'leta')
ACTIVE_ANY = ('anun_nak', 'bouraq', 'enki', 'iblis', 'echidna', 'sibittis', 'sloar', 'utug', 'leta')


def adjacent(tile, diagonal=False):
    r, c = divmod(tile, 6)
    return [rr*6+cc for rr in range(max(0, r-1), min(5, r+2))
            for cc in range(max(0, c-1), min(6, c+2))
            if (rr != r or cc != c) and (diagonal or abs(rr-r)+abs(cc-c) == 1)]


def board_counts(board):
    return [list(t['meeples']) for t in board]


def can_finish(counts, hand, position, previous):
    """Exact existential legality, not a rollout/score-based candidate filter."""
    remaining = sum(hand)
    for dest in adjacent(position):
        if dest == previous:
            continue
        for color, n in enumerate(hand):
            if not n:
                continue
            if remaining == 1:
                if counts[dest][color]:
                    return True
                continue
            hand[color] -= 1
            counts[dest][color] += 1
            legal = can_finish(counts, hand, dest, position)
            counts[dest][color] -= 1
            hand[color] += 1
            if legal:
                return True
    return False


def legal_starts(board):
    counts = board_counts(board)
    result = []
    for start, pile in enumerate(counts):
        if not sum(pile):
            continue
        hand = pile.copy()
        counts[start] = [0]*5
        legal = can_finish(counts, hand, start, -1)
        counts[start] = pile
        if legal:
            result.append(start)
    return result


def blue_neighborhood(board, tile):
    return sum(board[i]['color'] == 'blue' for i in [tile]+adjacent(tile, True))


@lru_cache(maxsize=4096)
def goods_score(counts, wild=0):
    counts = tuple(sorted(counts))
    if wild:
        best = 0
        # Symmetric counts are interchangeable: exact DP over possible wild types.
        for value in set(counts):
            updated = list(counts)
            updated[updated.index(value)] += 1
            best = max(best, goods_score(tuple(sorted(updated)), wild-1))
        return best
    return sum(SALE[sum(c >= level for c in counts)] for level in range(1, max(counts, default=0)+1))


def empty_tile(tile):
    return tile['owner'] is None and not sum(tile['meeples']) and not tile['trees'] and not tile['palaces']


class FiveTribes(DuoGame):
    coverage_status = 'research_only'
    known_unresolved_edges = ('no_affordable_bid_and_zero_slots_full',
                              'anun_nak_bag_fewer_than_three_requires_ruling')
    game_id = 'five_tribes'
    config_id = 'base_fakir_duo_four_markers'
    rules_version = '2'

    def _reset(self, first_player):
        board = [dict(kind=kind, color=color, points=points, meeples=[0]*5,
                      owner=None, trees=0, palaces=0)
                 for kind, color, points, n in DATA['tiles'] for _ in range(n)]
        self.rng.shuffle(board)
        bag = [i for i, c in enumerate(COLORS) for _ in range(DATA['meeples'][c])]
        self.rng.shuffle(bag)
        for tile in board:
            for _ in range(3):
                tile['meeples'][bag.pop()] += 1
        resources = [c for c, n in DATA['resources'].items() for _ in range(n)]
        djinn = list(VP)
        self.rng.shuffle(resources)
        self.rng.shuffle(djinn)
        first_marker = self.rng.choice([2*first_player, 2*first_player+1])
        markers = [m for m in range(4) if m != first_marker]
        self.rng.shuffle(markers)
        self.state.update(board=board, bag=bag, resource_deck=resources,
            resource_discard=[], resource_row=[], djinn_deck=djinn, djinn_discard=[], djinn_row=[],
            players=[dict(coins=50, camels=11, yellow=0, white=0, resources=[], djinn=[]) for _ in range(2)],
            tree_supply=12, palace_supply=10, round=1, phase='bid', bid_queue=[first_marker]+markers,
            bids=[], turn_queue=[], next_bid_queue=[], current_marker=None, history=[], used=[],
            end_trigger=False, blocked_reason=None, sow_hand=[0]*5, sow_position=None,
            sow_previous=-1, sow_start=None, endpoint=None, tribe_count=0, tribe_color=None,
            placement=None, djinn_offer=[], resume_phase=None)
        self._refill()
        self.state['current_player'] = first_player

    def _draw_card(self, kind):
        s = self.state
        deck, discard = kind+'_deck', kind+'_discard'
        if not s[deck] and s[discard]:
            s[deck] = s[discard]
            s[discard] = []
            self.rng.shuffle(s[deck])
        return s[deck].pop() if s[deck] else None

    def _refill(self):
        for kind, size in (('resource', 9), ('djinn', 3)):
            while len(self.state[kind+'_row']) < size:
                card = self._draw_card(kind)
                if card is None:
                    break
                self.state[kind+'_row'].append(card)

    def _event(self, event, **details):
        self.state['history'].append(dict(event=event, player=self.state['current_player'], **details))

    def _observe(self, player):
        s = self.state
        views = []
        for seat, p in enumerate(s['players']):
            view = {k: p[k] for k in ('camels', 'yellow', 'white', 'djinn')}
            view['resource_count'] = len(p['resources'])
            if seat == player or s['terminated']:
                view.update(coins=p['coins'], resources=p['resources'])
            views.append(view)
        obs = {k: s[k] for k in ('board', 'resource_row', 'resource_discard', 'djinn_row',
            'djinn_discard', 'tree_supply', 'palace_supply', 'round', 'phase', 'bid_queue',
            'bids', 'turn_queue', 'next_bid_queue', 'current_marker', 'history', 'used',
            'end_trigger', 'blocked_reason', 'sow_hand', 'sow_position', 'sow_previous',
            'sow_start', 'endpoint', 'tribe_count', 'tribe_color', 'placement')}
        obs['players'] = views
        obs['bag_count'] = len(s['bag'])
        if s['phase'] == 'djinn_choice' and player == s['current_player']:
            obs['djinn_offer'] = s['djinn_offer']
        return obs

    def _costs(self, djinn):
        p = self.state['players'][self.state['current_player']]
        options = ('elder', 'fakir') if djinn in SINGLE_COST else (
            ('fakir',) if djinn == 'sloar' else ('two_elders', 'elder_fakir'))
        return [key for key in options if p['white'] >= PAYMENTS[key][0]
                and p['resources'].count('fakir') >= PAYMENTS[key][1]]

    def _pay(self, payment):
        s = self.state
        p = s['players'][s['current_player']]
        elder, fakir = PAYMENTS[payment]
        p['white'] -= elder
        s['bag'].extend([COLORS.index('white')]*elder)
        self._spend_fakirs(fakir)
        self._event('pay_ability', payment=payment)

    def _spend_fakirs(self, n):
        p = self.state['players'][self.state['current_player']]
        for _ in range(n):
            p['resources'].remove('fakir')
            self.state['resource_discard'].append('fakir')

    def _active_actions(self):
        s = self.state
        p = s['players'][s['current_player']]
        result = []
        for djinn in ACTIVE_ANY:
            if djinn not in p['djinn'] or djinn in s['used']:
                continue
            if djinn in ('anun_nak', 'leta'):
                targets = [i for i, t in enumerate(s['board']) if empty_tile(t)]
            elif djinn == 'utug':
                targets = [i for i, t in enumerate(s['board']) if t['owner'] is None
                           and sum(t['meeples']) and not t['trees'] and not t['palaces']]
            else:
                targets = [None]
            if djinn in ('leta', 'utug') and not p['camels']:
                targets = []
            for payment in self._costs(djinn):
                result.extend(dict(type='activate', djinn=djinn, payment=payment, target=target) for target in targets)
        return result

    def _legal_actions(self):
        s = self.state
        p = s['players'][s['current_player']]
        phase = s['phase']
        if phase == 'unresolved':
            return []
        if phase == 'bid':
            zero_count = sum(b['slot'] == 0 for b in s['bids'])
            occupied = {b['slot'] for b in s['bids'] if b['slot']}
            result = []
            for index, slot in enumerate(BIDS):
                if slot in occupied or slot == 0 and zero_count >= 3:
                    continue
                max_fakir = p['resources'].count('fakir') if 'kumarbi' in p['djinn'] else 0
                for fakirs in range(max_fakir+1):
                    cost = BIDS[max(0, index-fakirs)]
                    if cost <= p['coins']:
                        result.append(dict(type='bid', slot=slot, fakirs=fakirs))
            if not result and zero_count == 3 and p['coins'] == 0:
                # Cathala's 2014-08-20 ruling permits goods sales when the
                # fourth marker cannot pay 1. It does not grant free credit.
                available = sorted(set(p['resources'])-{'fakir'})
                result = [dict(type='emergency_sell', goods=list(group))
                          for n in range(1, len(available)+1)
                          for group in combinations(available, n)]
            return result
        if phase == 'sow':
            result = []
            counts = board_counts(s['board'])
            hand = s['sow_hand'][:]
            for dest in adjacent(s['sow_position']):
                if dest == s['sow_previous']:
                    continue
                for color, n in enumerate(hand):
                    if not n:
                        continue
                    if sum(hand) == 1:
                        legal = counts[dest][color] > 0
                    else:
                        hand[color] -= 1
                        counts[dest][color] += 1
                        legal = can_finish(counts, hand, dest, s['sow_position'])
                        counts[dest][color] -= 1
                        hand[color] += 1
                    if legal:
                        result.append(dict(type='drop', tile=dest, color=COLORS[color]))
            return result
        if phase == 'placement':
            pending = s['placement']
            result = [dict(type='place', origin=i, tile=i, payment=None) for i in pending['origins']]
            relocator = 'hagis' if pending['kind'] == 'palace' else 'lamia'
            if relocator in p['djinn'] and relocator not in s['used']:
                result += [dict(type='place', origin=i, tile=j, payment=pay)
                           for i in pending['origins'] for j in adjacent(i, True) for pay in self._costs(relocator)]
            return result
        if phase == 'djinn_choice':
            return [dict(type='keep_djinn', card=c) for c in s['djinn_offer']]
        result = self._active_actions()
        if phase == 'move':
            starts = legal_starts(s['board'])
            result += [dict(type='start_sow', tile=i) for i in starts]
            if not starts:
                result.append(dict(type='no_move'))
        elif phase == 'builder':
            result += [dict(type='builder', fakirs=n) for n in range(p['resources'].count('fakir')+1)]
        elif phase == 'assassin':
            max_kills = 2 if 'iblis' in s['used'] else 1
            targets = []
            r, c = divmod(s['endpoint'], 6)
            for fakirs in range(p['resources'].count('fakir')+1):
                reach = s['tribe_count'] + fakirs
                for i, tile in enumerate(s['board']):
                    rr, cc = divmod(i, 6)
                    if abs(rr-r)+abs(cc-c) > reach:
                        continue
                    colors = [COLORS[k] for k, count in enumerate(tile['meeples']) for _ in range(count)]
                    take = min(max_kills, len(colors))
                    if take:
                        for victims in sorted(set(combinations(sorted(colors), take))):
                            targets.append(dict(type='assassinate', tile=i, opponent=None, colors=list(victims), fakirs=fakirs))
                other = s['players'][1-s['current_player']]
                if 'boaz' not in other['djinn']:
                    colors = ['yellow']*other['yellow'] + ['white']*other['white']
                    take = min(max_kills, len(colors))
                    for victims in sorted(set(combinations(sorted(colors), take))) if take else []:
                        targets.append(dict(type='assassinate', tile=None, opponent=1-s['current_player'], colors=list(victims), fakirs=fakirs))
            result += targets or [dict(type='no_assassination')]
        elif phase == 'tile':
            tile = s['board'][s['endpoint']]
            kind = tile['kind']
            if kind in ('oasis', 'village'):
                result.append(dict(type='tile_effect'))
            else:
                result.append(dict(type='skip_tile'))
                if kind in ('small_market', 'large_market'):
                    price, prefix, count = (3, 3, 1) if kind == 'small_market' else (6, 6, 2)
                    available = min(prefix, len(s['resource_row']))
                    if p['coins'] >= price and available:
                        result += [dict(type='market', indices=list(indices))
                            for indices in combinations(range(available), min(count, available))]
                elif kind == 'sacred':
                    for payment in self._costs('sacred'):
                        result += [dict(type='get_djinn', card=c, payment=payment) for c in s['djinn_row']]
        elif phase == 'sale':
            available = sorted(set(p['resources'])-{'fakir'})
            result += [dict(type='sell', goods=list(group)) for n in range(1, len(available)+1)
                       for group in combinations(available, n)]
            result.append(dict(type='end_turn'))
        else:
            raise ValueError(f'invalid phase {phase}')
        return result

    def _unresolved(self, reason):
        self.state.update(phase='unresolved', blocked_reason=reason)

    def _check_no_moves(self):
        # Check only complete board operations, never an unfinished sow. Once
        # triggered, later Anun-Nak use cannot undo the final-round condition.
        if not self.state['end_trigger'] and not legal_starts(self.state['board']):
            self.state['end_trigger'] = True

    def _claim(self, tile):
        s = self.state
        p = s['players'][s['current_player']]
        if tile['owner'] is None and p['camels']:
            tile['owner'] = s['current_player']
            p['camels'] -= 1
            if not p['camels']:
                s['end_trigger'] = True

    def _notify(self, djinn):
        s = self.state
        for seat, p in enumerate(s['players']):
            if djinn in p['djinn']:
                p['coins'] += 1 if seat == s['current_player'] else 2

    def _get_djinn(self, card):
        self._notify('baal')  # Ba'al does not trigger its own acquisition.
        self.state['players'][self.state['current_player']]['djinn'].append(card)
        self._event('gain_djinn', card=card)

    def _get_resource(self, card, public=True):
        if card is not None:
            self.state['players'][self.state['current_player']]['resources'].append(card)
            self._event('gain_resource', **({'card': card} if public else {'private_draw': 1}))

    def _begin_placement(self, kind, origins, resume):
        s = self.state
        supply = 'palace_supply' if kind == 'palace' else 'tree_supply'
        if not s[supply]:
            s['phase'] = resume
            return
        s['placement'] = dict(kind=kind, origins=origins, resume=resume)
        s['phase'] = 'placement'

    def _activate(self, action):
        s = self.state
        p = s['players'][s['current_player']]
        d = action['djinn']
        old_phase = s['phase']
        if d == 'anun_nak' and len(s['bag']) + PAYMENTS[action['payment']][0] < 3:
            self._unresolved('anun_nak_bag_fewer_than_three_requires_ruling')
            return
        self._pay(action['payment'])
        s['used'].append(d)
        self._event('activate', djinn=d, target=action['target'])
        if d == 'anun_nak':
            for _ in range(3):
                color = s['bag'].pop(self.rng.randrange(len(s['bag'])))
                s['board'][action['target']]['meeples'][color] += 1
            self._check_no_moves()
        elif d in ('bouraq', 'enki'):
            kind = 'palace' if d == 'bouraq' else 'tree'
            required = 'village' if d == 'bouraq' else 'oasis'
            self._begin_placement(kind, [i for i, t in enumerate(s['board']) if t['kind'] == required], old_phase)
        elif d in ('leta', 'utug'):
            self._claim(s['board'][action['target']])
        elif d == 'sloar':
            self._get_resource(self._draw_card('resource'), public=False)
        elif d == 'sibittis':
            offers = [self._draw_card('djinn') for _ in range(3)]
            s['djinn_offer'] = [x for x in offers if x is not None]
            if s['djinn_offer']:
                s['resume_phase'] = old_phase
                s['phase'] = 'djinn_choice'

    def _finish_sow(self, color):
        s = self.state
        tile = s['board'][s['sow_position']]
        count = tile['meeples'][color]
        tile['meeples'][color] = 0
        s.update(endpoint=s['sow_position'], tribe_count=count, tribe_color=COLORS[color])
        if not sum(tile['meeples']):
            self._claim(tile)
        p = s['players'][s['current_player']]
        name = COLORS[color]
        if name in ('yellow', 'white'):
            p[name] += count
            s['phase'] = 'tile'
        else:
            s['bag'].extend([color]*count)
            if name == 'green':
                for _ in range(min(count, len(s['resource_row']))):
                    self._get_resource(s['resource_row'].pop(0))
                s['phase'] = 'tile'
            else:
                s['phase'] = 'builder' if name == 'blue' else 'assassin'
        self._event('collect_tribe', tile=s['endpoint'], color=name, count=count)
        self._check_no_moves()

    def _start_turn(self):
        s = self.state
        marker = s['turn_queue'].pop(0)
        s['next_bid_queue'].append(marker)
        s.update(current_marker=marker, current_player=marker//2, used=[], phase='move',
                 endpoint=None, tribe_count=0, tribe_color=None, sow_hand=[0]*5,
                 sow_position=None, sow_previous=-1, sow_start=None, placement=None)
        self._check_no_moves()

    def _end_turn(self):
        s = self.state
        self._check_no_moves()
        if s['turn_queue']:
            self._start_turn()
        elif s['end_trigger']:
            self.finish(reason='end_of_triggered_round')
        else:
            self._refill()
            s.update(round=s['round']+1, phase='bid', bid_queue=s['next_bid_queue'],
                     next_bid_queue=[], bids=[], current_marker=None)
            s['current_player'] = s['bid_queue'][0]//2
            if not self._legal_actions():
                self._unresolved('no_affordable_bid_and_zero_slots_full')

    def _step(self, action):
        s = self.state
        p = s['players'][s['current_player']]
        t = action['type']
        if t == 'bid':
            marker = s['bid_queue'].pop(0)
            cost = BIDS[max(0, BIDS.index(action['slot'])-action['fakirs'])]
            p['coins'] -= cost
            self._spend_fakirs(action['fakirs'])
            s['bids'].append(dict(marker=marker, slot=action['slot'], order=len(s['bids'])))
            self._event('bid', marker=marker, slot=action['slot'], paid=cost, fakirs=action['fakirs'])
            if s['bid_queue']:
                s['current_player'] = s['bid_queue'][0]//2
                if not self._legal_actions():
                    self._unresolved('no_affordable_bid_and_zero_slots_full')
            else:
                s['turn_queue'] = [b['marker'] for b in sorted(s['bids'], key=lambda b: (b['slot'], b['order']), reverse=True)]
                self._start_turn()
        elif t == 'start_sow':
            tile = s['board'][action['tile']]
            s.update(sow_hand=tile['meeples'][:], sow_position=action['tile'],
                     sow_start=action['tile'], sow_previous=-1, phase='sow')
            tile['meeples'] = [0]*5
            self._event('start_sow', tile=action['tile'])
        elif t == 'drop':
            i, color = action['tile'], COLORS.index(action['color'])
            tile = s['board'][i]
            s['sow_hand'][color] -= 1
            tile['meeples'][color] += 1
            owner = tile['owner']
            if owner is not None and 'marid' in s['players'][owner]['djinn']:
                s['players'][owner]['coins'] += 1 if owner == s['current_player'] else 2
            s['sow_previous'], s['sow_position'] = s['sow_position'], i
            self._event('drop', tile=i, color=action['color'])
            if not sum(s['sow_hand']):
                self._finish_sow(color)
        elif t == 'no_move':
            s['end_trigger'] = True
            s['phase'] = 'sale'
        elif t == 'activate':
            self._activate(action)
        elif t == 'keep_djinn':
            c = action['card']
            s['djinn_offer'].remove(c)
            s['djinn_discard'].extend(s['djinn_offer'])
            s['djinn_offer'] = []
            self._get_djinn(c)
            s['phase'] = s['resume_phase']
            s['resume_phase'] = None
        elif t == 'builder':
            self._spend_fakirs(action['fakirs'])
            coins = (s['tribe_count']+action['fakirs'])*blue_neighborhood(s['board'], s['endpoint'])
            coins *= 2 if 'echidna' in s['used'] else 1
            p['coins'] += coins
            self._event('builder_income', coins=coins, fakirs=action['fakirs'])
            s['phase'] = 'tile'
        elif t == 'assassinate':
            self._spend_fakirs(action['fakirs'])
            target = s['board'][action['tile']] if action['tile'] is not None else None
            for name in action['colors']:
                color = COLORS.index(name)
                if target is not None:
                    target['meeples'][color] -= 1
                else:
                    s['players'][action['opponent']][name] -= 1
                if 'kandicha' in p['djinn'] and name in ('yellow', 'white'):
                    p[name] += 1
                else:
                    s['bag'].append(color)
                    if 'kandicha' in p['djinn']:
                        if name == 'green':
                            self._get_resource(self._draw_card('resource'), public=False)
                        elif name == 'blue':
                            # Literal scope: these are the victim's hypothetical
                            # earnings, not the acting player's Builder tribe.
                            p['coins'] += blue_neighborhood(s['board'], action['tile'])
            self._notify('nekir')
            if target is not None and not sum(target['meeples']):
                self._claim(target)
            self._event('assassinate', tile=action['tile'], opponent=action['opponent'], colors=action['colors'], fakirs=action['fakirs'])
            s['phase'] = 'tile'
            self._check_no_moves()
        elif t == 'no_assassination':
            s['phase'] = 'tile'
        elif t == 'tile_effect':
            kind = s['board'][s['endpoint']]['kind']
            self._begin_placement('tree' if kind == 'oasis' else 'palace', [s['endpoint']], 'sale')
        elif t == 'place':
            pending = s['placement']
            kind = pending['kind']
            if action['payment'] is not None:
                self._pay(action['payment'])
                s['used'].append('hagis' if kind == 'palace' else 'lamia')
            if kind == 'palace':
                s['palace_supply'] -= 1
                s['board'][action['tile']]['palaces'] += 1
                self._notify('monkir')
            else:
                s['tree_supply'] -= 1
                s['board'][action['tile']]['trees'] += 1
            self._event('place', kind=kind, tile=action['tile'])
            s['phase'] = pending['resume']
            s['placement'] = None
        elif t == 'skip_tile':
            s['phase'] = 'sale'
        elif t == 'market':
            price = 3 if s['board'][s['endpoint']]['kind'] == 'small_market' else 6
            p['coins'] -= price
            cards = [s['resource_row'][i] for i in action['indices']]
            s['resource_row'] = [c for i, c in enumerate(s['resource_row']) if i not in action['indices']]
            for card in cards:
                self._get_resource(card)
            s['phase'] = 'sale'
        elif t == 'get_djinn':
            self._pay(action['payment'])
            s['djinn_row'].remove(action['card'])
            self._get_djinn(action['card'])
            s['phase'] = 'sale'
        elif t in ('sell', 'emergency_sell'):
            for card in action['goods']:
                p['resources'].remove(card)
                s['resource_discard'].append(card)
            p['coins'] += SALE[len(action['goods'])]
            self._event(t, goods=action['goods'], coins=SALE[len(action['goods'])])
        elif t == 'end_turn':
            self._end_turn()

    def score_breakdown(self):
        s = self.state
        result = []
        for seat, p in enumerate(s['players']):
            djinn = p['djinn']
            tiles = [t for t in s['board'] if t['owner'] == seat]
            goods = Counter(p['resources'])
            scores = dict(coins=p['coins'], yellow=p['yellow']*(3 if 'jafaar' in djinn else 1),
                yellow_majority=10*int(p['yellow'] > s['players'][1-seat]['yellow']),
                white=p['white']*(4 if 'shamhat' in djinn else 2), djinn=sum(VP[c] for c in djinn),
                tiles=sum(t['points'] for t in tiles), trees=sum(t['trees'] for t in tiles)*(5 if 'haurvatat' in djinn else 3),
                palaces=sum(t['palaces'] for t in tiles)*5,
                goods=goods_score(tuple(goods[c] for c in GOODS), goods['fakir']//2 if 'al_amin' in djinn else 0))
            scores['total'] = sum(scores.values())
            result.append(scores)
        return result

    def scores(self):
        return [p['total'] for p in self.score_breakdown()]

    def outcome(self):
        # Do not reveal private coins/resources via step's info channel, and
        # avoid unnecessary private goods-set optimization at every action.
        result = (super().outcome() if self.state.get('terminated') else
                  dict(terminated=False, scores=[None, None], winners=[], reason=None))
        if self.state.get('blocked_reason'):
            result.update(unresolved_rule=True, reason=self.state['blocked_reason'])
        return result

    def validate(self):
        super().validate()
        s = self.state
        assert len(s['board']) == 30
        expected_tiles = Counter({(kind, color, points): n for kind, color, points, n in DATA['tiles']})
        assert Counter((t['kind'], t['color'], t['points']) for t in s['board']) == expected_tiles
        assert s['phase'] in ('bid', 'move', 'sow', 'builder', 'assassin', 'tile', 'placement', 'djinn_choice', 'sale', 'unresolved')
        meeples = Counter(s['bag'])
        for t in s['board']:
            assert len(t['meeples']) == 5 and all(type(x) is int and x >= 0 for x in t['meeples'])
            meeples.update({i: n for i, n in enumerate(t['meeples'])})
            assert t['owner'] in (None, 0, 1) and t['trees'] >= 0 and t['palaces'] >= 0
        meeples.update({i: n for i, n in enumerate(s['sow_hand'])})
        resources = Counter(s['resource_deck']+s['resource_discard']+s['resource_row'])
        djinn = Counter(s['djinn_deck']+s['djinn_discard']+s['djinn_row']+s['djinn_offer'])
        for seat, p in enumerate(s['players']):
            assert all(type(p[k]) is int and p[k] >= 0 for k in ('coins', 'camels', 'yellow', 'white'))
            assert p['camels']+sum(t['owner'] == seat for t in s['board']) == 11
            meeples[COLORS.index('yellow')] += p['yellow']
            meeples[COLORS.index('white')] += p['white']
            resources.update(p['resources'])
            djinn.update(p['djinn'])
        assert dict(meeples) == {i: DATA['meeples'][c] for i, c in enumerate(COLORS)}
        assert resources == Counter(DATA['resources'])
        assert djinn == Counter({c: 1 for c in VP})
        assert s['tree_supply']+sum(t['trees'] for t in s['board']) == 12
        assert s['palace_supply']+sum(t['palaces'] for t in s['board']) == 10
        assert len(s['used']) == len(set(s['used']))
        if s['phase'] == 'sow':
            assert sum(s['sow_hand']) > 0
            assert can_finish(board_counts(s['board']), s['sow_hand'][:], s['sow_position'], s['sow_previous'])

    @staticmethod
    def features(obs, action):
        f = [0.]*24
        seat = obs['player']
        p = obs['players'][seat]
        board = obs['board']
        t = action['type']
        if t == 'bid':
            cost = BIDS[max(0, BIDS.index(action['slot'])-action['fakirs'])]
            f[1], f[2], f[3] = cost/18, action['slot']/18, action['fakirs']/5
            f[0] = -cost-.4*action['fakirs']+.05*action['slot']
        elif t in ('start_sow', 'drop'):
            tile = board[action['tile']]
            if t == 'start_sow':
                hand = tile['meeples']
                f[0] = max((board[j]['points'] + max(board[j]['meeples'])*2
                            for j in adjacent(action['tile'])), default=0) - .3*sum(hand)
            else:
                color = COLORS.index(action['color'])
                if sum(obs['sow_hand']) == 1:
                    count = tile['meeples'][color]+1
                    f[4] = count/5
                    f[5] = float(sum(tile['meeples']) == tile['meeples'][color] and tile['owner'] is None and p['camels'] > 0)
                    f[0] = f[5]*(tile['points']+3*tile['trees']+5*tile['palaces'])
                    name = action['color']
                    f[0] += count*({'yellow': 3, 'white': 2.5, 'green': 3.5, 'red': 2}.get(name, 0))
                    if name == 'blue':
                        f[0] += count*blue_neighborhood(board, action['tile'])
                else:
                    left = obs['sow_hand'][:]
                    left[color] -= 1
                    f[0] = max((board[j]['points']*.4 + max((board[j]['meeples'][k]*2 for k in range(5) if left[k]), default=0)
                                for j in adjacent(action['tile']) if j != obs['sow_position']), default=0)
        elif t == 'builder':
            f[0] = action['fakirs'] * (blue_neighborhood(board, obs['endpoint']) - 1)
        elif t == 'assassinate':
            if action['tile'] is not None:
                tile = board[action['tile']]
                f[0] = (tile['points']+3*tile['trees']+5*tile['palaces'])*int(
                    tile['owner'] is None and sum(tile['meeples']) == len(action['colors']) and p['camels'] > 0)
                if 'kandicha' in p['djinn']:
                    f[0] += sum({'yellow': 3, 'white': 3, 'green': 3,
                        'blue': blue_neighborhood(board, action['tile'])}.get(c, 0) for c in action['colors'])
            else:
                f[0] = sum(3 if c == 'white' else 2 for c in action['colors'])
            f[0] -= action['fakirs']
        elif t == 'market':
            owned = set(p['resources'])
            cards = [obs['resource_row'][i] for i in action['indices']]
            f[0] = sum(4 if c not in owned else 1 for c in cards) - (3 if len(cards) == 1 else 6)
        elif t in ('get_djinn', 'keep_djinn'):
            f[0] = VP[action['card']] - (3 if t == 'get_djinn' else 0)
            f[0] += {'jafaar': p['yellow']*2, 'shamhat': p['white']*2,
                     'haurvatat': sum(x['trees'] for x in board if x['owner'] == seat)*2}.get(action['card'], 1)
        elif t == 'place':
            owner = board[action['tile']]['owner']
            f[0] = 5 if owner == seat else -5 if owner == 1-seat else 0
            f[0] -= 1.5 if action['payment'] else 0
        elif t == 'activate':
            d, target = action['djinn'], action['target']
            f[0] = -2
            if d in ('leta', 'utug'):
                f[0] = board[target]['points'] - 3
            elif d == 'sibittis':
                f[0] = 3
            elif d == 'sloar':
                f[0] = 1
            elif d == 'echidna' and obs['phase'] == 'builder':
                f[0] = obs['tribe_count']*blue_neighborhood(board, obs['endpoint'])-3
            elif d == 'iblis' and obs['phase'] == 'assassin':
                f[0] = 1
        elif t in ('sell', 'emergency_sell'):
            # End scoring is equivalent; sell large sets, retain partial sets.
            f[0] = len(action['goods'])-5 if p['coins'] < 6 else -1
        elif t in ('tile_effect', 'no_move', 'no_assassination'):
            f[0] = 0
        f[15] = p['coins']/100
        f[16] = p['camels']/11
        f[17] = p['white']/10
        f[18] = p['yellow']/10
        f[19] = len(p['resources'])/20
        f[20] = len(p['djinn'])/10
        f[21] = obs['round']/20
        return f


GAME_CLASS = FiveTribes
