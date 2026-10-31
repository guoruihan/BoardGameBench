"""Century: A New World (2019), original A1/B1/C1/D1 two-player setting.

Only point-deck top and bonus-stack tops are visible. Numerical component data
are transcribed from photographed original components; see the provenance file.
"""
from collections import Counter
from pathlib import Path
import json

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / 'data/century_a_new_world.json').read_text())
CARDS = {x['id']: x for x in DATA['point_cards']}
BONUSES = {x['id']: x for x in DATA['bonus_tiles']}
EXPLORATIONS = {x['id']: x for x in DATA['exploration_tiles']}
LOCATIONS = {x['id']: x for x in DATA['locations']}


def add(a, b, times=1):
    return [x + times*y for x, y in zip(a, b)]


def affordable(resources, cost):
    return all(a >= b for a, b in zip(resources, cost))


def upgrade_results(resources, count):
    """All distinct legal outcomes, including forfeited upgrades."""
    seen = {tuple(resources)}
    frontier = seen.copy()
    for _ in range(count):
        nxt = set()
        for value in frontier:
            for color in range(3):
                if value[color]:
                    result = list(value)
                    result[color] -= 1
                    result[color+1] += 1
                    nxt.add(tuple(result))
        seen.update(nxt)
        frontier = nxt
    return sorted(seen)


def icon_counts(player):
    icons = Counter(CARDS[c]['world_icon'] for c in player['cards'])
    for e in player['explorations']:
        tile = EXPLORATIONS[e]
        if tile['kind'] == 'icon':
            icons[tile['value']] += 1
    return icons


def bonus_value(bonus_id, player):
    tile = BONUSES[bonus_id]
    icons = icon_counts(player)
    if tile['kind'] == 'fixed':
        count = 1
    elif tile['kind'] == 'workers':
        count = (12-player['reserve'])//2
    elif tile['kind'] == 'icon':
        count = icons[tile['icons'][0]]
    else:
        required = Counter(tile['icons'])
        count = min(icons[k]//n for k, n in required.items())
    return tile['points']*count


def location_cost(location, occupation, player):
    base = occupation['count']+1 if occupation else location['workers']
    discount = sum(CARDS[c]['ability'] == 'help' and CARDS[c]['target'] in location['icons']
                   for c in player['cards'])
    return max(1, base-discount)


def tool_gain(location, player):
    gain = [0]*4
    for card_id in player['cards']:
        card = CARDS[card_id]
        if card['ability'] == 'tool' and card['target'] in location['icons']:
            gain[card['payload']] += 1
    return gain


def resources_value(resources):
    return sum(n*w for n, w in zip(resources, [1., 2., 3., 4.]))


class CenturyANewWorld(DuoGame):
    game_id = 'century_a_new_world'
    rules_version = '2019-03-12.v1'
    config_id = 'original_A1_B1_C1_D1_duo'

    def _reset(self, first_player):
        deck = list(CARDS)
        self.rng.shuffle(deck)
        bonuses = list(BONUSES)
        self.rng.shuffle(bonuses)
        exploration = list(EXPLORATIONS)
        self.rng.shuffle(exploration)
        players = [dict(resources=[3 if p == first_player else 4, 0, 0, 0],
                        available=7, reserve=5, cards=[], bonuses=[], explorations=[], turns=0)
                   for p in (0, 1)]
        stacks = []
        for count in DATA['bonus_stack_sizes']:
            stacks.append([bonuses.pop() for _ in range(count)])
        self.state.update(players=players, deck=deck[4:], market=deck[:4],
                          bonus_stacks=stacks, removed_bonuses=bonuses,
                          occupation={k: None for k in LOCATIONS},
                          covers={k: exploration.pop() if v['covered'] else None
                                  for k, v in LOCATIONS.items()},
                          phase='turn', pending_slot=None, final_round=False, history=[])
        self.validate()

    def _legal_actions(self):
        state = self.state
        p = self.current_player
        player = state['players'][p]
        if state['phase'] == 'explore':
            return [dict(type='explore', location=k) for k, v in state['covers'].items() if v]
        if state['phase'] == 'discard':
            # Discarding one cube is a genuine sequential choice; every multiset
            # can be reached without materializing a large combinatorial list.
            return [dict(type='discard', color=c) for c, n in enumerate(player['resources']) if n]
        actions = [dict(type='rest')]
        for key, location in LOCATIONS.items():
            occupation = state['occupation'][key]
            if state['covers'][key] or (occupation and occupation['owner'] == p):
                continue
            cost = location_cost(location, occupation, player)
            if player['available'] < cost:
                continue
            resources = add(player['resources'], tool_gain(location, player))
            action = dict(type='work', location=key)
            kind = location['kind']
            if kind == 'produce':
                actions.append(action)
            elif kind == 'upgrade':
                actions.extend(dict(action, resources=list(r)) for r in
                               upgrade_results(resources, location['upgrades']))
            elif kind == 'trade':
                maximum = min(resources[c]//n for c, n in enumerate(location['cost']) if n)
                actions.extend(dict(action, repeats=n) for n in range(1, maximum+1))
            else:
                card = state['market'][location['slot']]
                can_buy = card is not None and affordable(resources, CARDS[card]['cost'])
                bonus_options = [None]
                if len(player['bonuses']) < 3:
                    bonus_options += [i for i in location['bonus_stacks'] if state['bonus_stacks'][i]]
                for bonus in bonus_options:
                    for buy in ([False, True] if can_buy else [False]):
                        if bonus is not None or buy:
                            actions.append(dict(action, buy=buy, bonus_stack=bonus))
        return actions

    def _step(self, action):
        state = self.state
        p = self.current_player
        player = state['players'][p]
        state['history'].append(dict(player=p, action=action))
        if action['type'] == 'discard':
            player['resources'][action['color']] -= 1
            if sum(player['resources']) <= 10:
                self._end_turn()
            return
        if action['type'] == 'explore':
            tile_id = state['covers'][action['location']]
            state['covers'][action['location']] = None
            player['explorations'].append(tile_id)
            tile = EXPLORATIONS[tile_id]
            if tile['kind'] == 'resources':
                player['resources'] = add(player['resources'], tile['value'])
            elif tile['kind'] == 'settler':
                self._unlock(player, tile['value'])
            self._complete_work()
            return
        if action['type'] == 'rest':
            for key, occupation in state['occupation'].items():
                if occupation and occupation['owner'] == p:
                    player['available'] += occupation['count']
                    state['occupation'][key] = None
            self._complete_work()
            return
        key = action['location']
        location = LOCATIONS[key]
        occupation = state['occupation'][key]
        cost = location_cost(location, occupation, player)
        if occupation:
            state['players'][occupation['owner']]['available'] += occupation['count']
        player['available'] -= cost
        state['occupation'][key] = dict(owner=p, count=cost)
        player['resources'] = add(player['resources'], tool_gain(location, player))
        kind = location['kind']
        if kind == 'produce':
            player['resources'] = add(player['resources'], location['gain'])
        elif kind == 'upgrade':
            player['resources'] = action['resources']
        elif kind == 'trade':
            player['resources'] = add(add(player['resources'], location['cost'], -action['repeats']),
                                      location['gain'], action['repeats'])
        else:
            stack = action['bonus_stack']
            if stack is not None:
                player['bonuses'].append(state['bonus_stacks'][stack].pop(0))
            if action['buy']:
                slot = location['slot']
                card_id = state['market'][slot]
                state['market'][slot] = None
                state['pending_slot'] = slot
                card = CARDS[card_id]
                player['resources'] = add(player['resources'], card['cost'], -1)
                player['cards'].append(card_id)
                if len(player['cards']) >= 8:
                    state['final_round'] = True
                if card['ability'] == 'settlers':
                    self._unlock(player, card['payload'])
                elif card['ability'] == 'explore' and any(state['covers'].values()):
                    state['phase'] = 'explore'
                    return
        self._complete_work()

    @staticmethod
    def _unlock(player, amount):
        count = min(amount, player['reserve'])
        player['available'] += count
        player['reserve'] -= count

    def _complete_work(self):
        state = self.state
        # Resolve the acquired card's immediate choice before exposing the next
        # previously hidden deck card. The old top was already public.
        slot = state['pending_slot']
        if slot is not None:
            state['market'].pop(slot)
            state['market'].append(state['deck'].pop(0) if state['deck'] else None)
            state['pending_slot'] = None
        if sum(state['players'][self.current_player]['resources']) > 10:
            state['phase'] = 'discard'
        else:
            self._end_turn()

    def _end_turn(self):
        state = self.state
        p = self.current_player
        state['players'][p]['turns'] += 1
        state['phase'] = 'turn'
        if state['final_round'] and p != state['first_player']:
            scores = self.scores()
            winner = (1-state['first_player']) if scores[0] == scores[1] else int(scores[1] > scores[0])
            self.finish([winner], reason='eight_point_cards_round_complete')
        else:
            state['current_player'] = 1-p

    def _observe(self, player):
        state = self.state
        return dict(players=state['players'], phase=state['phase'], first_player=state['first_player'],
                    market=state['market'], deck_count=len(state['deck']),
                    deck_top=state['deck'][0] if state['deck'] else None,
                    bonus_stacks=[dict(top=stack[0] if stack else None, count=len(stack))
                                  for stack in state['bonus_stacks']],
                    occupation=state['occupation'], covers=state['covers'],
                    pending_slot=state['pending_slot'], final_round=state['final_round'],
                    history=state['history'])

    def score_breakdown(self):
        results = []
        for player in self.state['players']:
            parts = dict(cards=sum(CARDS[c]['vp'] for c in player['cards']),
                         bonuses=sum(bonus_value(b, player) for b in player['bonuses']),
                         exploration=sum(EXPLORATIONS[e]['value'] for e in player['explorations']
                                         if EXPLORATIONS[e]['kind'] == 'points'),
                         resources=sum(player['resources'][1:]))
            parts['total'] = sum(parts.values())
            results.append(parts)
        return results

    def scores(self):
        return [x['total'] for x in self.score_breakdown()]

    def validate(self):
        super().validate()
        state = self.state
        assert state['phase'] in ('turn', 'explore', 'discard')
        assert set(state['occupation']) == set(LOCATIONS) == set(state['covers'])
        all_cards = state['deck'] + [x for x in state['market'] if x]
        all_bonus = state['removed_bonuses'] + [b for stack in state['bonus_stacks'] for b in stack]
        all_exploration = [e for e in state['covers'].values() if e]
        assert len(state['market']) == 4
        assert len(state['bonus_stacks']) == 6
        for p, player in enumerate(state['players']):
            workers = 0
            for key, occupation in state['occupation'].items():
                if occupation:
                    assert occupation['owner'] in (0, 1)
                    assert type(occupation['count']) is int and occupation['count'] >= 1
                    assert state['covers'][key] is None
                    if occupation['owner'] == p:
                        workers += occupation['count']
            assert all(type(player[k]) is int and player[k] >= 0 for k in ('available', 'reserve', 'turns'))
            assert player['available'] + player['reserve'] + workers == 12
            assert 0 <= player['reserve'] <= 5
            assert len(player['resources']) == 4
            assert all(type(n) is int and n >= 0 for n in player['resources'])
            if p != state['current_player'] or state['phase'] == 'turn':
                assert sum(player['resources']) <= 10
            assert len(player['bonuses']) <= 3
            all_cards += player['cards']
            all_bonus += player['bonuses']
            all_exploration += player['explorations']
        assert Counter(all_cards) == Counter(CARDS.keys())
        assert Counter(all_bonus) == Counter(BONUSES.keys())
        assert Counter(all_exploration) == Counter(EXPLORATIONS.keys())
        assert (state['pending_slot'] is not None) == (state['phase'] == 'explore')
        if state['phase'] == 'explore':
            assert any(state['covers'].values())
            assert state['market'][state['pending_slot']] is None
        if state['phase'] == 'discard':
            assert sum(state['players'][state['current_player']]['resources']) > 10
        turns = [p['turns'] for p in state['players']]
        assert abs(turns[0]-turns[1]) <= 1
        if state['terminated']:
            assert state['final_round'] and turns[0] == turns[1]
            assert max(len(p['cards']) for p in state['players']) >= 8

    @staticmethod
    def features(observation, action):
        player = observation['players'][observation['current_player']]
        before = player['resources']
        after = before[:]
        feature = [0.0]*24
        if action['type'] == 'discard':
            after[action['color']] -= 1
            feature[0] = -[1., 2., 3., 4.][action['color']]
            feature[1] = action['color']/3
            return feature
        if action['type'] == 'rest':
            recalled = sum(o['count'] for o in observation['occupation'].values()
                           if o and o['owner'] == observation['current_player'])
            feature[0] = 0.35*recalled - 1.5 - 0.5*player['available']
            feature[2] = recalled/12
            return feature
        if action['type'] == 'explore':
            tile = EXPLORATIONS[observation['covers'][action['location']]]
            if tile['kind'] == 'resources':
                value = resources_value(tile['value'])
            elif tile['kind'] == 'settler':
                value = 2.5 if player['reserve'] else 0
            elif tile['kind'] == 'points':
                value = 3
            else:
                copy = dict(player, explorations=player['explorations']+[tile['id']])
                value = 1 + sum(bonus_value(b, copy)-bonus_value(b, player) for b in player['bonuses'])
            feature[0] = value
            feature[3] = value/10
            return feature
        location = LOCATIONS[action['location']]
        occupation = observation['occupation'][action['location']]
        workers = location_cost(location, occupation, player)
        after = add(after, tool_gain(location, player))
        kind = location['kind']
        reward = 0.0
        if kind == 'produce':
            after = add(after, location['gain'])
        elif kind == 'trade':
            after = add(add(after, location['cost'], -action['repeats']), location['gain'], action['repeats'])
        elif kind == 'upgrade':
            after = action['resources']
        else:
            if action['bonus_stack'] is not None:
                b = observation['bonus_stacks'][action['bonus_stack']]['top']
                reward += 1.0 + bonus_value(b, player)
                feature[4] = bonus_value(b, player)/12
            if action['buy']:
                card = CARDS[observation['market'][location['slot']]]
                after = add(after, card['cost'], -1)
                copy = dict(player, cards=player['cards']+[card['id']])
                synergy = sum(bonus_value(b, copy)-bonus_value(b, player) for b in player['bonuses'])
                reward += 5 + 1.2*card['vp'] + synergy
                if card['ability'] == 'tool':
                    reward += max(0, 4-len(player['cards'])/2)
                elif card['ability'] == 'help':
                    reward += max(0, 3-len(player['cards'])/3)
                elif card['ability'] == 'settlers':
                    reward += 1.4*min(player['reserve'], card['payload'])
                else:
                    reward += 3
                feature[5] = card['vp']/10
                feature[6] = synergy/10
                feature[7] = 1
        # Shape resource decisions toward visible orders, without reading any
        # hidden deck/stack order. Positive progress is rewarded even when a
        # trade keeps total resource tier value constant.
        market = [CARDS[c] for c in observation['market'] if c]
        def distance(resources):
            return min((sum(max(0, n-resources[c])*(c+1) for c, n in enumerate(card['cost']))
                        for card in market), default=0)
        progress = distance(before)-distance(after)
        gain = resources_value(after)-resources_value(before)
        overflow = max(0, sum(after)-10)
        reward += 0.5*gain + 1.2*progress - 0.4*workers - 1.3*overflow
        feature[0] = reward
        feature[8] = gain/12
        feature[9] = progress/12
        feature[10] = workers/12
        feature[11] = overflow/10
        feature[12] = len(player['cards'])/8
        feature[13] = player['available']/12
        feature[14] = float(bool(occupation))
        feature[15] = sum(after)/10
        return feature


GAME_CLASS = CenturyANewWorld
