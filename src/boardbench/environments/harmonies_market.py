"""Research curriculum, NOT official solo/duo: own board + random market depletion.

Five-slot vacancies refill after our turn; a stationary independent RNG removes
one bundle and one available card, then vacancies refill again. No other board,
score, active-card capacity, or opponent-triggered termination exists. Thirteen
own turns cap resource opportunity to the old solo maximum; scoring is unchanged.
"""
from collections import Counter
import random

from boardbench.contracts import InvalidAction, derive_seed
from boardbench.environments.board import integer, tuples
from boardbench.environments.harmonies import Harmonies, INVENTORY, CARDS, COLORS, CELLS, STACKS
from boardbench.v128.environment import FastHarmonies, public_copy


class HarmoniesMarket(Harmonies):
    task_id = 'harmonies_market'
    rules_version = 'harmonies_market_a_base32_random13_v1'
    turn_limit = 13

    @classmethod
    def task_spec(cls):
        return {**Harmonies.task_spec(), 'id': cls.task_id, 'version': cls.rules_version,
                'token_market_slots': 5, 'animal_market_slots': 5, 'turn_limit': cls.turn_limit,
                'supply_ruling': 'own vacancies refill; random bundle/card removed; vacancies refill',
                'research_variant': True, 'opponent_board': False, 'reward': 'own terminal score'}

    def reset(self, seed):
        self._bag = [c for c, n in INVENTORY.items() for _ in range(n)]
        self._deck = list(CARDS)
        random.Random(derive_seed(seed, 'tokens')).shuffle(self._bag)
        random.Random(derive_seed(seed, 'cards')).shuffle(self._deck)
        self._market_rng = random.Random(derive_seed(seed, 'random_market'))
        self.s = dict(board=[dict(stack_id=0, animal=None) for _ in CELLS],
                      token_market=[None]*5, animal_market=[None]*5,
                      active_cards=[], completed_cards=[], turn_index=1, completed_turns=0,
                      decision_index=0, bundle_taken=False, card_taken=False,
                      pending_tokens=dict.fromkeys(COLORS, 0), discarded_tokens=dict.fromkeys(COLORS, 0),
                      discarded_cards=[], remaining_tokens=dict(INVENTORY), remaining_cards=list(CARDS),
                      terminated=False, score=None, end_reason=None)
        self._refill()
        return self.observe(), self.reset_info()

    def _refill(self):
        exhausted = False
        for i, bundle in enumerate(self.s['token_market']):
            if bundle is None:
                if len(self._bag) < 3:
                    exhausted = True
                else:
                    self.s['token_market'][i], self._bag = self._bag[:3], self._bag[3:]
        for i, card in enumerate(self.s['animal_market']):
            if card is None and self._deck: self.s['animal_market'][i] = self._deck.pop(0)
        self.s['remaining_tokens'] = {c: self._bag.count(c) for c in COLORS}
        self.s['remaining_cards'] = sorted(self._deck)
        return exhausted

    def legal_actions(self):
        if self.s['terminated']: return []
        actions = [a for a in super().legal_actions()
                   if a['type'] != 'choose_bundle' and 'discard_card_slot' not in a]
        if not self.s['bundle_taken']:
            actions = [dict(type='choose_bundle', bundle_id=i) for i,b in enumerate(self.s['token_market'])
                       if b is not None] + actions
        return actions

    def _validate(self, action):
        if isinstance(action, dict) and action.get('type') == 'end_turn' and action != {'type': 'end_turn'}:
            raise InvalidAction('random-market curriculum has no optional card discard')
        if isinstance(action, dict) and action.get('type') in ('choose_bundle', 'take_card'):
            bundle = action['type'] == 'choose_bundle'
            key, field = ('bundle_id', 'token_market') if bundle else ('market_slot', 'animal_market')
            if (set(action) != {'type', key} or not integer(action[key], 0, 5)
                    or self.s[field][action[key]] is None
                    or self.s['bundle_taken' if bundle else 'card_taken']
                    or (not bundle and len(self.s['active_cards']) >= 4)):
                raise InvalidAction('invalid random market selection')
        else:
            super()._validate(action)

    def step(self, action):
        if not isinstance(action, dict) or action.get('type') != 'end_turn':
            return super().step(action)
        self._require_live(); self._validate(action)
        self.s['completed_turns'] += 1
        reason = ('board_full_enough' if sum(c['stack_id']==0 for c in self.s['board']) <= 2
                  else 'turn_limit' if self.s['completed_turns'] >= self.turn_limit else None)
        if reason is None:
            if self._refill():
                reason = 'supply_exhausted'
            else:
                # Choices only use current PUBLIC slots; this RNG never reads the
                # private bag/deck order, learner RNG, or reset seed as features.
                tokens = [i for i,b in enumerate(self.s['token_market']) if b is not None]
                i = self._market_rng.choice(tokens)
                for color in self.s['token_market'][i]: self.s['discarded_tokens'][color] += 1
                self.s['token_market'][i] = None
                cards = [i for i,c in enumerate(self.s['animal_market']) if c is not None]
                if cards:
                    i = self._market_rng.choice(cards)
                    self.s['discarded_cards'].append(self.s['animal_market'][i]); self.s['animal_market'][i] = None
                if self._refill(): reason = 'supply_exhausted'
        if reason is None:
            self.s['turn_index'] += 1
            self.s['bundle_taken'] = self.s['card_taken'] = False
        self.s['terminated'] = reason is not None
        self.s['end_reason'] = reason
        return self._result(reason)

    def _private_state(self):
        return {**super()._private_state(), 'market_rng': self._market_rng.getstate()}

    def _load_private(self, state):
        super()._load_private(state)
        self._market_rng = random.Random(); self._market_rng.setstate(tuples(state['market_rng']))
        self.validate()

    def _resample(self, seed):
        super()._resample(seed)
        self._market_rng = random.Random(derive_seed(seed, 'random_market'))
        self.validate()

    def validate(self):
        s = self.s
        if len(s['token_market']) != 5 or len(s['animal_market']) != 5 or len(s['board']) != 23:
            raise ValueError('market/board dimensions')
        if not integer(s['completed_turns'],0,14) or not integer(s['turn_index'],1,14):
            raise ValueError('turn counters')
        if s['turn_index'] != s['completed_turns'] + (not s['terminated']):
            raise ValueError('turn boundary mismatch')
        tokens = Counter(self._bag)
        if tokens != Counter(s['remaining_tokens']) or sorted(self._deck) != s['remaining_cards']:
            raise ValueError('private/public supply mismatch')
        cards = list(self._deck)+s['discarded_cards']
        for bundle in s['token_market']:
            if bundle is not None:
                if len(bundle) != 3: raise ValueError('bundle size')
                tokens.update(bundle)
        cards += [c for c in s['animal_market'] if c is not None]
        owned = s['active_cards']+s['completed_cards']
        cards += [c['card_id'] for c in owned]
        if len(s['active_cards']) > 4: raise ValueError('active card limit')
        animal_counts = Counter(c['animal'] for c in s['board'] if c['animal'] is not None)
        for c in s['board']:
            if not integer(c['stack_id'],0,len(STACKS)): raise ValueError('invalid stack')
            tokens.update(STACKS[c['stack_id']])
        for c in owned:
            if c['placed_count'] != animal_counts[c['card_id']]: raise ValueError('animal count mismatch')
        if set(animal_counts)-{c['card_id'] for c in owned}: raise ValueError('unowned animal')
        tokens.update(s['pending_tokens']); tokens.update(s['discarded_tokens'])
        if tokens != Counter(INVENTORY) or sorted(cards) != list(CARDS): raise ValueError('supply conservation')
        return True


class FastHarmoniesMarket(HarmoniesMarket, FastHarmonies):
    def observe(self):
        return {**public_copy(self.s), 'task_id': self.task_id, 'rules_version': self.rules_version,
                'legal_actions': self.legal_actions(), 'score_breakdown': self.score_breakdown()}
