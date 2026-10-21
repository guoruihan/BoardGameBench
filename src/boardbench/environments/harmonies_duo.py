"""Official two-player Harmonies, side A/base32, shared supply and equal turns.

Terrain/animal transitions and scores are reused unchanged from the solo engine.
This environment is deliberately separate from the single-agent benchmark runner.
"""
from collections import Counter
from copy import deepcopy
import random

from boardbench.contracts import InvalidAction, StepResult, derive_seed
from .board import BoardGame, integer
from .harmonies import Harmonies, CARDS, CELLS, COLORS, INVENTORY, STACKS
from boardbench.v128.environment import FastHarmonies, public_copy

PERSONAL = ('board', 'active_cards', 'completed_cards', 'turn_index', 'bundle_taken',
            'card_taken', 'pending_tokens')


class _Personal(Harmonies):
    """Reuse atomic token/card/animal steps, never the solo end-turn transition."""
    def _validate(self, action):
        if isinstance(action, dict) and action.get('type') in ('choose_bundle', 'take_card'):
            key = 'bundle_id' if action['type'] == 'choose_bundle' else 'market_slot'
            field = 'token_market' if key == 'bundle_id' else 'animal_market'
            if (set(action) != {'type', key} or not integer(action[key], 0, 5)
                    or self.s[field][action[key]] is None
                    or self.s['bundle_taken' if key == 'bundle_id' else 'card_taken']
                    or (key == 'market_slot' and len(self.s['active_cards']) >= 4)):
                raise InvalidAction('invalid shared market selection')
        else:
            super()._validate(action)

    def _result(self, reason=None):
        return None  # The outer game owns observations, rewards and turn order.


class _FastPersonal(_Personal, FastHarmonies):
    legal_actions = FastHarmonies.legal_actions


class HarmoniesDuo(BoardGame):
    task_id, rules_version = 'harmonies_duo', 'harmonies_duo_a_base32_v1'
    personal_class = _Personal

    def __init__(self):
        self.reset(0)

    @classmethod
    def task_spec(cls):
        return {**Harmonies.task_spec(), 'id': cls.task_id, 'version': cls.rules_version,
                'players': 2, 'token_market_slots': 5, 'animal_market_slots': 5,
                'max_actions': 768, 'reward': 'terminal score margin for the acting player',
                'supply_ruling': 'trigger when pouch is empty at refill; finish current round',
                'tie_break': 'most placed animal cubes; otherwise shared victory',
                'official_rules': 'https://cdn.svc.asmodee.net/production-libellud/uploads/2024/12/HARMONIES_Rules_EN.pdf'}

    def reset(self, seed, first_player=0):
        if not integer(first_player, 0, 2):
            raise ValueError('first_player must be a seat')
        self._bag = [c for c, n in INVENTORY.items() for _ in range(n)]
        self._deck = list(CARDS)
        random.Random(derive_seed(seed, 'tokens')).shuffle(self._bag)
        random.Random(derive_seed(seed, 'cards')).shuffle(self._deck)
        players = [dict(board=[dict(stack_id=0, animal=None) for _ in CELLS], active_cards=[],
                        completed_cards=[], turn_index=1, completed_turns=0, bundle_taken=False,
                        card_taken=False, pending_tokens=dict.fromkeys(COLORS, 0)) for _ in range(2)]
        self.s = dict(players=players, token_market=[None]*5, animal_market=[None]*5,
                      remaining_tokens=dict(INVENTORY), remaining_cards=list(CARDS),
                      discarded_tokens=dict.fromkeys(COLORS, 0), discarded_cards=[],
                      first_player=first_player, active_player=first_player, decision_index=0,
                      final_round=False, end_reason=None, terminated=False, scores=None, winners=None)
        self._refill()
        return self.observe(), self.reset_info()

    def _refill(self):
        exhausted = False
        for i, bundle in enumerate(self.s['token_market']):
            if bundle is None:
                if len(self._bag) >= 3:
                    self.s['token_market'][i], self._bag = self._bag[:3], self._bag[3:]
                else:
                    exhausted = True
        for i, card in enumerate(self.s['animal_market']):
            if card is None and self._deck:
                self.s['animal_market'][i] = self._deck.pop(0)
        self.s['remaining_tokens'] = {c: self._bag.count(c) for c in COLORS}
        self.s['remaining_cards'] = sorted(self._deck)
        return exhausted

    def personal(self, seat=None):
        seat = self.s['active_player'] if seat is None else seat
        obj = self.personal_class.__new__(self.personal_class)
        obj.s = {**self.s['players'][seat], **{k: self.s[k] for k in (
            'token_market', 'animal_market', 'remaining_tokens', 'remaining_cards',
            'discarded_tokens', 'discarded_cards', 'decision_index', 'terminated')}, 'score': None}
        return obj

    def legal_actions(self):
        if self.s['terminated']:
            return []
        actions = self.personal().legal_actions()
        # Personal geometry is identical; only market width and solo discard differ.
        actions = [a for a in actions if a['type'] != 'choose_bundle' and 'discard_card_slot' not in a]
        if not self.s['players'][self.s['active_player']]['bundle_taken']:
            actions = [dict(type='choose_bundle', bundle_id=i) for i, b in
                       enumerate(self.s['token_market']) if b is not None] + actions
        return actions

    def score_breakdown(self):
        return [self.personal(i).score_breakdown() for i in range(2)]

    def observe(self):
        return {**deepcopy(self.s), 'task_id': self.task_id, 'rules_version': self.rules_version,
                'legal_actions': self.legal_actions(), 'score_breakdown': self.score_breakdown()}

    def step(self, action):
        self._require_live()
        seat = self.s['active_player']
        person = self.personal(seat)
        if isinstance(action, dict) and action.get('type') == 'end_turn':
            if action != {'type': 'end_turn'}:
                raise InvalidAction('multiplayer has no optional market discard')
            person._validate(action)
            player = self.s['players'][seat]
            full = sum(c['stack_id'] == 0 for c in player['board']) <= 2
            empty = self._refill()
            if full or empty:
                self.s['final_round'] = True
                self.s['end_reason'] = self.s['end_reason'] or ('board_full_enough' if full else 'empty_pouch')
            player['completed_turns'] += 1
            player['turn_index'] += 1
            player['bundle_taken'] = player['card_taken'] = False
            if self.s['final_round'] and seat != self.s['first_player']:
                self.s['terminated'] = True
                self.s['scores'] = [s['total'] for s in self.score_breakdown()]
                keys = [(score, sum(c['animal'] is not None for c in p['board']))
                        for score, p in zip(self.s['scores'], self.s['players'])]
                self.s['winners'] = [i for i, key in enumerate(keys) if key == max(keys)]
            else:
                self.s['active_player'] = 1-seat
        else:
            person.step(action)
            self.s['players'][seat].update({k: person.s[k] for k in PERSONAL})
        self.s['decision_index'] += 1
        scores = [s['total'] for s in self.score_breakdown()]
        done = self.s['terminated']
        return StepResult(self.observe(), float(scores[seat]-scores[1-seat]) if done else 0., done,
                          info={'acting_player': seat, 'scores': scores, 'winners': self.s['winners'],
                                'reason': self.s['end_reason'] if done else None})

    def _private_state(self):
        return dict(bag=self._bag, deck=self._deck)

    def _load_private(self, private):
        self._bag, self._deck = list(private['bag']), list(private['deck'])
        self.validate()

    def _resample(self, sim_seed):
        self._bag = [c for c in COLORS for _ in range(self.s['remaining_tokens'][c])]
        self._deck = list(self.s['remaining_cards'])
        rng = random.Random(sim_seed); rng.shuffle(self._bag); rng.shuffle(self._deck)
        self.validate()

    def validate(self):
        s = self.s
        if (len(s['players']) != 2 or len(s['token_market']) != 5 or len(s['animal_market']) != 5
                or not integer(s['active_player'], 0, 2) or not integer(s['first_player'], 0, 2)):
            raise ValueError('invalid duo state shape/seat')
        tokens = Counter(self._bag); cards = list(self._deck)
        if tokens != Counter(s['remaining_tokens']) or sorted(cards) != s['remaining_cards']:
            raise ValueError('private/public supply mismatch')
        for bundle in s['token_market']:
            if bundle is not None:
                if len(bundle) != 3:
                    raise ValueError('not a three-token bundle')
                tokens.update(bundle)
        cards += [c for c in s['animal_market'] if c is not None]
        for p in s['players']:
            if len(p['board']) != 23 or len(p['active_cards']) > 4:
                raise ValueError('invalid personal board/cards')
            if (set(p['pending_tokens']) != set(COLORS) or
                    any(not integer(v, 0, 4) for v in p['pending_tokens'].values())
                    or sum(p['pending_tokens'].values()) > 3
                    or not integer(p['completed_turns'], 0, 40)
                    or p['turn_index'] != p['completed_turns']+1):
                raise ValueError('invalid pending supply or turn counter')
            tokens.update(p['pending_tokens'])
            counts = Counter(c['animal'] for c in p['board'] if c['animal'] is not None)
            owned = p['active_cards'] + p['completed_cards']
            for c in p['board']:
                if not integer(c['stack_id'], 0, len(STACKS)):
                    raise ValueError('invalid stack')
                tokens.update(STACKS[c['stack_id']])
            for c in owned:
                if (c['instance_id'] != c['card_id'] or c['card_id'] not in CARDS
                        or not integer(c['placed_count'], 0, CARDS[c['card_id']]['cube_count']+1)
                        or c['placed_count'] != counts[c['card_id']]
                        or (c in p['completed_cards']) != (c['placed_count'] == CARDS[c['card_id']]['cube_count'])):
                    raise ValueError('animal ownership/placement mismatch')
            if set(counts)-{c['card_id'] for c in owned}:
                raise ValueError('unowned animal')
            cards += [c['card_id'] for c in owned]
        if tokens != Counter(INVENTORY) or sorted(cards) != list(CARDS):
            raise ValueError('token/card conservation violated')
        if any(s['discarded_tokens'].values()) or s['discarded_cards']:
            raise ValueError('no solo discard in multiplayer')
        turns = [p['completed_turns'] for p in s['players']]
        if abs(turns[0]-turns[1]) > 1 or (s['terminated'] and turns[0] != turns[1]):
            raise ValueError('equal-round turn order violated')
        return True


class FastHarmoniesDuo(HarmoniesDuo):
    personal_class = _FastPersonal

    def observe(self):
        state = {k: v for k, v in self.s.items() if k != 'players'}
        return {**public_copy(state), 'players': [public_copy(p) for p in self.s['players']],
                'task_id': self.task_id, 'rules_version': self.rules_version,
                'legal_actions': self.legal_actions(), 'score_breakdown': self.score_breakdown()}


def player_view(obs, seat=None, market_slots=None):
    """Explicit internal solo-shaped projection; not a different game or simulator.

    All public fields remain accessible on the original duo observation. Selecting
    three slots is only used by the pretrained feature encoder, never by the game.
    """
    seat = obs['active_player'] if seat is None else seat
    slots = range(5) if market_slots is None else market_slots
    return {**public_copy(obs['players'][seat]), **public_copy({k: obs[k] for k in (
        'remaining_tokens', 'remaining_cards', 'discarded_tokens', 'discarded_cards', 'decision_index', 'terminated')}),
        'token_market': [deepcopy(obs['token_market'][i]) for i in slots],
        'animal_market': [obs['animal_market'][i] for i in slots],
        'task_id': 'harmonies', 'rules_version': 'harmonies_solo_a_v1', 'score': None,
        'score_breakdown': deepcopy(obs['score_breakdown'][seat]), 'legal_actions': []}
