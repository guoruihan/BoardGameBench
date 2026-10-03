"""Public-state encodings; game rules and semantic engine actions are unchanged."""
import math
from numbers import Integral

from boardbench.contracts import InvalidAction
from boardbench.environments.harmonies import CARDS, STACK_INDEX
from boardbench.environments.numerical import action_id as legacy_id, features, onehot

COUNTS = {'legacy4564_v1': 4564, 'no_rotation884_v1': 884, 'slots240_v1': 240}
OBSERVATIONS = ('legacy_v1', 'slots_v1', 'cards_v1')
STOP = {'type': 'stop_episode'}
MAX_PATTERN = max(len(c['pattern_cells']) for c in CARDS.values())
MAX_CUBES = max(c['cube_count'] for c in CARDS.values())


def _card_vector(card_id):
    card = CARDS[card_id]
    result = []
    for req in card['pattern_cells']:
        allowed = (12, 13, 14) if req['terrain'] == 'building' else (
            STACK_INDEX[tuple(req['stack_bottom_to_top'])],)
        q, r = req['cell']
        result.extend([1., q / 4, r / 4, float(q == r == 0)])
        result.extend(float(i in allowed) for i in range(15))
    result.extend([0.] * ((MAX_PATTERN - len(card['pattern_cells'])) * 19))
    track = card['score_by_placed_count']
    result.extend([card['cube_count'] / MAX_CUBES] + [v / 30 for v in track])
    result.extend([0.] * (MAX_CUBES + 1 - len(track)))
    return tuple(result)


CARD_VECTORS = {i: _card_vector(i) for i in CARDS}
EMPTY_CARD = (0.,) * len(CARD_VECTORS[1])


def index(obs, action, version, allow_stop=False):
    if version not in COUNTS or obs['task_id'] != 'harmonies':
        raise ValueError('unsupported Harmonies action encoding')
    if action == STOP:
        if not allow_stop:
            raise InvalidAction('STOP is disabled in this protocol')
        return COUNTS[version]
    if version == 'legacy4564_v1':
        return legacy_id('harmonies', action)
    if action['type'] == 'place_animal':
        card = action['instance_id'] - 1
        if version == 'slots240_v1':
            card = next((i for i, c in enumerate(obs['active_cards'])
                         if c['instance_id'] == action['instance_id']), None)
            if card is None:
                raise InvalidAction('animal is not in the observation slot map')
        return 144 + card * 23 + action['anchor_cell_id']
    if action['type'] == 'end_turn':
        slot = action.get('discard_card_slot')
        return COUNTS[version] - 4 + (0 if slot is None else slot + 1)
    return legacy_id('harmonies', action)


def action_map(obs, version, allow_stop=False, episode_finished=False):
    if episode_finished or obs['terminated']:
        return {}
    mapping = {}
    for action in obs['legal_actions']:
        code = index(obs, action, version, allow_stop)
        if code in mapping or not 0 <= code < COUNTS[version]:
            raise ValueError('action mapping collision or out-of-range index')
        mapping[code] = action
    if allow_stop:
        mapping[COUNTS[version]] = dict(STOP)
    return mapping


def decode(obs, action_index, version, allow_stop=False, episode_finished=False):
    if isinstance(action_index, bool) or not isinstance(action_index, Integral):
        raise InvalidAction('action index must be an integer')
    mapping = action_map(obs, version, allow_stop, episode_finished)
    if action_index not in mapping:
        raise InvalidAction('masked action index')
    return dict(mapping[action_index])


def encode(obs, version='slots240_v1', observation_version='cards_v1', allow_stop=True,
           *, remaining_seconds=None, time_budget_seconds=None, episode_finished=False):
    if observation_version not in OBSERVATIONS:
        raise ValueError('unknown observation encoding')
    if version == 'slots240_v1' and observation_version == 'legacy_v1':
        raise ValueError('slot actions require matching slot observations')
    x = list(features(obs))
    active = obs['active_cards']
    if observation_version != 'legacy_v1':
        for slot in range(4):
            c = active[slot] if slot < len(active) else None
            x.extend(onehot(c['card_id'] if c else 0, 33))
            x.append(c['placed_count'] / MAX_CUBES if c else 0.)
    if observation_version == 'cards_v1':
        ids = [c['card_id'] for c in active] + [None] * (4 - len(active)) + obs['animal_market']
        for card_id in ids:
            x.extend(CARD_VECTORS.get(card_id, EMPTY_CARD))
    has_time = remaining_seconds is not None and time_budget_seconds is not None
    if has_time and (not math.isfinite(remaining_seconds) or not math.isfinite(time_budget_seconds)
                     or time_budget_seconds <= 0):
        raise ValueError('invalid time conditioning')
    x.extend([float(has_time), math.log1p(max(0., remaining_seconds)) if has_time else 0.,
              max(0., min(1., remaining_seconds / time_budget_seconds)) if has_time else 0.])
    mapping = action_map(obs, version, allow_stop, episode_finished)
    mask = tuple(i in mapping for i in range(COUNTS[version] + int(allow_stop)))
    return {'features': tuple(x), 'action_mask': mask}
