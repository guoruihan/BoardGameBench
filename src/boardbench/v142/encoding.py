"""Public own-board encoding; zero opponent block preserves weight compatibility."""
import numpy as np

from boardbench.v12.encoding import feature_vector, index
from boardbench.v129.encoding import pack
from boardbench.v140.encoding import BASE, FEATURES, ACTIONS

LAYOUT = 'market_two_native_views_zero_opponent_plan819_meta8_slots244_v1'


def view(obs, slots=(0, 1, 2)):
    return {**obs, 'task_id': 'harmonies', 'rules_version': 'harmonies_solo_a_v1',
            'token_market': [obs['token_market'][i] for i in slots],
            'animal_market': [obs['animal_market'][i] for i in slots]}


def action_map(obs):
    if obs['terminated']: return {}
    projected = view(obs); result = {}
    for action in obs['legal_actions']:
        if action['type'] == 'choose_bundle' and action['bundle_id'] >= 3:
            code = 240 + action['bundle_id']-3
        elif action['type'] == 'take_card' and action['market_slot'] >= 3:
            code = 242 + action['market_slot']-3
        else:
            code = index(projected, action, 'slots240_v1')
        if code in result or not 0 <= code < ACTIONS or code in (237, 238, 239):
            raise ValueError('curriculum action collision/forbidden discard')
        result[code] = action
    return result


def encode_batch(observations, plans, native=None):
    projected = [view(obs, slots) for obs in observations for slots in ((0, 1, 2), (3, 4, 0))]
    base = (native.encode(pack(projected)) if native else
            np.asarray([feature_vector(v) for v in projected], np.float32))
    meta = np.array([[1, o['turn_index'] == 13, o['completed_turns']/20, 0,
                      o['score_breakdown']['total']/150, 0, o['score_breakdown']['animals']/100, 0]
                     for o in observations], np.float32)
    return np.concatenate((base.reshape(len(observations), 2*BASE),
                           np.zeros((len(observations), BASE), np.float32),
                           np.asarray(plans, np.float32), meta), -1)
