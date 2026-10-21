"""Versioned public-only duo encoding; every market slot and both boards visible."""
import numpy as np

from boardbench.environments.harmonies_duo import player_view
from boardbench.v12.encoding import index, feature_vector
from boardbench.v129.encoding import pack
from boardbench.v134.plans import PLAN_DIM

ACTIONS = 244
BASE = 1346
FEATURES = 3*BASE + PLAN_DIM + 8
LAYOUT = 'duo_three_native_views_plan819_meta8_slots244_v1'


def action_map(obs):
    if obs['terminated']:
        return {}
    view = player_view(obs)
    result = {}
    for a in obs['legal_actions']:
        if a['type'] == 'choose_bundle' and a['bundle_id'] >= 3:
            code = 240 + a['bundle_id']-3
        elif a['type'] == 'take_card' and a['market_slot'] >= 3:
            code = 242 + a['market_slot']-3
        else:
            code = index(view, a, 'slots240_v1')
        if code in result or not 0 <= code < ACTIONS or code in (237, 238, 239):
            raise ValueError('duo action collision/forbidden solo discard')
        result[code] = a
    return result


def views(obs):
    seat = obs['active_player']
    return [player_view(obs, seat, (0, 1, 2)), player_view(obs, seat, (3, 4, 0)),
            player_view(obs, 1-seat, (0, 1, 2))]


def meta(obs):
    seat = obs['active_player']; own, other = obs['players'][seat], obs['players'][1-seat]
    score, opposing = obs['score_breakdown'][seat], obs['score_breakdown'][1-seat]
    return np.array([seat == obs['first_player'], obs['final_round'], own['completed_turns']/20,
                     other['completed_turns']/20, score['total']/150, opposing['total']/150,
                     score['animals']/100, opposing['animals']/100], np.float32)


def encode_batch(observations, plans, native=None):
    projected = [v for obs in observations for v in views(obs)]
    base = (native.encode(pack(projected)) if native else
            np.asarray([feature_vector(v) for v in projected], np.float32))
    base = base.reshape(len(observations), 3*BASE)
    return np.concatenate((base, np.asarray(plans, np.float32),
                           np.stack([meta(o) for o in observations])), -1)
