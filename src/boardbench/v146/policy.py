"""Reuse solo geometry, never solo transitions; models retain public planner state."""
from copy import deepcopy

import numpy as np
import torch

from boardbench.environments.harmonies import CARDS, STACKS, STACK_INDEX
from boardbench.environments.harmonies_duo import HarmoniesDuo, player_view
from boardbench.solvers.board import habitat_potential, landscape_value
from boardbench.v128.environment import FastHarmonies
from boardbench.v123.policies import rank_actions
from boardbench.v125.policies import GoalContinuation, action_key
from boardbench.v138.baseline import Planner
from boardbench.v140.encoding import ACTIONS, action_map, encode_batch


def personal_view(obs):
    if obs['rules_version'] != HarmoniesDuo.rules_version:
        raise ValueError('official duo required')
    view = player_view(obs)
    view['legal_actions'] = deepcopy(obs['legal_actions'])
    return view


def terrain_rank(view):
    # This object is used ONLY for the identical personal-board score function.
    # No reset, step, fork or solo hidden supply is constructed here.
    engine = FastHarmonies.__new__(FastHarmonies)
    engine.s = deepcopy(view)
    baseline = landscape_value(engine)
    potentials = {c: habitat_potential(view['board'], c) for c in view['animal_market'] if c is not None}
    ranked = []
    for action in view['legal_actions']:
        kind = action['type']
        if kind == 'place_token':
            cell = engine.s['board'][action['cell_id']]; old = cell['stack_id']
            cell['stack_id'] = STACK_INDEX[STACKS[old] + (action['color'],)]
            value = landscape_value(engine) - baseline; cell['stack_id'] = old
        elif kind == 'place_animal':
            card = next(c for c in view['active_cards'] if c['instance_id'] == action['instance_id'])
            track = CARDS[card['card_id']]['score_by_placed_count']; n = card['placed_count']
            value = track[n+1] - track[n] + 1.
        elif kind == 'take_card':
            value = .2 + potentials[view['animal_market'][action['market_slot']]]
        elif kind == 'choose_bundle':
            value = sum({'gray': .6, 'blue': 1., 'brown': .7, 'green': 1.3, 'yellow': .8, 'red': .5}[c]
                        for c in view['token_market'][action['bundle_id']])
        elif action == {'type': 'end_turn'}:
            value = -1.
        else:
            raise ValueError('unexpected duo action')
        ranked.append((value, action))
    return sorted(ranked, key=lambda pair: -pair[0])


class Habitat:
    def __init__(self):
        self.goals = GoalContinuation(compiled=True, capacity=2048)

    def rank(self, obs):
        view = personal_view(obs)
        return rank_actions(view, 'habitat', goal_function=self.goals.goals,
                            base_ranking=terrain_rank(view),
                            future_turns=sum(view['remaining_tokens'].values()) // 6)

    def decide(self, obs):
        return dict(self.rank(obs)[0][1])


class ModelPlayer:
    def __init__(self, model, native=None):
        self.model, self.native = model, native

    def rank_batch(self, observations, planners):
        if len(observations) != len(planners) or not observations:
            raise ValueError('one isolated planner per observation')
        maps = [action_map(obs) for obs in observations]
        plans = [p.vector(player_view(obs)) for p, obs in zip(planners, observations)]
        x = encode_batch(observations, plans, self.native)
        mask = np.asarray([[i in m for i in range(ACTIONS)] for m in maps])
        with torch.inference_mode():
            logits, values = self.model(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(mask))
        raw = logits.numpy()
        # Descending score, ascending action code exactly matches torch.argmax.
        rankings = [[dict(m[i]) for i in sorted(m, key=lambda i: (-raw[j, i], i))]
                    for j, m in enumerate(maps)]
        return rankings, x, mask, values.numpy()

    def decide(self, obs, planner):
        return self.rank_batch([obs], [planner])[0][0][0]


def mixed_roots(obs, model_ranking, habitat, width=6):
    streams = [model_ranking, [a for _, a in terrain_rank(personal_view(obs))],
               [a for _, a in habitat.rank(obs)]]
    result, seen = [], set()
    for rank in range(max(map(len, streams))):
        for stream in streams:
            if rank >= len(stream):
                continue
            a = stream[rank]; key = action_key(a)
            if key not in seen:
                result.append(dict(a)); seen.add(key)
            if len(result) == width:
                return result
    return result
