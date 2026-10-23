"""Own-score MCTS over public outcomes, with exact stochastic duo transitions.

Every simulation resamples a future from the public remaining inventory. Edges
branch by the next *observed* state, never by the private sampled bag/deck. The
opponent follows a frozen policy; multiple own atomic actions do not flip signs.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import time

import numpy as np
import torch

from boardbench.environments.harmonies_duo import FastHarmoniesDuo, player_view
from boardbench.v140.encoding import ACTIONS, action_map, encode_batch
from boardbench.v146.policy import Habitat, personal_view, terrain_rank
from boardbench.v125.policies import action_key


def public_key(obs):
    # Include all public fields. Do not merge different planner histories across
    # parents: outcome dictionaries belong to a particular action/history edge.
    return hashlib.sha256(json.dumps(obs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def policy_evaluation(player, obs, planner):
    mapping = action_map(obs)
    vector = planner.vector(player_view(obs))
    x = encode_batch([obs], [vector], player.native)
    mask = np.asarray([[i in mapping for i in range(ACTIONS)]])
    device = next(player.model.parameters()).device
    with torch.inference_mode():
        logits, value = player.model(torch.as_tensor(x, dtype=torch.float64, device=device),
                                    torch.as_tensor(mask, device=device))
        probabilities = logits.softmax(-1)[0].cpu().numpy()
    ranking = [dict(mapping[i]) for i in sorted(mapping, key=lambda i: (-probabilities[i], i))]
    return ranking, {action_key(a): float(probabilities[i]) for i, a in mapping.items()}, float(value[0])


@dataclass
class Edge:
    action: dict
    prior: float
    visits: int = 0
    total: float = 0.
    squares: float = 0.
    outcomes: dict = field(default_factory=dict)

    @property
    def mean(self):
        return self.total / self.visits if self.visits else None


@dataclass
class Node:
    actions: list
    priors: dict
    visits: int = 0
    edges: list = field(default_factory=list)

    def select(self, initial_width, widening, exploration):
        allowed = min(len(self.actions), max(initial_width, int(widening * math.sqrt(self.visits + 1))))
        if len(self.edges) < allowed:
            action = self.actions[len(self.edges)]
            edge = Edge(action, self.priors[action_key(action)])
            self.edges.append(edge)
            return edge
        return max(self.edges, key=lambda e: (e.mean / 150. if e.visits else 0.)
                   + exploration * e.prior * math.sqrt(self.visits + 1) / (1 + e.visits))


class Search:
    def __init__(self, focal, opponent, seat, continuation='mcts', seconds=900., seed=0,
                 *, max_rounds=None, max_depth=16, initial_width=6, widening=1.5,
                 exploration=.6, clock=time.monotonic, value_evaluator=None,
                 value_fraction=.75, min_value_depth=4):
        if (seat not in (0, 1) or continuation not in ('mcts', 'mcts_wide', 'mcts_value')
                or not math.isfinite(seconds) or not 0 < seconds <= 900
                or not 1 <= initial_width <= 32 or not 1 <= max_depth <= 128
                or not math.isfinite(widening) or widening <= 0
                or not math.isfinite(exploration) or exploration <= 0
                or not 0 <= value_fraction <= 1 or min_value_depth < 1
                or (max_rounds is not None and max_rounds < 1)):
            raise ValueError('invalid bounded public MCTS configuration')
        if continuation == 'mcts_value' and value_evaluator is None:
            raise ValueError('validated search value evaluator required; old PPO critic is not a leaf oracle')
        self.players = [focal if i == seat else opponent for i in (0, 1)]
        self.seat, self.variant = seat, continuation
        self.seconds, self.clock, self.max_rounds = seconds, clock, max_rounds
        self.max_depth, self.initial_width = max_depth, initial_width
        self.widening = widening * (2 if continuation == 'mcts_wide' else 1)
        self.exploration, self.value_evaluator = exploration, value_evaluator
        self.value_fraction, self.min_value_depth = value_fraction, min_value_depth
        self.rng = random.Random(seed)
        self.habitat = Habitat()
        self.spent, self.horizon, self.last = 0., 75., {}
        self.steps = self.nodes = self.max_reached_depth = self.max_reached_turns = 0
        self.terminal_leaves = self.value_leaves = 0

    def node(self, obs, plans):
        model, probabilities, _ = policy_evaluation(self.players[self.seat], obs, plans[self.seat])
        streams = [model, [a for _, a in self.habitat.rank(obs)],
                   [a for _, a in terrain_rank(personal_view(obs))]]
        # Keep the established mixed candidate ordering, then progressively admit
        # ALL legal actions. Nonzero uniform mass prevents zero-prior starvation.
        actions, seen = [], set()
        prior = {action_key(a): .3 * probabilities[action_key(a)] + .1 / len(model) for a in model}
        for stream in streams[1:]:
            weights = [1 / (i + 1)**1.5 for i in range(len(stream))]
            denominator = sum(weights)
            for action, weight in zip(stream, weights):
                prior[action_key(action)] += .3 * weight / denominator
        for rank in range(len(model)):
            for stream in streams:
                key = action_key(stream[rank])
                if key not in seen:
                    actions.append(dict(stream[rank])); seen.add(key)
        self.nodes += 1
        return Node(actions, prior)

    def step(self, game, action):
        self.steps += 1
        return game.step(action).observation

    def opponent_turn(self, game, obs, plans, deadline):
        for _ in range(768):
            if obs['terminated'] or obs['active_player'] == self.seat:
                return obs
            if self.clock() >= deadline:
                return None
            action = self.players[1-self.seat].decide(obs, plans[1-self.seat])
            obs = self.step(game, action)
        raise RuntimeError('opponent safety bound')

    def rollout(self, game, obs, plans, deadline):
        for _ in range(768):
            if self.clock() >= deadline:
                return None
            if obs['terminated']:
                self.terminal_leaves += 1
                return float(obs['scores'][self.seat])
            seat = obs['active_player']
            action = (self.habitat.decide(obs) if seat == self.seat
                      else self.players[seat].decide(obs, plans[seat]))
            obs = self.step(game, action)
        raise RuntimeError('rollout safety bound; partial scores cannot be targets')

    def leaf(self, game, obs, plans, depth, deadline):
        if self.clock() >= deadline:
            return None
        if obs['terminated']:
            self.terminal_leaves += 1
            return float(obs['scores'][self.seat])
        if (self.variant == 'mcts_value' and depth >= self.min_value_depth
                and self.rng.random() < self.value_fraction):
            value = self.value_evaluator(obs, plans)
            if not math.isfinite(value):
                raise ValueError('nonfinite leaf score')
            if self.clock() >= deadline:
                return None
            self.value_leaves += 1
            return value
        return self.rollout(game, obs, plans, deadline)

    def simulate(self, root, observation, planners, deadline, future_seed):
        game = FastHarmoniesDuo.from_observation(observation, future_seed)
        obs, plans, node, path = observation, deepcopy(planners), root, []
        for depth in range(1, self.max_depth + 1):
            if self.clock() >= deadline:
                return False
            edge = node.select(self.initial_width, self.widening, self.exploration)
            path.append((node, edge))
            obs = self.step(game, edge.action)
            obs = self.opponent_turn(game, obs, plans, deadline)
            if obs is None:
                return False
            if self.clock() >= deadline:
                return False
            self.max_reached_depth = max(self.max_reached_depth, depth)
            self.max_reached_turns = max(self.max_reached_turns,
                obs['players'][self.seat]['completed_turns'] - observation['players'][self.seat]['completed_turns'])
            if obs['terminated'] or depth == self.max_depth:
                value = self.leaf(game, obs, plans, depth, deadline)
                break
            key = public_key(obs)
            child = edge.outcomes.get(key)
            if child is None:
                child = self.node(obs, plans)
                edge.outcomes[key] = child
                value = self.leaf(game, obs, plans, depth, deadline)
                break
            # Cached rankings do not skip updates to simulated public memory.
            plans[self.seat].vector(player_view(obs))
            node = child
        if value is None or self.clock() >= deadline:
            return False  # Nothing on the partial path is backed up.
        for node, edge in path:
            node.visits += 1
            edge.visits += 1
            edge.total += value
            edge.squares += value * value
        return True

    def decide(self, obs, planners):
        if obs['active_player'] != self.seat or obs['terminated']:
            raise ValueError('live focal observation required')
        begin = self.clock()
        remaining = max(0., self.seconds - self.spent)
        allowance = max(0., remaining - .03*self.horizon) * min(1., 2/max(2., self.horizon))
        deadline = begin + allowance
        self.steps = self.nodes = self.max_reached_depth = self.max_reached_turns = 0
        self.terminal_leaves = self.value_leaves = 0
        # Cheap policy fallback remains charged to the cumulative budget.
        if allowance <= .01 or len(obs['legal_actions']) == 1:
            ranking, _, _ = policy_evaluation(self.players[self.seat], obs, planners[self.seat])
            root = Node(ranking, {})
            deadline = begin
        else:
            root = self.node(obs, planners)
        completed = attempts = 0
        while (len(root.actions) > 1 and self.clock() < deadline
               and (self.max_rounds is None or attempts < self.max_rounds)):
            attempts += 1
            if self.simulate(root, obs, planners, deadline, self.rng.getrandbits(63)):
                completed += 1
        visited = [e for e in root.edges if e.visits]
        chosen = max(visited, key=lambda e: (e.visits, e.mean)).action if visited else root.actions[0]
        elapsed = self.clock() - begin
        self.spent += elapsed
        self.horizon = max(1., self.horizon - 1)
        self.last = dict(variant=self.variant, algorithm='public_outcome_mcts', completed_rounds=completed,
            simulation_attempts=attempts, simulation_steps=self.steps, expanded_nodes=self.nodes,
            max_tree_depth=self.max_reached_depth, max_turn_depth=self.max_reached_turns,
            terminal_leaves=self.terminal_leaves, value_leaves=self.value_leaves,
            decision_seconds=elapsed, cumulative_decision_seconds=self.spent,
            forced_action=len(root.actions)==1, fallback=not visited and len(root.actions)>1,
            public_future_only=True, value_target='absolute_own_terminal_score',
            root_visits=root.visits, legal_action_count=len(root.actions),
            candidates=[dict(action=e.action, samples=e.visits, visits=e.visits, prior=e.prior,
                mean=e.mean, variance=max(0., (e.squares-e.total**2/e.visits)/(e.visits-1)) if e.visits>1 else None)
                for e in root.edges])
        return dict(chosen)
