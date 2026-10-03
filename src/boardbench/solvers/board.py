"""Public-information baselines. Heuristics/pruning never change engine legality."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random

from boardbench.contracts import Solver
from boardbench.environments import TASKS
from boardbench.environments.board import tuples
from boardbench.environments.micro import EDGES, LINES as MICRO_LINES
from boardbench.environments.take_it_easy import LINES, TILES
from boardbench.environments.harmonies import CARDS, PATTERNS, STACKS, STACK_INDEX


def micro_value(board):
    edges = sum(board[i] >= 0 and board[i] == board[j] for i, j in EDGES)
    lines = 0.
    for line in MICRO_LINES:
        colors = [board[i] for i in line if board[i] >= 0]
        if colors and len(set(colors)) == 1:
            lines += 3*(len(colors)/3)**2
    return edges+lines


def tie_value(board):
    total = 0.
    for line in LINES:
        values = [TILES[board[i]][line["axis"]] for i in line["cells"] if board[i] != -1]
        if values and len(set(values)) == 1:
            total += values[0]*len(line["cells"])*(len(values)/len(line["cells"]))**2
    return total


def landscape_value(engine):
    # Intermediate stack potential is a solver feature, not a game reward.
    return engine.score_breakdown()["total"] + sum({6: .7, 7: 1.8, 11: 1., 3: .25, 4: .5}.get(c["stack_id"], 0.)
                                                   for c in engine.s["board"])


def habitat_potential(board, card_id):
    best = 0.
    for anchor, orientations in enumerate(PATTERNS[card_id]):
        if board[anchor]["animal"] is not None:
            continue
        for pattern in orientations.values():
            matched = 0
            for cell, allowed in pattern:
                sid = board[cell]["stack_id"]
                if sid in allowed:
                    matched += 1
                elif board[cell]["animal"] is not None or not any(
                        STACKS[target][:len(STACKS[sid])] == STACKS[sid] for target in allowed):
                    break
            else:
                best = max(best, matched/len(pattern))
    return best


def ranked_actions(obs):
    """Return every legal action with a deterministic, documented heuristic value."""
    task, actions = obs["task_id"], obs["legal_actions"]
    if task in ("micro_tiles", "take_it_easy"):
        scored = []
        for action in actions:
            board = list(obs["board"])
            board[action["cell_id"]] = (obs["offers"][action["offer_index"]] if task == "micro_tiles"
                                         else obs["current_tile"])
            scored.append((micro_value(board) if task == "micro_tiles" else tie_value(board), action))
        return sorted(scored, key=lambda x: x[0], reverse=True)
    engine = TASKS[task].factory.from_observation(obs, 0)
    baseline = landscape_value(engine)
    card_potential = {c: habitat_potential(obs["board"], c) for c in obs["animal_market"] if c is not None}
    scored = []
    for action in actions:
        kind = action["type"]
        if kind == "place_token":
            cell = engine.s["board"][action["cell_id"]]
            old = cell["stack_id"]
            cell["stack_id"] = STACK_INDEX[STACKS[old]+(action["color"],)]
            value = landscape_value(engine)-baseline
            cell["stack_id"] = old
        elif kind == "place_animal":
            card = next(c for c in obs["active_cards"] if c["instance_id"] == action["instance_id"])
            track, n = CARDS[card["card_id"]]["score_by_placed_count"], card["placed_count"]
            value = track[n+1]-track[n]+1.
        elif kind == "take_card":
            value = .2+card_potential[obs["animal_market"][action["market_slot"]]]
        elif kind == "choose_bundle":
            value = sum({"gray": .6, "blue": 1., "brown": .7, "green": 1.3,
                         "yellow": .8, "red": .5}[c] for c in obs["token_market"][action["bundle_id"]])
        else:
            slot = action.get("discard_card_slot")
            value = -1. if slot is None else -1.1-card_potential[obs["animal_market"][slot]]
        scored.append((value, action))
    return sorted(scored, key=lambda x: x[0], reverse=True)


class BoardSolver(Solver):
    def __init__(self, seed, method="random", width=4, rollouts=2, depth=4, max_simulation_steps=128):
        self.rng = random.Random(seed)
        if method not in ("random", "heuristic", "search"):
            raise ValueError("unknown board method")
        self.method = method
        self.width, self.rollouts, self.depth = int(width), int(rollouts), int(depth)
        self.max_simulation_steps = int(max_simulation_steps)
        if min(self.width, self.rollouts, self.depth, self.max_simulation_steps) < 1:
            raise ValueError("search budgets must be positive")
        self.last_search = {"simulation_steps": 0, "rollouts": 0}

    def decide(self, observation, context):
        actions = observation["legal_actions"]
        if not actions:
            raise ValueError("cannot decide at terminal state")
        if self.method == "random":
            groups = defaultdict(list)
            for action in actions:
                groups[action.get("type", "place")].append(action)
            return self.rng.choice(groups[self.rng.choice(list(groups))])
        ranked = ranked_actions(observation)
        if self.method == "heuristic":
            return ranked[0][1]
        if context.reference_simulator is None:
            raise ValueError("search requires reference_simulator capability")
        used = completed = 0
        choices = []
        # Common random numbers per candidate; budget fixed independently of score.
        seeds = [self.rng.getrandbits(63) for _ in range(self.rollouts)]
        candidate_count = min(self.width, len(ranked), self.max_simulation_steps // self.rollouts)
        if not candidate_count:
            raise ValueError("search budget smaller than rollout count")
        horizon = min(self.depth, self.max_simulation_steps // (candidate_count*self.rollouts))
        for _, action in ranked[:candidate_count]:
            returns = []
            for seed in seeds:
                branch = context.reference_simulator.fork(seed)
                current_action = action
                for index in range(horizon):
                    result = branch.step(current_action)
                    used += 1
                    if result.terminated:
                        break
                    if index+1 < horizon:
                        current_action = ranked_actions(result.observation)[0][1]
                obs = result.observation
                if obs["terminated"]:
                    value = obs["score"]
                elif obs["task_id"] == "micro_tiles":
                    value = micro_value(obs["board"])
                elif obs["task_id"] == "take_it_easy":
                    value = tie_value(obs["board"])
                else:
                    value = obs["score_breakdown"]["total"]
                returns.append(value)
                completed += 1
            choices.append((sum(returns)/len(returns), action))
        self.last_search = {"simulation_steps": used, "rollouts": completed}
        return max(choices, key=lambda x: x[0])[1]

    def save(self, directory):
        Path(directory, "board_solver.json").write_text(json.dumps({"method": self.method,
            "rng": self.rng.getstate(), "width": self.width, "rollouts": self.rollouts,
            "depth": self.depth, "max_simulation_steps": self.max_simulation_steps}))

    def load(self, directory):
        state = json.loads(Path(directory, "board_solver.json").read_text())
        self.rng.setstate(tuples(state.pop("rng")))
        for key, value in state.items():
            setattr(self, key, value)
