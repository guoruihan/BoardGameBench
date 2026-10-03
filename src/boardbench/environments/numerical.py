"""Versioned finite float32-compatible features and stable masked action IDs."""
from numbers import Integral

from boardbench.contracts import InvalidAction
from . import TASKS
from .harmonies import COLORS, CARDS
from .take_it_easy import TILES

ENCODING_VERSION = "board_features_v1"
ACTION_COUNTS = {"micro_tiles": 18, "take_it_easy": 19, "harmonies": 4564}


def onehot(value, size):
    return [float(i == value) for i in range(size)]


def action_id(task_id, action):
    if task_id == "micro_tiles":
        return action["offer_index"]*9+action["cell_id"]
    if task_id == "take_it_easy":
        return action["cell_id"]
    kind = action["type"]
    if kind == "choose_bundle":
        return action["bundle_id"]
    if kind == "place_token":
        return 3+COLORS.index(action["color"])*23+action["cell_id"]
    if kind == "take_card":
        return 141+action["market_slot"]
    if kind == "place_animal":
        return 144+((action["instance_id"]-1)*23+action["anchor_cell_id"])*6+action["orientation"]
    slot = action.get("discard_card_slot")
    return 4560+(0 if slot is None else slot+1)


def features(obs):
    task = obs["task_id"]
    if task == "micro_tiles":
        x = [v for color in obs["board"] for v in onehot(color+1, 4)]
        for index in range(2):
            x += onehot(obs["offers"][index]+1 if obs["offers"] else 0, 4)
        x += [obs["placed_count"]/9]
    elif task == "take_it_easy":
        x = []
        for tile in obs["board"]+[obs["current_tile"]]:
            x += ([0., 0., 0., 0.] if tile is None or tile == -1
                  else [1., *[v/9 for v in TILES[tile]]])
        x += [float(left) for left in obs["remaining_tiles"]]
        x += [obs["placed_count"]/19]
    else:
        x = [v for cell in obs["board"] for v in
             (onehot(cell["stack_id"], 15)+[float(cell["animal"] is not None), (cell["animal"] or 0)/32])]
        for group in obs["token_market"]:
            x += [(group or []).count(c)/3 for c in COLORS]
        x += [obs["pending_tokens"][c]/3 for c in COLORS]
        x += [obs["remaining_tokens"][c]/23 for c in COLORS]
        x += [obs["discarded_tokens"][c]/23 for c in COLORS]
        for card in obs["animal_market"]:
            x += onehot(card or 0, 33)
        owned = {c["card_id"]: c for c in obs["active_cards"]+obs["completed_cards"]}
        for card_id in range(1, 33):
            card = owned.get(card_id)
            x += [float(card is not None), card["placed_count"]/CARDS[card_id]["cube_count"] if card else 0.,
                  float(card_id in obs["remaining_cards"])]
        x += [obs["turn_index"]/13, float(obs["bundle_taken"]), float(obs["card_taken"])]
    return tuple(x+[float(obs["terminated"])])


def encode(obs):
    mask = [False]*ACTION_COUNTS[obs["task_id"]]
    for action in obs["legal_actions"]:
        mask[action_id(obs["task_id"], action)] = True
    return {"features": features(obs), "action_mask": tuple(mask)}


def decode(obs, index):
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise InvalidAction("action ID must be an integer scalar")
    for action in obs["legal_actions"]:
        if action_id(obs["task_id"], action) == index:
            return action
    raise InvalidAction("masked or out-of-range action ID")


class BoardRL:
    encoding_version = ENCODING_VERSION
    dtype = "float32"

    def __init__(self, task_id, engine=None):
        self.engine = engine if engine is not None else TASKS[task_id].factory()
        self.action_count = ACTION_COUNTS[task_id]
        self.observation_shape = (len(features(self.engine.observe())),)

    def reset(self, *, seed=0):
        obs, info = self.engine.reset(seed)
        return encode(obs), info

    def step(self, index):
        result = self.engine.step(decode(self.engine.observe(), index))
        return encode(result.observation), result.reward, result.terminated, result.truncated, result.info
