"""El Dorado (2017), official first route, two explorers per player.

The environment exposes actual choices: card commitment, individual movement,
post-draw trashing, buying and keeping cards. Deck orders stay in snapshots only.
"""
from collections import Counter, deque
from copy import deepcopy
from itertools import combinations
import heapq
import math

from boardbench.families.base import DuoGame
from boardbench.families.data.el_dorado import (
    BLOCKADES, CARDS, FINISH_LOCAL, INITIAL_MARKET, MARKET_CARDS, ROUTE,
    STARTER, TILE_ROWS,
)

DIRECTIONS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


def card_type(card):
    return card.split(":")[0]


def card_data(card):
    return CARDS[card_type(card)]


def _make_board():
    cells, starts = [], {}
    for tile, rotation, (oq, or_) in ROUTE:
        for row, tokens in enumerate(TILE_ROWS[tile]):
            r = row - 3
            for col, token in enumerate(tokens.split()):
                q, rr = max(-3, -3-r) + col, r
                for _ in range(rotation):
                    q, rr = -rr, q + rr
                color = token[0]
                cell = dict(tile=tile, q=q+oq, r=rr+or_, color=color,
                            cost=0 if color in ("s", "x") else int(token[1]), finish=False)
                if color == "s":
                    starts[int(token[1])] = len(cells)
                    cell.update(color="g", start=int(token[1]))
                cells.append(cell)
    oq, or_ = ROUTE[-1][2]
    for q, r in FINISH_LOCAL:
        cells.append(dict(tile="finish", q=q+oq, r=r+or_, color="b", cost=1, finish=True))
    coords = {(c["q"], c["r"]): i for i, c in enumerate(cells)}
    assert len(coords) == len(cells) == 188
    neighbors, seams = [], {}
    pairs = [(ROUTE[i][0], ROUTE[i+1][0]) for i in range(4)]
    for i, c in enumerate(cells):
        ns = [coords[q, r] for dq, dr in DIRECTIONS
              if (q := c["q"]+dq, r := c["r"]+dr) in coords]
        neighbors.append(tuple(ns))
        for j in ns:
            t = cells[j]["tile"]
            if c["tile"] != t and t != "finish" and c["tile"] != "finish":
                pair = (c["tile"], t)
                assert pair in pairs or pair[::-1] in pairs
                seams[i, j] = pairs.index(pair if pair in pairs else pair[::-1])
    return tuple(cells), tuple(neighbors), seams, starts


BOARD, NEIGHBORS, SEAMS, STARTS = _make_board()
FINISHES = tuple(i for i, c in enumerate(BOARD) if c["finish"])


def _distances():
    # Public, static terrain-distance heuristic, not an official game score.
    d = [math.inf] * len(BOARD)
    heap = []
    for i in FINISHES:
        d[i] = 0.
        heapq.heappush(heap, (0., i))
    while heap:
        cost, i = heapq.heappop(heap)
        if cost != d[i]:
            continue
        for j in NEIGHBORS[i]:
            if BOARD[j]["color"] == "x":
                continue
            entry = BOARD[i]["cost"]
            edge = max(.8, entry) + (0.6 if BOARD[i]["color"] == "r" else 0)
            if cost+edge < d[j]:
                d[j] = cost+edge
                heapq.heappush(heap, (d[j], j))
    return tuple(d)


DISTANCE = _distances()


def _pool():
    counters = Counter()
    players = []
    for _ in (0, 1):
        cards = []
        for name in STARTER:
            cards.append(f"{name}:{counters[name]}")
            counters[name] += 1
        players.append(cards)
    market = {name: [f"{name}:{i}" for i in range(3)] for name in MARKET_CARDS}
    return players, market


STARTER_IDS, MARKET_IDS = _pool()
ALL_IDS = frozenset(c for deck in STARTER_IDS for c in deck) | frozenset(c for pile in MARKET_IDS.values() for c in pile)


class ElDorado(DuoGame):
    game_id = "el_dorado"
    config_id = "2017_first_route_duo_no_caves"

    def _reset(self, first_player):
        players = []
        for p in (0, 1):
            deck = list(STARTER_IDS[p])
            self.rng.shuffle(deck)
            slots = (1, 3) if p == first_player else (2, 4)
            players.append(dict(deck=deck, hand=[], discard=[], used=[],
                                pawns=[STARTS[x] for x in slots], blockades=[], fresh=[]))
        blocks = deepcopy(BLOCKADES)
        self.rng.shuffle(blocks)
        self.state.update(players=players, market=deepcopy(MARKET_IDS),
                          market_slots=list(INITIAL_MARKET), blockades=blocks[:4],
                          unused_blockades=blocks[4:], removed=[], phase="play", active=None,
                          money2=0, bought=False, trash_remaining=0, return_phase=None,
                          turns=[0, 0], finish_order=[], final_round=False, history=[])
        for p in (0, 1):
            self._draw(p, 4)
        self.validate()

    def _observe(self, player):
        s = self.state
        people = []
        for i, p in enumerate(s["players"]):
            public = dict(pawns=list(p["pawns"]), blockades=list(p["blockades"]),
                          discard=list(p["discard"]), used=list(p["used"]),
                          hand_count=len(p["hand"]), deck_count=len(p["deck"]),
                          fresh=list(p["fresh"]))
            if i == player:
                public["hand"] = list(p["hand"])
            people.append(public)
        return dict(phase=s["phase"], players=people, market_slots=s["market_slots"],
                    market={k: len(v) for k, v in s["market"].items()},
                    blockades=s["blockades"], removed=s["removed"], active=s["active"],
                    money2=s["money2"], bought=s["bought"],
                    trash_remaining=s["trash_remaining"], turns=s["turns"],
                    first_player=s["first_player"], finish_order=s["finish_order"],
                    final_round=s["final_round"], board=BOARD, history=s["history"])

    def scores(self):
        return [None, None]

    def score_breakdown(self):
        return [dict(total=None, arrived=sum(x is None for x in p["pawns"]),
                     blockades=len(p["blockades"])) for p in self.state["players"]]

    def _draw(self, player, n):
        p = self.state["players"][player]
        for _ in range(n):
            if not p["deck"]:
                if not p["discard"]:
                    break
                p["deck"] = p["discard"]
                p["discard"] = []
                self.rng.shuffle(p["deck"])
                self.state["history"].append(dict(event="shuffle", player=player,
                                                   count=len(p["deck"])))
            p["hand"].append(p["deck"].pop())

    def _available(self):
        p = self.state["players"][self.current_player]
        return [c for c in p["hand"] if c not in p["fresh"]]

    def _occupied(self):
        return {x for p in self.state["players"] for x in p["pawns"] if x is not None}

    def _block_at(self, src, dst):
        index = SEAMS.get((src, dst))
        return index if index is not None and self.state["blockades"][index] is not None else None

    def _touching_blocks(self, pos):
        return sorted({b for j in NEIGHBORS[pos] if (b := self._block_at(pos, j)) is not None})

    def _colored_targets(self, pos, color, power):
        occupied = self._occupied()
        targets = [j for j in NEIGHBORS[pos]
                   if j not in occupied and self._block_at(pos, j) is None
                   and BOARD[j]["color"] == color and BOARD[j]["cost"] <= power]
        blocks = [b for b in self._touching_blocks(pos)
                  if self.state["blockades"][b]["color"] == color
                  and self.state["blockades"][b]["cost"] <= power]
        return targets, blocks

    def _special_card_actions(self, hand):
        out, s = [], self.state
        occupied = self._occupied()
        for c in hand:
            d = card_data(c)
            if d["effect"] in ("draw", "transmitter"):
                out.append(dict(type="play_action", card=c))
            elif d["effect"] == "native":
                for pawn, pos in enumerate(s["players"][self.current_player]["pawns"]):
                    if pos is None:
                        continue
                    for j in NEIGHBORS[pos]:
                        if j not in occupied and BOARD[j]["color"] != "x" and self._block_at(pos, j) is None:
                            out.append(dict(type="native_move", card=c, pawn=pawn, to=j))
                    for b in self._touching_blocks(pos):
                        out.append(dict(type="native_blockade", card=c, pawn=pawn, blockade=b))
        return out

    def _legal_actions(self):
        s, p = self.state, self.state["players"][self.current_player]
        phase, hand = s["phase"], self._available()
        if phase == "move":
            a = s["active"]
            out = [dict(type="stop_move")]
            pos = p["pawns"][a["pawn"]]
            targets, blocks = self._colored_targets(pos, a["color"], a["power"])
            out += [dict(type="move", to=j) for j in targets]
            out += [dict(type="remove_blockade", blockade=b) for b in blocks]
            return out
        if phase == "trash":
            return [dict(type="finish_effect")] + [dict(type="trash", card=c) for c in p["hand"]]
        if phase == "transmitter":
            return [dict(type="gain", card=name) for name, pile in s["market"].items() if pile]
        if phase == "discard":
            return [dict(type="refill")] + [dict(type="discard", card=c) for c in p["hand"]]
        out = self._special_card_actions(hand)
        out.append(dict(type="end_turn"))
        if phase == "buy":
            if not s["bought"]:
                for c in hand:
                    d = card_data(c)
                    if d["color"] in ("y", "w"):
                        out.append(dict(type="spend", card=c, mode="coin"))
                        if d["item"]:
                            out.append(dict(type="spend", card=c, mode="half"))
                    else:
                        out.append(dict(type="spend", card=c, mode="half"))
                vacant = [i for i, name in enumerate(s["market_slots"]) if name is None]
                for name, pile in s["market"].items():
                    if not pile or 2*CARDS[name]["price"] > s["money2"]:
                        continue
                    if name in s["market_slots"]:
                        out.append(dict(type="buy", card=name, slot=s["market_slots"].index(name)))
                    else:
                        out += [dict(type="buy", card=name, slot=i) for i in vacant]
            return out
        assert phase == "play"
        out.append(dict(type="start_buy"))
        occupied = self._occupied()
        for pawn, pos in enumerate(p["pawns"]):
            if pos is None:
                continue
            for c in hand:
                d = card_data(c)
                colors = ("g", "b", "y") if d["color"] == "w" else (d["color"],)
                for color in colors:
                    if color not in ("g", "b", "y"):
                        continue
                    targets, blocks = self._colored_targets(pos, color, d["power"])
                    if targets or blocks:
                        out.append(dict(type="play_move", card=c, pawn=pawn, color=color))
            for j in NEIGHBORS[pos]:
                d = BOARD[j]
                if j in occupied or self._block_at(pos, j) is not None or d["color"] not in ("r", "c"):
                    continue
                out += [dict(type="pay_move", pawn=pawn, to=j, cards=list(cs))
                        for cs in combinations(sorted(hand), d["cost"])]
            for b in self._touching_blocks(pos):
                d = s["blockades"][b]
                if d["color"] == "r":
                    out += [dict(type="pay_blockade", pawn=pawn, blockade=b, cards=list(cs))
                            for cs in combinations(sorted(hand), d["cost"])]
        return out

    def _use(self, card, printed=False, trash=False):
        p = self.state["players"][self.current_player]
        p["hand"].remove(card)
        removed = trash or (printed and card_data(card)["item"])
        if removed:
            self.state["removed"].append(card)
        else:
            p["used"].append(card)
        self.state["history"].append(dict(event="play", player=self.current_player, card=card,
                                           removed=removed, printed=printed))

    def _acquire(self, name, slot=None):
        s = self.state
        card = s["market"][name].pop()
        p = s["players"][self.current_player]
        p["discard"].append(card)
        p["fresh"].append(card)
        if slot is not None:
            s["market_slots"][slot] = name if s["market"][name] else None
        if name in s["market_slots"] and not s["market"][name]:
            s["market_slots"][s["market_slots"].index(name)] = None
        s["history"].append(dict(event="acquire", player=self.current_player, card=card,
                                 normal_purchase=slot is not None))

    def _remove_blockade(self, index):
        s = self.state
        b = s["blockades"][index]
        s["players"][self.current_player]["blockades"].append(b["rank"])
        s["blockades"][index] = None
        s["history"].append(dict(event="blockade", player=self.current_player, rank=b["rank"], seam=index))

    def _move(self, pawn, target):
        s = self.state
        p = s["players"][self.current_player]
        p["pawns"][pawn] = None if BOARD[target]["finish"] else target
        if BOARD[target]["finish"] and all(x is None for x in p["pawns"]):
            s["finish_order"].append(self.current_player)
            s["final_round"] = True
            if self.current_player != s["first_player"]:
                s["turns"][self.current_player] += 1
                self._finish_race()

    def _finish_race(self):
        s = self.state
        finishers = s["finish_order"]
        best = max((len(s["players"][p]["blockades"]), max(s["players"][p]["blockades"], default=0))
                   for p in finishers)
        tied = [p for p in finishers if (len(s["players"][p]["blockades"]),
                max(s["players"][p]["blockades"], default=0)) == best]
        # Unique printed blockade ranks eliminate any nonzero residual tie.
        self.finish([tied[0]], reason="both_explorers_reached_city")

    def _end_turn(self):
        s = self.state
        who = self.current_player
        p = s["players"][who]
        self._draw(who, max(0, 4-len(p["hand"])))
        s["turns"][who] += 1
        if s["final_round"] and who != s["first_player"]:
            self._finish_race()
            return
        s.update(current_player=1-who, phase="play", active=None, money2=0, bought=False,
                 trash_remaining=0, return_phase=None)
        s["players"][1-who]["fresh"] = []

    def _step(self, action):
        s = self.state
        who, kind = self.current_player, action["type"]
        p = s["players"][who]
        if kind == "play_move":
            c = action["card"]
            self._use(c, printed=True)
            s.update(phase="move", active=dict(pawn=action["pawn"], color=action["color"], power=card_data(c)["power"]))
        elif kind == "move":
            a = s["active"]
            a["power"] -= BOARD[action["to"]]["cost"]
            self._move(a["pawn"], action["to"])
            if p["pawns"][a["pawn"]] is None:
                s.update(phase="play", active=None)
        elif kind == "remove_blockade":
            s["active"]["power"] -= s["blockades"][action["blockade"]]["cost"]
            self._remove_blockade(action["blockade"])
        elif kind == "stop_move":
            s.update(phase="play", active=None)
        elif kind in ("native_move", "native_blockade"):
            self._use(action["card"], printed=True)
            if kind == "native_move":
                self._move(action["pawn"], action["to"])
            else:
                self._remove_blockade(action["blockade"])
        elif kind in ("pay_move", "pay_blockade"):
            for c in action["cards"]:
                self._use(c, trash=kind == "pay_move" and BOARD[action["to"]]["color"] == "c")
            if kind == "pay_move":
                self._move(action["pawn"], action["to"])
            else:
                self._remove_blockade(action["blockade"])
        elif kind == "play_action":
            d = card_data(action["card"])
            self._use(action["card"], printed=True)
            if d["effect"] == "transmitter":
                if any(s["market"].values()):
                    s.update(return_phase=s["phase"], phase="transmitter")
            else:
                self._draw(who, d["draw"])
                if d["trash"] and p["hand"]:
                    s.update(return_phase=s["phase"], phase="trash", trash_remaining=d["trash"])
        elif kind == "gain":
            self._acquire(action["card"])
            s.update(phase=s["return_phase"], return_phase=None)
        elif kind == "trash":
            p["hand"].remove(action["card"])
            s["removed"].append(action["card"])
            s["history"].append(dict(event="trash", player=who, card=action["card"]))
            s["trash_remaining"] -= 1
            if not s["trash_remaining"] or not p["hand"]:
                s.update(phase=s["return_phase"], return_phase=None, trash_remaining=0)
        elif kind == "finish_effect":
            s.update(phase=s["return_phase"], return_phase=None, trash_remaining=0)
        elif kind == "start_buy":
            s["phase"] = "buy"
        elif kind == "spend":
            printed = action["mode"] == "coin"
            s["money2"] += card_data(action["card"])["power"]*2 if printed else 1
            self._use(action["card"], printed=printed)
        elif kind == "buy":
            s["money2"] -= 2*CARDS[action["card"]]["price"]
            s["bought"] = True
            self._acquire(action["card"], action["slot"])
        elif kind == "end_turn":
            p["discard"].extend(p["used"])
            p["used"] = []
            s["phase"] = "discard"
        elif kind == "discard":
            p["hand"].remove(action["card"])
            p["discard"].append(action["card"])
            s["history"].append(dict(event="discard", player=who, card=action["card"]))
        elif kind == "refill":
            self._end_turn()
        else:
            raise AssertionError(kind)

    def validate(self):
        super().validate()
        s = self.state
        cards = list(s["removed"])
        for pile in s["market"].values():
            cards.extend(pile)
        ranks = [b["rank"] for b in s["blockades"] + s["unused_blockades"] if b]
        positions = []
        for p in s["players"]:
            for zone in ("deck", "hand", "discard", "used"):
                cards.extend(p[zone])
            ranks.extend(p["blockades"])
            if len(p["pawns"]) != 2:
                raise ValueError("two explorers per player")
            for pos in p["pawns"]:
                if pos is not None:
                    if type(pos) is not int or not 0 <= pos < len(BOARD) or BOARD[pos]["color"] == "x" or BOARD[pos]["finish"]:
                        raise ValueError("invalid explorer position")
                    positions.append(pos)
        if len(cards) != 70 or set(cards) != ALL_IDS:
            raise ValueError("expedition card conservation")
        if sorted(ranks) != list(range(1, 7)):
            raise ValueError("blockade conservation")
        if len(positions) != len(set(positions)):
            raise ValueError("occupied explorer space")
        slots = [x for x in s["market_slots"] if x is not None]
        if len(s["market_slots"]) != 6 or len(slots) != len(set(slots)) or any(not s["market"][x] for x in slots):
            raise ValueError("invalid marketplace")
        if s["phase"] not in ("play", "buy", "move", "trash", "transmitter", "discard"):
            raise ValueError("invalid phase")
        if s["phase"] == "move":
            a = s["active"]
            if not a or a["power"] < 0 or s["players"][s["current_player"]]["pawns"][a["pawn"]] is None:
                raise ValueError("invalid active movement")
        if s["terminated"] and not s["finish_order"]:
            raise ValueError("natural race ending without finisher")

    @staticmethod
    def features(observation, action):
        return _features(observation, action)


def _card_value(name, collection):
    """Small public deck-building baseline. Weights are a tunable policy, not rules."""
    base = {"scout": 3.3, "trailblazer": 5.8, "pioneer": 8., "giant_machete": 3.,
            "photographer": 3.0, "journalist": 4.7, "millionaire": 5., "treasure_chest": 1.5,
            "captain": 4.9, "jack_of_all_trades": 2., "adventurer": 6.3, "prop_plane": 2.5,
            "cartographer": 7., "compass": 2., "scientist": 5., "travel_log": 3., "native": 7., "transmitter": 5.}
    return base.get(name, .3) / (1 + .6*collection.count(name))


def _features(o, a):
    f = [0.] * 24
    k, p = a["type"], o["players"][o["player"]]
    occupied = {x for person in o["players"] for x in person["pawns"] if x is not None}
    hand = p.get("hand", [])
    # Collection derives from visible own cards and public acquisition history.
    collection = ["explorer"]*3 + ["traveler"]*4 + ["sailor"]
    for event in o["history"]:
        if event.get("player") != o["player"]:
            continue
        if event["event"] == "acquire":
            collection.append(card_type(event["card"]))
        elif event["event"] == "trash" or event["event"] == "play" and event.get("removed"):
            name = card_type(event["card"])
            if name in collection:
                collection.remove(name)
    def gain(src, dst):
        return DISTANCE[src] - DISTANCE[dst] + (5 if BOARD[dst]["finish"] else 0)
    def block_between(src, dst):
        b = SEAMS.get((src, dst))
        return b if b is not None and o["blockades"][b] is not None else None
    def card_move_gain(pos, color, power):
        # Same-color continuation only; zero-cost start hexes cannot recurse forever.
        seen, queue, best = {pos: power}, [(pos, power)], -1.
        while queue:
            src, left = queue.pop()
            for dst in NEIGHBORS[src]:
                if dst in occupied or BOARD[dst]["color"] != color:
                    continue
                b = block_between(src, dst)
                cost = BOARD[dst]["cost"]
                if b is not None:
                    bb = o["blockades"][b]
                    if bb["color"] != color:
                        continue
                    cost += bb["cost"]
                if cost > left:
                    continue
                best = max(best, gain(pos, dst))
                remaining = left-cost
                if not BOARD[dst]["finish"] and remaining > seen.get(dst, -1):
                    seen[dst] = remaining
                    queue.append((dst, remaining))
        # A blocked seam can be removed without entering the next space.
        for dst in NEIGHBORS[pos]:
            b = block_between(pos, dst)
            if b is not None and o["blockades"][b]["color"] == color and o["blockades"][b]["cost"] <= power:
                best = max(best, .7)
        return best
    c = a.get("card")
    d = card_data(c) if c and c not in CARDS else CARDS.get(c, {})
    f[3] = float(d.get("power", 0))
    f[4] = float(d.get("price", 0))
    f[5] = float(d.get("item", False))
    f[6] = float(d.get("draw", 0))
    f[7] = float(d.get("trash", 0))
    f[8] = len(hand) / 4
    f[9] = len(collection) / 10
    f[10] = o["turns"][o["player"]] / 20
    f[11] = o["money2"] / 2
    f[12] = sum(x is None for x in p["pawns"])
    f[13] = sum(x is None for x in o["players"][1-o["player"]]["pawns"])
    if k == "play_move":
        pos = p["pawns"][a["pawn"]]
        f[1] = card_move_gain(pos, a["color"], d["power"])
        f[0] = 3*f[1] - .25
    elif k in ("move", "pay_move", "native_move"):
        pawn = o["active"]["pawn"] if k == "move" else a["pawn"]
        pos = p["pawns"][pawn]
        f[1] = gain(pos, a["to"])
        f[14] = BOARD[a["to"]]["cost"]
        f[15] = float(BOARD[a["to"]]["finish"])
        f[16] = float(BOARD[a["to"]]["color"] == "c")
        f[17] = len(a.get("cards", []))
        f[0] = 3*f[1] - .3*f[17]
        if f[16]:
            f[0] += sum(1 if card_type(c) in STARTER else -3 for c in a.get("cards", []))
        if k == "move":
            # Look beyond this cell so a temporary detour can unlock useful reach.
            left = o["active"]["power"] - BOARD[a["to"]]["cost"]
            if not f[15]:
                f[0] += max(0., card_move_gain(a["to"], o["active"]["color"], left))
    elif k in ("remove_blockade", "pay_blockade", "native_blockade"):
        f[2] = 1.
        f[18] = o["blockades"][a["blockade"]]["rank"] / 6
        f[0] = 2.5 + f[18]
    elif k == "play_action":
        f[0] = 3 + 2*d["draw"] + d["trash"]
        if d["effect"] == "transmitter":
            f[0] = 9.
    elif k in ("buy", "gain"):
        f[19] = _card_value(c, collection)
        f[0] = f[19] - .05*d["price"]
    elif k == "start_buy":
        money = sum(2*card_data(c)["power"] if card_data(c)["color"] in ("w", "y") else 1
                    for c in hand if c not in p["fresh"])
        options = [name for name, count in o["market"].items() if count and
                   (name in o["market_slots"] or None in o["market_slots"]) and CARDS[name]["price"]*2 <= money]
        f[19] = max((_card_value(name, collection) for name in options), default=0.)
        f[0] = f[19] * (1 if max(o["turns"]) < 10 else .45) - 1.
    elif k == "spend":
        extra = 2*d["power"] if a["mode"] == "coin" else 1
        f[20] = extra / 2
        potential = o["money2"] + sum(
            2*card_data(x)["power"] if card_data(x)["color"] in ("y", "w") else 1
            for x in hand if x not in p["fresh"])
        options = [name for name, count in o["market"].items() if count and
                   (name in o["market_slots"] or None in o["market_slots"]) and
                   CARDS[name]["price"]*2 <= potential]
        target = max(options, key=lambda n: _card_value(n, collection)-.05*CARDS[n]["price"], default=None)
        f[0] = (_card_value(target, collection)+.1+.01*extra
                if target and o["money2"] < 2*CARDS[target]["price"] else -2.)
    elif k == "trash":
        f[21] = float(card_type(c) in STARTER)
        f[0] = (2. if f[21] and len(collection) > 7 else -5.)
    elif k == "discard":
        f[21] = float(card_type(c) in STARTER)
        # Discard unused starters to cycle; keep valuable cards, but never lock 4.
        f[0] = 1. if f[21] or len(hand) >= 4 else -.3
    elif k in ("end_turn", "refill", "stop_move", "finish_effect"):
        f[0] = 0.
    return f


GAME_CLASS = ElDorado
