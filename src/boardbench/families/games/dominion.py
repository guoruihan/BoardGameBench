"""Dominion second edition, official First Game kingdom, two players.

All ten kingdom cards in this fixed official setup are implemented. This is not
an implementation claim for the other sixteen second-edition kingdom cards.
"""
from collections import Counter
import json
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / "data/dominion_first_game.json").read_text())
CARDS = DATA["cards"]
ZONES = ("deck", "hand", "discard", "in_play")


def total_cards(player):
    return Counter(c for zone in ZONES for c in player[zone])


class Dominion(DuoGame):
    game_id = "dominion"
    config_id = "second_edition_first_game_duo"

    def _reset(self, first_player):
        players = []
        for _ in range(2):
            deck = ['copper'] * 7 + ['estate'] * 3
            self.rng.shuffle(deck)
            players.append(dict(deck=deck, hand=[], discard=[], in_play=[], turns=0))
        self.state.update(players=players, supply={c: d["supply"] for c, d in CARDS.items()},
            trash=[], history=[], active_player=first_player, phase="action", actions=1,
            buys=1, coins=0, merchants=0, silver_played=False, cellar_count=0,
            gain_limit=0, gain_treasure_only=False, gain_destination="discard")
        for seat in (0, 1):
            self._draw(seat, 5)

    def _draw(self, seat, count):
        p = self.state["players"][seat]
        for _ in range(count):
            if not p["deck"]:
                if not p["discard"]:
                    break
                p["deck"] = p["discard"]
                p["discard"] = []
                self.rng.shuffle(p["deck"])
            p["hand"].append(p["deck"].pop())

    def _event(self, event, seat, card=None, **kwargs):
        record = dict(event=event, player=seat, **kwargs)
        if card is not None:
            record["card"] = card
        self.state["history"].append(record)

    def _observe(self, player):
        s = self.state
        players = []
        for seat, p in enumerate(s["players"]):
            view = dict(hand_count=len(p["hand"]), in_play=p["in_play"],
                        discard_top=p["discard"][-1] if p["discard"] else None,
                        turns=p["turns"])
            if seat == player:
                view.update(hand=p["hand"], draw_count=len(p["deck"]))
            if s["terminated"]:
                view["final_cards"] = dict(total_cards(p))
            players.append(view)
        result = {k: s[k] for k in ("supply", "trash", "history", "active_player", "phase", "actions",
            "buys", "coins", "merchants", "silver_played", "cellar_count", "gain_limit",
            "gain_treasure_only", "gain_destination")}
        result["players"] = players
        if s["terminated"]:
            result["final_scores"] = self.scores()
        return result

    def _legal_actions(self):
        s = self.state
        p = s["players"][s["current_player"]]
        hand = sorted(set(p["hand"]))
        phase = s["phase"]
        if phase == "action":
            return [{"type": "end_actions"}] + ([{"type": "play_action", "card": c} for c in hand
                if "action" in CARDS[c]["types"]] if s["actions"] > 0 else [])
        if phase == "treasure":
            return [{"type": "start_buy"}] + [{"type": "play_treasure", "card": c} for c in hand
                if "treasure" in CARDS[c]["types"]]
        if phase == "buy":
            # Cleanup ordering can reveal its chosen top card. Other order is
            # irrelevant in this kingdom because the next access shuffles it.
            tops = sorted(set(p["hand"] + p["in_play"]))
            actions = [{"type": "end_turn", "discard_top": c} for c in tops] or [
                {"type": "end_turn", "discard_top": None}]
            if s["buys"]:
                actions += [{"type": "buy", "card": c} for c, n in s["supply"].items()
                            if n and CARDS[c]["cost"] <= s["coins"]]
            return actions
        if phase in ("cellar", "militia_discard"):
            return ([{"type": "finish_cellar"}] if phase == "cellar" else []) + [
                {"type": "discard", "card": c} for c in hand]
        if phase == "reaction":
            return [{"type": "accept_attack"}] + ([{"type": "reveal_moat"}] if "moat" in hand else [])
        if phase in ("remodel", "mine"):
            actions = [{"type": "trash", "card": c} for c in hand
                       if phase == "remodel" or "treasure" in CARDS[c]["types"]]
            if phase == "mine":
                actions.append({"type": "decline_mine"})
            return actions
        if phase == "gain":
            return [{"type": "gain", "card": c} for c, n in s["supply"].items()
                    if n and CARDS[c]["cost"] <= s["gain_limit"]
                    and (not s["gain_treasure_only"] or "treasure" in CARDS[c]["types"])]
        raise ValueError(f"invalid phase {phase}")

    def _return_action(self):
        s = self.state
        s["current_player"] = s["active_player"]
        s["phase"] = "action" if s["actions"] > 0 else "treasure"

    def _gain_phase(self, limit, treasures=False, destination="discard"):
        self.state.update(phase="gain", gain_limit=limit,
                          gain_treasure_only=treasures, gain_destination=destination)
        if not self._legal_actions():
            self._return_action()

    def _gain(self, seat, card, destination="discard"):
        self.state["supply"][card] -= 1
        self.state["players"][seat][destination].append(card)
        self._event("gain", seat, card, destination=destination)

    def _action(self, card):
        s = self.state
        seat = s["active_player"]
        p = s["players"][seat]
        if card in ("market", "merchant", "village"):
            self._draw(seat, 1)
            s["actions"] += 2 if card == "village" else 1
            if card == "market":
                s["buys"] += 1
                s["coins"] += 1
            elif card == "merchant":
                s["merchants"] += 1
            self._return_action()
        elif card in ("smithy", "moat"):
            self._draw(seat, 3 if card == "smithy" else 2)
            self._return_action()
        elif card == "cellar":
            s["actions"] += 1
            s["cellar_count"] = 0
            s["phase"] = "cellar"
        elif card == "workshop":
            self._gain_phase(4)
        elif card == "remodel":
            s["phase"] = "remodel"
            if not p["hand"]:
                self._return_action()
        elif card == "mine":
            s["phase"] = "mine"
        elif card == "militia":
            s["coins"] += 2
            s["phase"] = "reaction"
            s["current_player"] = 1-seat
        else:
            raise ValueError("unsupported kingdom card")

    def _cleanup(self, top):
        s = self.state
        seat = s["active_player"]
        p = s["players"][seat]
        cards = p["in_play"] + p["hand"]
        if top is not None:
            cards.remove(top)
            cards.append(top)
        p["discard"].extend(cards)
        p["in_play"], p["hand"] = [], []
        if top is not None:
            self._event("cleanup_top", seat, top)
        self._draw(seat, 5)
        p["turns"] += 1
        if s["supply"]["province"] == 0 or sum(n == 0 for n in s["supply"].values()) >= 3:
            scores = self.scores()
            winners = [i for i in (0, 1) if scores[i] == max(scores)]
            least_turns = min(s["players"][i]["turns"] for i in winners)
            self.finish([i for i in winners if s["players"][i]["turns"] == least_turns], "supply_end")
            return
        s.update(active_player=1-seat, current_player=1-seat, phase="action",
                 actions=1, buys=1, coins=0, merchants=0, silver_played=False,
                 cellar_count=0, gain_limit=0, gain_treasure_only=False, gain_destination="discard")

    def _step(self, action):
        s = self.state
        seat = s["current_player"]
        p = s["players"][seat]
        t = action["type"]
        if t == "play_action":
            c = action["card"]
            p["hand"].remove(c)
            p["in_play"].append(c)
            s["actions"] -= 1
            self._event("play", seat, c)
            self._action(c)
        elif t == "end_actions":
            s["phase"] = "treasure"
        elif t == "play_treasure":
            c = action["card"]
            p["hand"].remove(c)
            p["in_play"].append(c)
            s["coins"] += CARDS[c]["coins"]
            if c == "silver" and not s["silver_played"]:
                s["coins"] += s["merchants"]
                s["silver_played"] = True
            self._event("play", seat, c)
        elif t == "start_buy":
            s["phase"] = "buy"
        elif t == "buy":
            c = action["card"]
            s["buys"] -= 1
            s["coins"] -= CARDS[c]["cost"]
            self._event("buy", seat, c)
            self._gain(seat, c)
        elif t == "end_turn":
            self._cleanup(action["discard_top"])
        elif t == "discard":
            c = action["card"]
            p["hand"].remove(c)
            p["discard"].append(c)
            self._event("discard", seat, c)
            if s["phase"] == "cellar":
                s["cellar_count"] += 1
            elif len(p["hand"]) <= 3:
                self._return_action()
        elif t == "finish_cellar":
            self._draw(seat, s["cellar_count"])
            self._return_action()
        elif t in ("accept_attack", "reveal_moat"):
            if t == "reveal_moat":
                self._event("reveal", seat, "moat")
            if t == "accept_attack" and len(p["hand"]) > 3:
                s["phase"] = "militia_discard"
            else:
                self._return_action()
        elif t == "trash":
            c = action["card"]
            p["hand"].remove(c)
            s["trash"].append(c)
            self._event("trash", seat, c)
            mine = s["phase"] == "mine"
            self._gain_phase(CARDS[c]["cost"] + (3 if mine else 2), mine, "hand" if mine else "discard")
        elif t == "decline_mine":
            self._return_action()
        elif t == "gain":
            self._gain(seat, action["card"], s["gain_destination"])
            self._return_action()

    def scores(self):
        return [sum(CARDS[c].get("points", 0)*n for c, n in total_cards(p).items()) for p in self.state["players"]]

    def score_breakdown(self):
        return [dict(total=score, **{c: total_cards(p)[c] for c in ("estate", "duchy", "province", "curse")})
                for score, p in zip(self.scores(), self.state["players"])]

    def validate(self):
        super().validate()
        s = self.state
        assert s["phase"] in ("action", "treasure", "buy", "cellar", "reaction", "militia_discard", "remodel", "mine", "gain")
        assert s["active_player"] in (0, 1)
        assert (s["current_player"] != s["active_player"]) == (s["phase"] in ("reaction", "militia_discard"))
        assert all(type(s[k]) is int and s[k] >= 0 for k in ("actions", "buys", "coins", "merchants"))
        all_cards = Counter(s["trash"])
        for p in s["players"]:
            all_cards.update(total_cards(p))
            assert p["turns"] >= 0
        assert set(s["supply"]) == set(CARDS)
        for c, d in CARDS.items():
            assert type(s["supply"][c]) is int and s["supply"][c] >= 0
            expected = d["supply"] + (14 if c == "copper" else 6 if c == "estate" else 0)
            assert all_cards[c] + s["supply"][c] == expected, c
        assert set(all_cards) <= set(CARDS)

    @staticmethod
    def features(obs, action):
        seat = obs["player"]
        p = obs["players"][seat]
        counts = Counter(copper=7, estate=3)
        # Ownership is reconstructed from public gain/trash history, never by
        # inspecting hidden deck zones or a privileged composition vector.
        for event in obs["history"]:
            if event["player"] == seat and event["event"] in ("gain", "trash"):
                counts[event["card"]] += 1 if event["event"] == "gain" else -1
        f = [0.] * 24
        t, c = action["type"], action.get("card")
        remaining = obs["supply"]["province"]
        values = dict(copper=-4, silver=5, gold=11, province=25,
                      duchy=13 if remaining <= 4 else -2,
                      estate=7 if remaining <= 2 else -5, curse=-20,
                      market=9, merchant=5, village=4, militia=7, smithy=7,
                      moat=3, mine=3, remodel=2, cellar=2, workshop=2)
        for card in ("militia", "smithy", "moat", "mine", "remodel", "cellar", "workshop"):
            values[card] -= counts[card]*5
        values["village"] += min(3, sum(counts[x] for x in ("smithy", "militia", "moat"))) - counts["village"]
        if c:
            f[1] = CARDS[c]["cost"]/8
            f[2] = counts[c]/10
            f[3] = float("treasure" in CARDS[c]["types"])
            f[4] = float("action" in CARDS[c]["types"])
            f[5] = float("victory" in CARDS[c]["types"])
            f[6] = CARDS[c].get("points", 0)/6
        if t in ("buy", "gain"):
            f[0] = values[c]
        elif t == "play_action":
            f[0] = {"village": 30, "market": 29, "merchant": 28, "cellar": 20,
                    "smithy": 12, "militia": 10, "mine": 8, "moat": 7, "remodel": 5, "workshop": 4}[c]
        elif t == "play_treasure":
            f[0] = 10 + CARDS[c]["coins"]
        elif t == "discard":
            if CARDS[c]["types"] in (["victory"], ["curse"]):
                f[0] = 15
            elif obs["phase"] == "cellar":
                f[0] = -3
            else:
                f[0] = 5 - CARDS[c].get("coins", 0)*2 - .1*CARDS[c]["cost"]
        elif t == "trash":
            f[0] = {"curse": 30, "estate": 15 if remaining > 2 else 0, "copper": 10,
                    "silver": 8, "gold": -5}.get(c, -CARDS[c]["cost"])
            if obs["phase"] == "mine":
                f[0] = {"copper": 5, "silver": 10, "gold": -5}[c]
        elif t == "reveal_moat":
            f[0] = 20
        f[7] = obs["coins"]/12
        f[8] = remaining/8
        f[9] = obs["actions"]/5
        f[10] = obs["buys"]/5
        f[11] = p["hand_count"]/15
        f[12] = p["turns"]/50
        f[13] = sum(counts.values())/50
        return f


GAME_CLASS = Dominion
