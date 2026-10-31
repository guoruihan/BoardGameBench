"""Classic 2014 English Machi Koro, complete fixed two-player base market."""
from copy import deepcopy
import json
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / "data/machi_koro_cards.json").read_text())
CARDS = {c["card_id"]: c for c in DATA["establishments"]}
LANDMARKS = {c["id"]: c for c in DATA["landmarks"]}


def income(player, card):
    """Per-copy bank/restaurant amount, before poverty limits."""
    effect = CARDS[card]["effect"]
    amount = effect["amount"] or 0
    if effect["type"] == "bank_per_icon":
        amount *= sum(n for c, n in player["cards"].items()
                      if CARDS[c]["icon"] == effect["count_icon"])
    if player["landmarks"]["shopping_mall"] and CARDS[card]["icon"] in ("shop", "cup"):
        amount += 1
    return amount


def roll_value(players, seat, dice):
    """Public expected relative coin gain, used only by the baseline."""
    active, other = players[seat], players[1-seat]
    value = 0.
    distribution = {v: 1/6 for v in range(1, 7)} if dice == 1 else {
        v: (6-abs(7-v))/36 for v in range(2, 13)}
    for total, prob in distribution.items():
        gain = 0.
        for c, n in active["cards"].items():
            if total in CARDS[c]["activation_totals"] and CARDS[c]["color"] in ("blue", "green"):
                gain += n * income(active, c)
            if total == 6 and c in ("stadium", "tv_station"):
                gain += n * (2 if c == "stadium" else 5) * .8
        for c, n in other["cards"].items():
            if total in CARDS[c]["activation_totals"]:
                if CARDS[c]["color"] == "red":
                    gain -= min(active["coins"], n * income(other, c)) * 1.4
                elif CARDS[c]["color"] == "blue":
                    gain -= n * income(other, c) * .5
        value += prob * gain
    return value


class MachiKoro(DuoGame):
    game_id = "machi_koro"
    config_id = "classic_2014_base_duo"

    def _reset(self, first_player):
        self.state.update(players=[dict(coins=3,
            cards={c: int(c in ("wheat_field", "bakery")) for c in CARDS},
            landmarks={c: False for c in LANDMARKS}) for _ in range(2)],
            market={c: d["supply_copies"] for c, d in CARDS.items()},
            phase="roll", dice=[], pending_purple=[], extra_turn=False, turn=1,
            last_roll=None)

    def _observe(self, player):
        return {k: self.state[k] for k in ("players", "market", "phase", "dice",
                "pending_purple", "extra_turn", "turn", "last_roll")}

    def _legal_actions(self):
        s = self.state
        p, q = s["players"][s["current_player"]], s["players"][1-s["current_player"]]
        phase = s["phase"]
        if phase == "roll":
            return [{"type": "roll", "dice": n} for n in (1, 2)
                    if n == 1 or p["landmarks"]["train_station"]]
        if phase == "reroll":
            return [{"type": "keep"}, {"type": "reroll"}]
        if phase == "purple":
            actions = []
            for c in s["pending_purple"]:
                if c == "business_center":
                    actions.append({"type": "business", "give": None, "take": None})
                    actions.extend({"type": "business", "give": a, "take": b}
                        for a in CARDS for b in CARDS
                        if p["cards"][a] and q["cards"][b]
                        and CARDS[a]["color"] != "purple" and CARDS[b]["color"] != "purple")
                else:
                    actions.append({"type": "purple", "card": c, "target": 1-s["current_player"]})
            return actions
        if phase == "construction":
            actions = [{"type": "pass"}]
            actions += [{"type": "buy", "card": c} for c, d in CARDS.items()
                        if s["market"][c] and p["coins"] >= d["cost"]
                        and (d["max_owned"] is None or p["cards"][c] < d["max_owned"])]
            actions += [{"type": "landmark", "landmark": c} for c, d in LANDMARKS.items()
                        if not p["landmarks"][c] and p["coins"] >= d["cost"]]
            return actions
        raise ValueError("invalid phase")

    def _transfer(self, payer, receiver, amount):
        players = self.state["players"]
        amount = min(amount, players[payer]["coins"])
        players[payer]["coins"] -= amount
        players[receiver]["coins"] += amount

    def _resolve_income(self):
        s = self.state
        a, b = s["current_player"], 1-s["current_player"]
        total = sum(s["dice"])
        p, q = s["players"][a], s["players"][b]
        # Record the trigger at the roll event: a later purchase cannot rewrite it.
        s["extra_turn"] = bool(p["landmarks"]["amusement_park"] and
            len(s["dice"]) == 2 and s["dice"][0] == s["dice"][1])
        for c, n in q["cards"].items():
            if CARDS[c]["color"] == "red" and total in CARDS[c]["activation_totals"]:
                self._transfer(a, b, n * income(q, c))
        for seat, player in enumerate(s["players"]):
            for c, n in player["cards"].items():
                d = CARDS[c]
                if total in d["activation_totals"] and (d["color"] == "blue"
                        or d["color"] == "green" and seat == a):
                    player["coins"] += n * income(player, c)
        s["pending_purple"] = [c for c, n in p["cards"].items()
            if n and CARDS[c]["color"] == "purple" and total in CARDS[c]["activation_totals"]]
        s["last_roll"] = dict(player=a, dice=list(s["dice"]), total=total)
        s["phase"] = "purple" if s["pending_purple"] else "construction"

    def _step(self, action):
        s = self.state
        a, b = s["current_player"], 1-s["current_player"]
        p, q = s["players"][a], s["players"][b]
        t = action["type"]
        if t in ("roll", "reroll"):
            count = action["dice"] if t == "roll" else len(s["dice"])
            s["dice"] = [self.rng.randint(1, 6) for _ in range(count)]
            if t == "roll" and p["landmarks"]["radio_tower"]:
                s["phase"] = "reroll"
            else:
                self._resolve_income()
        elif t == "keep":
            self._resolve_income()
        elif t in ("purple", "business"):
            if t == "purple":
                self._transfer(b, a, 2 if action["card"] == "stadium" else 5)
                s["pending_purple"].remove(action["card"])
            else:
                x, y = action["give"], action["take"]
                if x is not None:
                    p["cards"][x] -= 1
                    q["cards"][x] += 1
                    q["cards"][y] -= 1
                    p["cards"][y] += 1
                s["pending_purple"].remove("business_center")
            if not s["pending_purple"]:
                s["phase"] = "construction"
        else:
            if t == "buy":
                c = action["card"]
                p["coins"] -= CARDS[c]["cost"]
                s["market"][c] -= 1
                p["cards"][c] += 1
            elif t == "landmark":
                c = action["landmark"]
                p["coins"] -= LANDMARKS[c]["cost"]
                p["landmarks"][c] = True
                if all(p["landmarks"].values()):
                    self.finish([a], "four_landmarks")
                    return
            s["current_player"] = a if s["extra_turn"] else b
            s.update(phase="roll", dice=[], pending_purple=[], extra_turn=False, turn=s["turn"]+1)

    def scores(self):
        return [None, None]  # Race victory; landmark count is not a victory-point score.

    def score_breakdown(self):
        return [dict(total=None, landmarks=sum(p["landmarks"].values()), coins=p["coins"])
                for p in self.state["players"]]

    def validate(self):
        super().validate()
        s = self.state
        assert s["phase"] in ("roll", "reroll", "purple", "construction")
        for p in s["players"]:
            assert type(p["coins"]) is int and p["coins"] >= 0
            assert set(p["cards"]) == set(CARDS) and set(p["landmarks"]) == set(LANDMARKS)
            assert all(type(v) is bool for v in p["landmarks"].values())
            for c, n in p["cards"].items():
                assert type(n) is int and n >= 0
                assert CARDS[c]["max_owned"] is None or n <= CARDS[c]["max_owned"]
        for c, d in CARDS.items():
            assert type(s["market"][c]) is int and s["market"][c] >= 0
            assert s["market"][c] + sum(p["cards"][c] for p in s["players"]) == d["supply_copies"] + 2*int(c in ("wheat_field", "bakery"))
        assert all(type(v) is int and 1 <= v <= 6 for v in s["dice"])

    @staticmethod
    def features(obs, action):
        players = obs["players"]
        seat = obs["player"]
        p = players[seat]
        t = action["type"]
        f = [0.] * 24
        if t == "roll":
            f[1] = roll_value(players, seat, action["dice"])
            f[0] = f[1] + .6 * (action["dice"] == 2 and p["landmarks"]["amusement_park"])
        elif t in ("keep", "reroll"):
            # Evaluate final observed result using exact immediate public income.
            count = len(obs["dice"])
            total = sum(obs["dice"])
            value = sum(n * income(p, c) for c, n in p["cards"].items()
                if total in CARDS[c]["activation_totals"] and CARDS[c]["color"] in ("green", "blue", "purple"))
            value -= sum(min(p["coins"], n*income(players[1-seat], c)) for c, n in players[1-seat]["cards"].items()
                if total in CARDS[c]["activation_totals"] and CARDS[c]["color"] == "red")
            f[0] = value if t == "keep" else roll_value(players, seat, count)
        elif t == "landmark":
            c = action["landmark"]
            f[2] = LANDMARKS[c]["cost"] / 22
            f[3] = sum(p["landmarks"].values()) / 4
            f[0] = 6 + f[3]*5 + f[2]
            if all(v for k, v in p["landmarks"].items() if k != c):
                f[0] = 1000
            if c == "shopping_mall":
                f[0] += sum(n for k, n in p["cards"].items() if CARDS[k]["icon"] in ("shop", "cup"))
        elif t == "buy":
            c = action["card"]
            new = deepcopy(players)
            new[seat]["cards"][c] += 1
            n = 2 if p["landmarks"]["train_station"] and roll_value(players, seat, 2) > roll_value(players, seat, 1) else 1
            marginal = roll_value(new, seat, n) - roll_value(players, seat, n)
            if CARDS[c]["color"] in ("blue", "red"):
                prob = sum(1/6 if v <= 6 else 0 for v in CARDS[c]["activation_totals"])
                marginal += income(p, c)*prob*.8
            f[4] = marginal
            f[5] = CARDS[c]["cost"] / 8
            f[6] = sum(p["cards"].values()) / 20
            f[0] = 18*marginal / max(1, CARDS[c]["cost"]**.5) - .65*sum(p["cards"].values())
        elif t == "business":
            if action["give"] is not None:
                f[0] = CARDS[action["take"]]["cost"] - CARDS[action["give"]]["cost"]
        elif t == "purple":
            f[0] = 1
        f[7] = p["coins"] / 30
        f[8] = obs["turn"] / 100
        return f


GAME_CLASS = MachiKoro
