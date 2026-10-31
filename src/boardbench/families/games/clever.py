"""That's Pretty Clever: original sheet, official six-round two-player game."""
from copy import deepcopy
import json
from pathlib import Path

from boardbench.families.base import DuoGame

DATA = json.loads((Path(__file__).parents[1] / "data/clever_sheet.json").read_text())
COLORS = DATA["dice_colors"]
YELLOW = [v for row in DATA["yellow"]["grid"] for v in row]
BLUE = DATA["blue"]["grid"]


def empty_sheet():
    return dict(yellow=[i for i, v in enumerate(YELLOW) if v is None], blue=[],
                green=0, orange=[], purple=[], foxes=0, rerolls=0, extras=0, claimed=[])


def color_scores(sheet):
    yellow = sum(score for col, score in enumerate(DATA["yellow"]["column_scores"])
                 if all(row*4+col in sheet["yellow"] for row in range(4)))
    return [yellow, DATA["blue"]["score_by_cross_count"][len(sheet["blue"])],
            DATA["green"]["score_by_cross_count"][sheet["green"]],
            sum(sheet["orange"]), sum(sheet["purple"])]


def breakdown(sheet):
    scores = color_scores(sheet)
    result = dict(zip(COLORS[:5], scores))
    result.update(foxes=sheet["foxes"], fox_points=min(scores)*sheet["foxes"],
                  total=sum(scores)+min(scores)*sheet["foxes"])
    return result


def mark_options(sheet, color, value=None, free=False):
    if color == "yellow":
        targets = [i for i, v in enumerate(YELLOW)
                   if i not in sheet["yellow"] and (free or v == value)]
    elif color == "blue":
        targets = [v for v in range(2, 13) if v not in sheet["blue"] and (free or v == value)]
    elif color == "green":
        n = sheet["green"]
        targets = [n] if n < 11 and (free or value >= DATA["green"]["minimum_values"][n]) else []
    elif color == "orange":
        targets = [len(sheet[color])] if len(sheet[color]) < 11 else []
    else:
        track = sheet["purple"]
        targets = [len(track)] if len(track) < 11 and (not track or track[-1] == 6 or value > track[-1]) else []
    return [dict(color=color, target=t, value=value if value is not None else 0) for t in targets]


def apply_mark(sheet, mark):
    """Apply a legal mark and return every newly earned printed reward."""
    color, target, value = mark["color"], mark["target"], mark["value"]
    if color == "yellow":
        sheet[color].append(target)
        sheet[color].sort()
    elif color == "blue":
        sheet[color].append(target)
        sheet[color].sort()
    elif color == "green":
        sheet[color] += 1
    elif color == "orange":
        sheet[color].append(value * DATA["orange"]["multipliers"][target])
    else:
        sheet[color].append(value)
    rewards = []

    def earn(key, condition, reward):
        if condition and key not in sheet["claimed"]:
            sheet["claimed"].append(key)
            rewards.append(reward)

    if color == "yellow":
        for r in range(4):
            earn(f"yellow_row_{r}", all(r*4+c in sheet[color] for c in range(4)),
                 DATA[color]["row_bonuses"][r])
        earn("yellow_diagonal", all(i*5 in sheet[color] for i in range(4)), "extra_die")
    elif color == "blue":
        for r, row in enumerate(BLUE):
            earn(f"blue_row_{r}", all(v in sheet[color] for v in row if v is not None),
                 DATA[color]["row_bonuses"][r])
        for c in range(4):
            earn(f"blue_col_{c}", all(row[c] in sheet[color] for row in BLUE if row[c] is not None),
                 DATA[color]["column_bonuses"][c])
    else:
        count = sheet[color] if color == "green" else len(sheet[color])
        reward = DATA[color]["bonuses"].get(str(count))
        if reward:
            earn(f"{color}_{count}", True, reward)
    return rewards


def bonus_options(sheet, reward):
    if reward == "round_four":
        return [a for c in COLORS[:5] for a in mark_options(sheet, c,
                6 if c in ("orange", "purple") else None, c in ("yellow", "blue", "green"))]
    if isinstance(reward, dict):
        key, value = next(iter(reward.items()))
        return mark_options(sheet, key.split("_")[0], value)
    return mark_options(sheet, reward.split("_")[0], free=True)


class Clever(DuoGame):
    game_id = "clever"
    config_id = "original_sheet_duo_six_rounds"

    def _reset(self, first_player):
        self.state.update(sheets=[empty_sheet(), empty_sheet()], round=1,
                          active_player=first_player, turn_in_round=0, phase="active",
                          dice={c: 1 for c in COLORS}, available=[], platter=[], kept=[],
                          selections=0, extra_used=[], bonuses=[], round_bonus_seat=None)
        self._start_round()

    def _start_round(self):
        s = self.state
        r = s["round"]
        for sheet in s["sheets"]:
            if r in (1, 3):
                sheet["rerolls"] += 1
            elif r == 2:
                sheet["extras"] += 1
        s["active_player"] = s["first_player"]
        s["current_player"] = s["first_player"]
        s["turn_in_round"] = 0
        if r == 4:
            s["round_bonus_seat"] = s["first_player"]
            s["phase"] = "round_bonus"
            s["bonuses"] = ["round_four"]
            self._settle()
        else:
            self._start_active()

    def _start_active(self):
        s = self.state
        s.update(current_player=s["active_player"], phase="active", available=list(COLORS),
                 platter=[], kept=[], selections=0, extra_used=[], bonuses=[])
        self._roll_available()

    def _roll_available(self):
        for c in self.state["available"]:
            self.state["dice"][c] = self.rng.randint(1, 6)

    def _observe(self, player):
        return {k: self.state[k] for k in ("sheets", "round", "active_player", "turn_in_round",
                "phase", "dice", "available", "platter", "kept", "selections", "extra_used", "bonuses")}

    def _dice_options(self, dice, action_type):
        s = self.state
        sheet = s["sheets"][s["current_player"]]
        actions = []
        for die in dice:
            for color in COLORS[:5] if die == "white" else [die]:
                value = s["dice"]["blue"] + s["dice"]["white"] if color == "blue" else s["dice"][die]
                actions.extend(dict(type=action_type, die=die, **m) for m in mark_options(sheet, color, value))
        return actions

    def _legal_actions(self):
        s = self.state
        sheet = s["sheets"][s["current_player"]]
        if s["bonuses"]:
            return [dict(type="bonus", index=i, **m) for i, bonus in enumerate(s["bonuses"])
                    for m in bonus_options(sheet, bonus)]
        phase = s["phase"]
        if phase == "active":
            actions = self._dice_options(s["available"], "select")
            if not actions:
                actions = [{"type": "unusable_roll"}]
            if sheet["rerolls"]:
                actions.append({"type": "reroll"})
            return actions
        if phase == "passive":
            actions = self._dice_options(s["platter"], "select")
            if not actions:
                actions = self._dice_options(s["kept"], "select")
            return actions or [{"type": "no_passive_mark"}]
        if phase in ("active_extra", "passive_extra"):
            actions = [{"type": "end_window"}]
            if sheet["extras"]:
                actions += self._dice_options([c for c in COLORS if c not in s["extra_used"]], "extra")
            return actions
        raise ValueError(f"unsettled phase {phase}")

    def _add_rewards(self, rewards):
        s = self.state
        sheet = s["sheets"][s["current_player"]]
        for reward in rewards:
            if reward == "reroll":
                sheet["rerolls"] += 1
            elif reward == "extra_die":
                sheet["extras"] += 1
            elif reward == "fox":
                sheet["foxes"] += 1
            else:
                s["bonuses"].append(reward)

    def _settle(self):
        """Finish forced transitions, never discard a legal optional choice."""
        s = self.state
        while not s["terminated"]:
            sheet = s["sheets"][s["current_player"]]
            # A full target area loses an unusable immediate reward; never bank it.
            s["bonuses"] = [b for b in s["bonuses"] if bonus_options(sheet, b)]
            if s["bonuses"]:
                return
            if s["phase"] == "after_active":
                if s["selections"] < 3 and s["available"]:
                    self._roll_available()
                    s["phase"] = "active"
                else:
                    s["platter"].extend(s["available"])
                    s["available"] = []
                    s["phase"] = "active_extra"
                    s["extra_used"] = []
                return
            if s["phase"] == "round_bonus":
                if s["round_bonus_seat"] == s["first_player"]:
                    s["round_bonus_seat"] = 1-s["first_player"]
                    s["current_player"] = s["round_bonus_seat"]
                    s["bonuses"] = ["round_four"]
                    continue
                s["round_bonus_seat"] = None
                self._start_active()
            return

    def _step(self, action):
        s = self.state
        t = action["type"]
        sheet = s["sheets"][s["current_player"]]
        if t == "bonus":
            s["bonuses"].pop(action["index"])
            self._add_rewards(apply_mark(sheet, action))
        elif t == "reroll":
            sheet["rerolls"] -= 1
            self._roll_available()
        elif t in ("select", "extra"):
            self._add_rewards(apply_mark(sheet, action))
            if t == "extra":
                sheet["extras"] -= 1
                s["extra_used"].append(action["die"])
            elif s["phase"] == "active":
                die = action["die"]
                s["available"].remove(die)
                s["kept"].append(die)
                lower = [c for c in s["available"] if s["dice"][c] < s["dice"][die]]
                s["platter"].extend(lower)
                s["available"] = [c for c in s["available"] if c not in lower]
                s["selections"] += 1
                s["phase"] = "after_active"
            else:
                s["phase"] = "passive_extra"
                s["extra_used"] = []
        elif t == "unusable_roll":
            s["selections"] += 1
            s["phase"] = "after_active"
        elif t == "no_passive_mark":
            s["phase"] = "passive_extra"
            s["extra_used"] = []
        elif t == "end_window":
            if s["phase"] == "active_extra":
                s["current_player"] = 1-s["active_player"]
                s["phase"] = "passive"
                s["extra_used"] = []
            elif s["turn_in_round"] == 0:
                s["turn_in_round"] = 1
                s["active_player"] = 1-s["active_player"]
                self._start_active()
            elif s["round"] == 6:
                totals = self.scores()
                if totals[0] != totals[1]:
                    self.finish(reason="six_rounds")
                else:
                    best = [max(color_scores(p)) for p in s["sheets"]]
                    self.finish([p for p in (0, 1) if best[p] == max(best)], "six_rounds")
            else:
                s["round"] += 1
                self._start_round()
        self._settle()

    def scores(self):
        return [breakdown(p)["total"] for p in self.state["sheets"]]

    def score_breakdown(self):
        return [breakdown(p) for p in self.state["sheets"]]

    def validate(self):
        super().validate()
        s = self.state
        assert 1 <= s["round"] <= 6 and s["active_player"] in (0, 1)
        assert s["phase"] in ("active", "after_active", "active_extra", "passive", "passive_extra", "round_bonus")
        assert all(type(v) is int and 1 <= v <= 6 for v in s["dice"].values())
        assert set(s["dice"]) == set(COLORS)
        parts = s["available"] + s["platter"] + s["kept"]
        assert sorted(parts) == sorted(COLORS)
        assert len(s["extra_used"]) == len(set(s["extra_used"]))
        assert set(s["extra_used"]) <= set(COLORS)
        for p in s["sheets"]:
            assert set(i for i, v in enumerate(YELLOW) if v is None) <= set(p["yellow"])
            assert len(p["yellow"]) == len(set(p["yellow"])) and all(0 <= v < 16 for v in p["yellow"])
            assert len(p["blue"]) == len(set(p["blue"])) and all(2 <= v <= 12 for v in p["blue"])
            assert 0 <= p["green"] <= 11
            assert len(p["orange"]) <= 11 and len(p["purple"]) <= 11
            assert all(type(n) is int and 1 <= n <= 6 for n in p["purple"])
            assert all(a == 6 or a < b for a, b in zip(p["purple"], p["purple"][1:]))
            assert all(1 <= n <= 6*DATA["orange"]["multipliers"][i] and n % DATA["orange"]["multipliers"][i] == 0
                       for i, n in enumerate(p["orange"]))
            assert all(type(p[k]) is int and p[k] >= 0 for k in ("rerolls", "extras", "foxes"))
            assert len(p["claimed"]) == len(set(p["claimed"]))

    @staticmethod
    def features(obs, action):
        sheet = obs["sheets"][obs["player"]]
        f = [0.] * 24
        kind = action["type"]
        if "color" in action:
            after = deepcopy(sheet)
            rewards = apply_mark(after, action)
            old = breakdown(sheet)
            new = breakdown(after)
            f[1] = (new["total"]-old["total"])/10
            f[2] = len(rewards)
            f[3] = action["value"]/12
            color = action["color"]
            for i, c in enumerate(COLORS[:5]):
                f[4+i] = float(color == c)
            # Partial lines have genuine future value even before a score is paid.
            if color == "yellow":
                r, c = divmod(action["target"], 4)
                f[9] = sum(r*4+x in after["yellow"] for x in range(4))/4
                f[10] = sum(x*4+c in after["yellow"] for x in range(4))/4
            f[11] = float(kind == "select" and obs["phase"] == "active") * sum(
                obs["dice"][d] >= obs["dice"][action["die"]] for d in obs["available"] if d != action["die"])/5 if kind != "bonus" else 0
            # Green/blue progressive scores, yellow future lines, balanced fox income.
            minimum = min(color_scores(sheet))
            f[12] = float(old[color] == minimum)
            f[0] = 2 + f[1]*3 + f[2]*4 + f[9]+f[10] + f[11]*2 + f[12]*.8
            if color == "purple":
                f[0] += action["value"]*.25
            if color == "yellow":
                f[0] += 1.5
        elif kind == "reroll":
            # Spend only when current available faces look poor; actions are finite.
            f[0] = 2.3 if max(obs["dice"][c] for c in obs["available"]) <= 3 else -1
        elif kind == "end_window":
            f[0] = .5 if obs["round"] < 6 else -1
        f[13] = obs["round"]/6
        f[14] = sheet["extras"]/10
        f[15] = sheet["rerolls"]/10
        f[16] = sheet["foxes"]/6
        return f


GAME_CLASS = Clever
