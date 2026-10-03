#!/usr/bin/env python3
"""Validate handoff data with the standard library; this does not test an engine."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distance(a, b):
    q, r = a[0] - b[0], a[1] - b[1]
    return max(abs(q), abs(r), abs(q + r))


def check_cards():
    cards = read("data/harmonies/animal_cards.base32.v1.json")["cards"]
    other = {row["source_id"]: row for row in read("reference/score_crosscheck.json")["cards"]}
    require(len(cards) == 32, "Expected 32 base animal cards")
    require({c["card_id"] for c in cards} == set(range(1, 33)), "Card IDs must be 1..32")
    for card in cards:
        tag = f"card {card['card_id']}"
        cells = [tuple(c["cell"]) for c in card["pattern_cells"]]
        require(len(cells) == len(set(cells)), tag + ": duplicate cells")
        require(card["anchor_cell"] == [0, 0] and cells.count((0, 0)) == 1, tag + ": anchor")
        seen = {cells[0]}
        while True:
            nxt = seen | {b for b in cells if any(distance(a, b) == 1 for a in seen)}
            if nxt == seen:
                break
            seen = nxt
        require(seen == set(cells), tag + ": disconnected pattern")
        scores = card["score_by_placed_count"]
        require(len(scores) == card["cube_count"] + 1 and scores[0] == 0, tag + ": score length")
        require(all(a < b for a, b in zip(scores, scores[1:])), tag + ": score order")
        ref = other[card["card_id"]]
        require(scores == [0] + ref["points"] == [0] + ref["second_source_points"], tag + ": scores")
        require(card["name_en"] == ref["name"], tag + ": name")
        for cell in card["pattern_cells"]:
            if cell["terrain"] == "building":
                require(cell["height"] == 2, tag + ": building height")
                require(set(cell["allowed_base_colors"]) == {"red", "gray", "brown"}, tag + ": bases")
            else:
                require(len(cell["stack_bottom_to_top"]) == cell["height"], tag + ": stack height")
    require(sum(c["cube_count"] for c in cards) == 96, "Expected 96 slots across the deck")


def check_witness():
    w = read("reference/bag_exhaustion_witness.json")
    bag = Counter(w["initial_bag"])
    require(sum(bag.values()) == 120, "Witness bag size")
    require(Counter(w["draw_order"]) == bag, "Witness draw order inventory")
    draw_pos, discarded, board = 0, 0, {}
    for turn in w["turns"]:
        bundles = turn["supply_bundles"]
        flat = [color for bundle in bundles for color in bundle]
        require(len(bundles) == 3 and all(len(b) == 3 for b in bundles), "Witness bundles")
        require(flat == w["draw_order"][draw_pos:draw_pos + 9], "Witness order mismatch")
        draw_pos += 9
        bag.subtract(flat)
        require(all(v >= 0 for v in bag.values()), "Negative inventory")
        chosen = bundles[turn["chosen_bundle"]]
        require(chosen in [["gray"] * 3, ["brown", "brown", "green"]], "Invalid witness stack")
        require(chosen == turn["stack_bottom_to_top"], "Witness selected stack")
        cell = turn["place_in_new_cell"]
        require(0 <= cell < 23 and cell not in board, "Witness occupied target")
        board[cell] = chosen
        discarded += 6
        require(23 - len(board) > 2, "Witness prematurely meets board termination")
        require(sum(bag.values()) == turn["tokens_in_bag_after_round_supply"], "Witness bag count")
    require(len(board) == 13 and discarded == 78 and sum(bag.values()) == 3, "Witness final counts")


def check_scoring_examples():
    x = read("data/scoring_examples.json")
    for ex in x["micro_tiles"]:
        b = ex["board"]
        require(len(b) == 9 and set(b) <= {0, 1, 2}, "Micro example colors")
        edges = [(i, i + 1) for i in range(9) if i % 3 < 2]
        edges += [(i, i + 3) for i in range(6)]
        lines = [b[3 * r:3 * r + 3] for r in range(3)]
        lines += [[b[3 * r + c] for r in range(3)] for c in range(3)]
        adj = sum(b[i] == b[j] for i, j in edges)
        line_score = 3 * sum(len(set(line)) == 1 for line in lines)
        require({"adjacency": adj, "lines": line_score, "total": adj + line_score} == ex["expected"], "Micro score")
    for ex in x["take_it_easy"]:
        board = ex["board"]
        cells = [(q, r) for q in range(-2, 3) for r in range(-2, 3)
                 if max(abs(q), abs(r), abs(q + r)) <= 2]
        require([(t["q"], t["r"]) for t in board] == cells, "TIE topology/order")
        require(len({t["tile_id"] for t in board}) == 19, "TIE repeated tiles")
        for i, t in enumerate(board):
            tid = t["tile_id"]
            require(t["cell_id"] == i and 0 <= tid < 27, "TIE IDs")
            require((t["vertical"], t["rising"], t["falling"]) ==
                    ([1, 5, 9][tid // 9], [2, 6, 7][tid % 9 // 3], [3, 4, 8][tid % 3]), "TIE tile mapping")
        totals = {}
        for direction, key in [("vertical", lambda t: t["q"]),
                               ("rising", lambda t: t["q"] + t["r"]),
                               ("falling", lambda t: t["r"])]:
            groups = defaultdict(list)
            for t in board:
                groups[key(t)].append(t[direction])
            scores = {str(k): len(v) * v[0] if len(set(v)) == 1 else 0 for k, v in groups.items()}
            require(scores == ex["expected_lines"][direction], "TIE line scores")
            totals[direction] = sum(scores.values())
        totals["total"] = sum(totals.values())
        require(totals == ex["expected"], "TIE total")


def check_manifest():
    manifest = read("PACKAGE_MANIFEST.json")
    for item in manifest["files"]:
        p = ROOT / item["path"]
        require(p.is_file(), "Missing " + item["path"])
        content = p.read_bytes()
        require(len(content) == item["bytes"], "Size changed: " + item["path"])
        require(hashlib.sha256(content).hexdigest() == item["sha256"], "Hash changed: " + item["path"])


if __name__ == "__main__":
    check_manifest()
    check_cards()
    check_witness()
    check_scoring_examples()
    print("PASS: package files, 32 animal cards, independent score table, bag witness, and scoring examples.")
    print("This validates the handoff only; game-engine and training validation remain implementation work.")
