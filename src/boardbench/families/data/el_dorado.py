"""2017 El Dorado first-game numerical component transcription.

Art is deliberately not distributed. See docs/v160/games/el_dorado.md for the
publisher setup-sheet and original component-face provenance/checksums.
Rows run top to bottom, axial r=-3..3, q=max(-3,-3-r)..min(3,3-r).
"""

TILE_ROWS = {
    "B": ["g1 b1 c1 b1", "g1 g1 y1 xx g1", "g1 y1 g1 g1 g1 g1",
          "b1 g1 y1 g1 y1 g1 g1", "g1 g1 b1 g1 g1 g1", "g1 g1 g1 g1 g1", "s4 s3 s2 s1"],
    "C": ["g1 g1 r1 r1", "g1 y1 r1 b1 b1", "b1 b1 y1 y1 r1 b1",
          "b1 y1 r1 xx b1 r1 r1", "y1 r1 b1 b1 y1 y1", "y1 r1 g1 y1 b1", "g1 g1 b1 b1"],
    "N": ["g1 g1 g1 g1", "y1 g1 g2 g1 b1", "y1 y2 g1 b1 b1 b1",
          "b1 b1 y4 y4 y2 y1 y1", "b1 b1 b1 g1 y2 y1", "g1 g1 g3 g1 g1", "g1 g1 g1 g1"],
    "I": ["g1 g1 g1 g1", "y1 g1 xx g1 g1", "y1 y2 g1 xx g2 g1",
          "y1 y2 g1 g2 c1 xx xx", "y2 r2 xx xx g2 g1", "b2 b1 b1 g1 g1", "b2 b2 b1 g1"],
    "K": ["c1 g2 g2 g1", "g1 g1 b3 g1 g2", "g1 g2 g1 g3 g1 g2",
          "g2 g1 g3 g1 g3 g1 g2", "g2 g1 g3 g1 g2 g1", "g2 g1 y4 g1 g1", "g1 g2 g2 c1"],
}
# Clockwise rotations in sixty-degree steps, relative to the faces cited above.
# A common rotation of the entire board does not change its adjacency graph.
ROUTE = [("B", 2, (0, 0)), ("C", 5, (7, -3)), ("N", 2, (10, 1)),
         ("I", 2, (13, 5)), ("K", 0, (20, 2))]
FINISH_LOCAL = [(3, -4), (4, -4), (4, -3)]  # around K's upper-right corner
# Printed number breaks a tie; number of symbols is the entry cost.
BLOCKADES = [dict(rank=1, color="g", cost=1), dict(rank=2, color="y", cost=1),
             dict(rank=3, color="r", cost=1), dict(rank=4, color="b", cost=1),
             dict(rank=5, color="g", cost=2), dict(rank=6, color="r", cost=2)]

# price, movement color, power, once-only, action, draw count, optional trash count
_CARD_ROWS = [
    ("explorer", 0, "g", 1, False, None, 0, 0),
    ("traveler", 0, "y", 1, False, None, 0, 0),
    ("sailor", 0, "b", 1, False, None, 0, 0),
    ("scout", 1, "g", 2, False, None, 0, 0),
    ("trailblazer", 3, "g", 3, False, None, 0, 0),
    ("pioneer", 5, "g", 5, False, None, 0, 0),
    ("giant_machete", 3, "g", 6, True, None, 0, 0),
    ("photographer", 2, "y", 2, False, None, 0, 0),
    ("journalist", 3, "y", 3, False, None, 0, 0),
    ("millionaire", 5, "y", 4, False, None, 0, 0),
    ("treasure_chest", 3, "y", 4, True, None, 0, 0),
    ("captain", 2, "b", 3, False, None, 0, 0),
    ("jack_of_all_trades", 2, "w", 1, False, None, 0, 0),
    ("adventurer", 4, "w", 2, False, None, 0, 0),
    ("prop_plane", 4, "w", 4, True, None, 0, 0),
    ("transmitter", 4, None, 0, True, "transmitter", 0, 0),
    ("cartographer", 4, None, 0, False, "draw", 2, 0),
    ("compass", 2, None, 0, True, "draw", 3, 0),
    ("scientist", 4, None, 0, False, "draw", 1, 1),
    ("travel_log", 3, None, 0, True, "draw", 2, 2),
    ("native", 5, None, 0, False, "native", 0, 0),
]
CARDS = {n: dict(price=p, color=c, power=v, item=i, effect=e, draw=d, trash=t)
         for n, p, c, v, i, e, d, t in _CARD_ROWS}
MARKET_CARDS = [r[0] for r in _CARD_ROWS[3:]]
INITIAL_MARKET = ["scout", "trailblazer", "jack_of_all_trades", "photographer", "treasure_chest", "transmitter"]
STARTER = ["explorer"] * 3 + ["traveler"] * 4 + ["sailor"]
