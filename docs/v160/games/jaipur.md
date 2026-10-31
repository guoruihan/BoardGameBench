# Jaipur: Space Cowboys 2019, two-player match

The environment plays the official match to two seals, resetting all round cards
and tokens between rounds and letting the previous loser start. Official match
`scores` are seal counts; `score_breakdown` also reports rupees. It includes full
multiset exchanges, voluntary large sales, depleted goods/bonus piles, camel
majority, immediate round ends and both printed round tiebreakers.

Exact component counts and token multisets are in `families/data/jaipur.json`.
The [publisher game page](https://www.spacecowboys-games.com/game/jaipur/) and
[publisher English rules](https://cdn.svc.asmodee.net/production-spacecowboys/uploads/2025/11/Rules-JAIPUR-12x17-Version-EN_BD.pdf)
identify the 2019 edition and rules. Bonus frequencies were cross-checked against
the [firsthand photographed component inventory](https://homeofmark.com/jaipur-card-game-review/)
on 2026-10-31: 3-card bonuses have 1,1,2,2,2,3,3; 4-card bonuses have
4,4,5,5,6,6; 5-card bonuses have 8,8,9,10,10. No external engine code is copied.

Own hand, own herd, own earned bonus values, market, goods piles, discard and
public action history are observable. Opponent hand identities, exact herd,
private bonus values, undrawn card identities and bonus order are withheld.
Public exchange/take/sale history remains available for inference. Full snapshots
and score breakdowns are privileged evaluator data, never policy inputs.

## Explicit unresolved rule edge

The 2019 rules break a rupee tie by bonus-token count, then goods-token count.
Neither the official rules nor the bounded publisher/FAQ search specified what
happens if all three are equal. This implementation leaves that round
`blocked_reason=unresolved_round_tie`, `outcome.unresolved=true`, with no legal
actions, **without declaring a winner or a natural termination**. Evaluation must
count it as an unresolved run, not discard it or assign a reward. This boundary
prevents claiming exhaustive official-rule coverage; it is not a house rule.
Camel-count ties, which are explicitly specified, give neither player 5 rupees.

Tests cover component conservation, full seeded matches, large-sale bonuses,
hand limits and exchange exclusions, deck refill timing, match seals, tiebreaks,
hidden information, atomic illegal actions and JSON snapshot replay. There is no
artificial turn limit in game rules. External caps remain truncations.
