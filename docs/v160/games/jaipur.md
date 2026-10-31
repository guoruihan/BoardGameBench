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

### Exact-tie audit, 2026-10-31

The `league_dev_r1` evaluation encountered this boundary in 2 of 384 scheduled
matches (0.52%). Replaying both with the recorded policies and policy seeds
reproduced the decisions, score breakdown and unresolved outcome exactly;
component-conservation checks passed after every action. The two cases both have
`heuristic` in seat 0 and `cem_r1` in seat 1, orientation 1:

| Environment seed | Round | Rupees, each | Bonus tokens, each | Goods tokens, each | Seals before boundary |
| --- | --- | --- | --- | --- | --- |
| 33000005 | 3 | 74 | 3 | 14 | 1 : 1 |
| 33000055 | 1 | 75 | 3 | 16 | 0 : 0 |

The camel token's five rupees are already included in both cases; the token is a
separate component category and does not add to either printed tiebreak count.
The [current publisher French rules, page 6](https://cdn.svc.asmodee.net/production-spacecowboys/uploads/2025/12/Rules-JAIPUR_SCJAI01FR_HD_ss_reperes.pdf)
agree with the English edition. The [original GameWorks French rules, PDF page 4](https://www.gameworks.ch/JAIPUR/JAIPUR_FR.pdf)
also stop after goods-token count. None of these books specifies a winner,
shared seal, rematch, or next starting player after an exact tie.

A bounded search of publisher/designer sources found no verifiable additional
ruling. A fan FAQ points to [BGG thread 548575](https://boardgamegeek.com/thread/548575),
but its original content and author identity could not be checked (HTTP 403 / web
fetch unavailable); it is **an unverified lead**, not evidence of a designer ruling.
This audit does not establish that no clarification exists anywhere. The rule
implementation therefore remains unchanged; these runs remain unresolved and
count in the full evaluation denominator, with no draw reward.

Local audit artifacts are under
`outputs/v160/bots/jaipur/league_dev_r1/rule_audit_20261031/`: reproducible action
traces, privileged snapshots, source PDF hashes and `REPORT.md`. They remain
outside Git and are not policy inputs. The tests retain the two scoring-inventory
cases and verify no seal award, new round, RNG advance or normal termination,
including JSON restoration of the unresolved state.

Tests cover component conservation, full seeded matches, large-sale bonuses,
hand limits and exchange exclusions, deck refill timing, match seals, tiebreaks,
hidden information, atomic illegal actions and JSON snapshot replay. There is no
artificial turn limit in game rules. External caps remain truncations.
