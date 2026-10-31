# Five Tribes — base Fakir edition, two players

Status: **implemented_with_rulings_and_unresolved_boundaries**.
`config_id=base_fakir_duo_four_markers`, rules schema 1. This is the standard
30-tile, 90-meeple, 22-djinn, 54-resource game with two markers and 11 camels per
player. Expansions and promotional djinn are absent. Do not report this as an
unconditional complete rules certification: the unresolved cases below remain
observable nonterminal outcomes.

## Primary data recovery

- [Days of Wonder base rules](https://cdn.svc.asmodee.net/production-daysofwonder/uploads/2024/07/8401-FT-Rules-EN.pdf), setup pp. 1–2, bidding/movement pp. 3–5, tribes/tiles pp. 5–7 and ending p. 8.
- [Official 2016 help sheet](https://cdn.svc.asmodee.net/production-daysofwonder/uploads/2024/07/ft_summary_2016-04_en.pdf), p. 2, contains **all 22 card faces**, including printed VP. Text extraction misses those image-only numbers. We rendered that page at 2200 pixels and visually transcribed them into `families/data/five_tribes_base.json`. The PDF SHA256 is `327c57abd545dc496a75ae9a7473d237467048d0f51a046e5ce812e7f9f7b0ce`.
- [Official product](https://www.daysofwonder.com/game/fives-tribes-the-djinns-of-naqala/) links both documents and corroborates component quantities.

| Djinn | VP | Djinn | VP |
|---|---:|---|---:|
| Al-Amin | 5 | Anun-Nak | 8 |
| Ba'al | 6 | Boaz | 6 |
| Bouraq | 6 | Echidna | 4 |
| Enki | 8 | Hagis | 10 |
| Haurvatat | 8 | Iblis | 8 |
| Jafaar | 6 | Kandicha | 6 |
| Kumarbi | 6 | Lamia | 10 |
| Leta | 4 | Marid | 6 |
| Monkir | 6 | Nekir | 6 |
| Shamhat | 6 | Sibittis | 4 |
| Sloar | 8 | Utug | 4 |

No third-party engine code or card artwork is redistributed. The online
`spiffycoffee/fivetribes-ai` example was examined but rejected as a rules source:
its own README explicitly omits bidding, refills and djinn effects.

## API and implemented behavior

A sow is committed as `start_sow` followed by `drop` choices. Exact recursive
feasibility admits **all** choices with a legal completion. There is no score
pruning, trial placement, cancellation, diagonal movement, immediate backtrack
or mid-sow power use. Longer loops and matching an earlier dropped meeple work.
Snapshots can resume inside the sow. Four marker actions occur each round.

All 22 powers have implementations, including paid-power reuse per personal
turn, Iblis/Kandicha multi-kill, relocation during Bouraq/Enki, Sibittis choice,
Kumarbi bid discounts and exact Al-Amin wild-goods optimization. Resources and
djinn refill only after the whole round. Last-camel events finish the round.

Exact opposing coins/resources and random decks/bag draws stay hidden. Public
acquisitions remain in event history; Sloar/Kandicha private draws do not reveal
card identity. Sibittis's pending offer is visible only to its owner. Nonterminal
`step.info.scores` is null for both players to prevent a score-channel leak.
`scores()`/`score_breakdown()` are referee methods with privileged state.

## Interpretations and remaining boundaries

These are explicit versioned interpretations, not claims of publisher FAQ
approval. The [community FAQ](https://www.scribd.com/document/432691144/UNOFFICIAL-FIVE-TRIBES-FAQ-V1-pdf)
and [French FAQ](http://faqjeux.pbworks.com/w/page/84748465/Five%20tribes) corroborate
hidden resource hands, preselected Anun-Nak targets, separate personal-turn
power limits, relocation chains and an end trigger that cannot be cancelled.
They are secondary evidence; author-linked BGG pages were not fully retrievable.

| Case | Implemented interpretation / status |
|---|---|
| Kandicha kills a Builder | One victim's local blue-neighborhood income; no extra Fakir Builder multiplier. Echidna affects the acting player's Builder tribe, not the victim's hypothetical income. The latter scope remains provisional. |
| Lamia / Hagis neighboring tile | Eight surrounding cells, as indicated by the card's multi-direction icon; sowing and assassin distance remain orthogonal. |
| Ba'al acquires itself | No income on its own acquisition; its trigger exists for subsequent acquisitions. |
| No moves then Anun-Nak | Check at a player's movement opportunity / turn completion; once triggered, ending stays scheduled for this round even if later power use restores moves. |
| Fourth marker cannot pay and all zero slots are full | **Unresolved**: `no_affordable_bid_and_zero_slots_full`. No free bid, debt, forced sale or invented winner is inserted. The community FAQ proposes emergency handling but publisher/author evidence was not verified. |
| Anun-Nak can draw fewer than three, even after its payment returns an Elder | **Unresolved**: `anun_nak_bag_fewer_than_three_requires_ruling`. Stop before payment/random draws. |

An unresolved state has `phase=unresolved`, nonempty `blocked_reason`, no legal
actions, `terminated=false`, no winners and `outcome.unresolved_rule=true`.
Preserve it in completion statistics; never score it as a draw or discard the seed.

## Verification

`tests/families/test_five_tribes.py` checks quantities, all power families,
sequential sow legality, longer loops, phase snapshots, first capture before
assassination, market prefixes, exact goods scoring, information equivalence and
explicit unresolved behavior. Initial smoke batch (environment/policy seeds
0–3, both seat starts): **four of four heuristic games naturally completed;
three of four random games completed**. Random seed 0 reached the unaffordable
fourth-bid boundary after 197 decisions and is retained as unresolved. This is a
small correctness batch, not an estimate of playing strength or general
completion probability. The baseline is a simple public tactical heuristic.
