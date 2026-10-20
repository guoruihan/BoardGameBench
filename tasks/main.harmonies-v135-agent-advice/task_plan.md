# V135: agent-authored action advice and PPO trust

## Stable question and success signal

Can public, interpretable strategic judgment improve a mature model's actual
Harmonies score, beyond matched continued PPO, with measurable beneficial
dependence on that judgment? No game/reward/information changes or paid LLM calls.
This version tests coding-agent-authored structured rules, NOT online LLM reasoning.

Success requires fixed-endpoint, paired fresh512 results versus both parent and
continued-PPO control, and development-only removal/shuffle diagnostics. Two
training seeds are exploratory; environment intervals are not seed uncertainty.
Do not call success merely because advised training beats its own starting score.

## Experiment

- Predesignated V134 reactive_a final parent; preserve actor, critic, adapter and
  existing Adam moments. A-replica convention, not selection on new confirmation.
- Four methods × two seeds: no action advice, fixed soft advice, learned bounded
  trust, learned trust plus two-turn persistent goals.
- Stage1 +4M actions each; autonomously continue control and strongest noncontrol
  on a separate development audit512, stage2 +8M actions each.
- Fresh curve512 + audit512 + confirmation512 in [50M,60M), all old train/test
  pools unchanged. Formal test sealed. Confirmation cannot drive promotion.
- Advice uses public blueprint and action codec; never overrides legality.
  End-turn discard refers to animal MARKET, not active cards.
- a226 GPUs0–3 only, 8×(6 workers+1 learner)+1 controller =57 physical cores,
  below half-machine64. Stage2 uses GPUs0–1 only.
- Per-stage limits3600/4800s; campaign hard guard3h, finalization reserve1200s.
  No automatic deadline extension; all costs and failed preflights recorded.

## Phases

- [x] Inspect V134 results, source and live resources.
- [x] Specify interpretable goal-to-action advice and independent controls.
- [x] Unit/regression tests, mature CUDA update/resume and diagnostic gate.
- [x] Freeze source, commit small version, launch bounded autonomous campaign.
- [x] Verify real progress:all8 learners,6–12 PPO batches,586264 added actions.
- [ ] Read completed paired results and decide next cycle.

## Evidence

V134 reactive confirmation111.0801 vs control110.4531, but zero-plan intervention
did not hurt either reactive replica on development. Beneficial inference-time
goal usage is unproven. Direct advice is a falsifiable alternative to more features.
2026-10-21 04:44CST a226 GPU0–3 idle14MiB; GPU4–7 busy unrelated; load13.49.

## Next action

Analyze stage promotion and fresh confirmation after this autonomous campaign
finishes. No further user decision is needed to run the current two stages.
