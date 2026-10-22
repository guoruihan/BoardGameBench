# Task Plan: V142 own-score learning under random market depletion

## Stable Anchor
- Question: can mature solo/duo knowledge improve own layout under stationary random market interference, before adversarial training?
- Outcome: separate reproducible curriculum environment, matched initial-model evaluation, four actual bounded PPO runs with own-score selection.
- Constraints: keep official solo/duo rules and published artifacts intact. a226 GPUs0–3, <=half physical CPUs (57 total); new four-hour wall cap including gates/evaluation/save.
- Non-goals: claim equivalent scores to official solo/duo, adversarial training in this phase, changing terrain/animal scoring, tuning on old confirmation/test environments.

## Current Cycle
- Hypothesis: eliminating opponent-board effects and aligning reward/selection to own score exposes and trains layout capability.
- Protocol: five public slots; after each nonterminal own turn refill own vacancies, uniformly remove one resource bundle and one available card, then refill; 13 own turns maximum or own board <=2 empty; no opponent board/score.
- Decisive experiment: solo-derived versus trained duo-own-score initialization, two matched replicas each, fresh development and confirmation seeds.
- Expected signal: improved own score on paired development, then independent512 environments; animal score/turns/tokens/completions diagnosed separately.
- Fallback: preserve failed gates; do not launch formal training if semantics/behavior/resume fail.

## Success Criteria
- [ ] Reference/fast/native/pool parity, conservation, hidden RNG isolation, masks and natural terminal tests pass.
- [ ] Actual CUDA gradients, behavior ratios and exact resume pass on authorized cards.
- [ ] Initial common-environment comparison recorded before training; four jobs genuinely update with recovery/checkpoints and all trajectories.
- [ ] Coherent commits pushed and remote SHA checked; verified run handoff documented.

## Parallel Tracks
- Primary only; no delegated agents. Run tests concurrently with documentation after code freeze.

## Phases
### Phase 1: Environment and training implementation
- **Status:** complete
- **Evidence:** independent environment; V142 own-score trainer/encoding/policy/campaign; no original-game edits.
### Phase 2: Verification and baseline comparison
- **Status:** in_progress
- **Evidence:** initial39 CPU tests pass/1 actual-CUDA skip; fresh a226 probe GPUs0-3 idle,4-7 occupied. Expanded final regression running.
### Phase 3: Bounded launch and publication
- **Status:** pending
- **Evidence:** none

## Verification Contract
- Existing-game regressions unchanged; dedicated V142 full-game parity and public-only feature tests.
- Complete trajectories before PPO; true score differences telescope; own score alone selects best; no opponent reward or animal bonus.
- New independent source/config/weights hashes, random streams, split namespaces; no silent optimizer continuation across different objectives/environments.
- Full exact within-version restore and deadline accounting; new task, not renewal of expired V141 budget.

## Next Action
Finish regression, publish frozen source, launch actual-CUDA gates then common128 initial comparison and four bounded training jobs.
