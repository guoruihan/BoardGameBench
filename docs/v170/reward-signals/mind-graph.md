# Reward signals: topic graph

## Terminal outcome and richer policy targets
- **Description**: Sparse game outcomes can coexist with search-produced action supervision.
- **Key papers**: alphazero2017; douzero2021 (terminal return regression without MCTS).

## Decomposing an outcome for representation learning
- **Description**: Predict finer variables that explain the final outcome.
- **Key papers**: katago2019 (§4.1); relation to our proposed score/component heads is an inference.

## More informative or earlier rewards
- **Description**: Distinguish terminal numeric payoff, actual score increments and predicted global-reward differences.
- **Key papers**: douzero2021 (payoff); hanabihle (score increments); suphx2020 (round attribution).

## Choosing the objective
- **Description**: Maximizing expected margin, probability of winning and rank are distinct objectives.
- **Key papers**: scorewin2022; douzero2021; katago2019 (Appendix F).
- **Related**: pluribus2019 uses payoff-based CFR, not PPO.

## Tabletop-specific context
- **Description**: Richer event evaluations and unified multi-game frameworks are relevant but are not interchangeable evidence about RL rewards.
- **Key papers**: rinascimento2020 (planning); rainbowarena2025 (framework; exact reward formulas not verified).
