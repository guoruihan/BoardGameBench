# Reward signals literature index

Last updated: 2026-11-01. Citation counts were not verified. Descriptions below are paraphrases, not abstracts copied from publishers.

### [alphazero2017] Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
- **Authors**: David Silver, Thomas Hubert, Julian Schrittwieser, et al.
- **Venue/year**: arXiv preprint, 2017; this entry cites that version, not the later Science bibliographic record.
- **URL**: https://arxiv.org/abs/1712.01815
- **Status**: reward/value/policy targets checked in full-text main algorithm, Eq. (1).
- **Topics**: terminal outcome; MCTS policy supervision.
- **Abstract**: Self-play search trains policy and value for chess, shogi and Go.
- **Notes**: Value target is loss/draw/win; policy learns the search distribution. Sparse outcome does not mean the network has no other training target.

### [katago2019] Accelerating Self-Play Learning in Go
- **Authors**: David J. Wu.
- **Venue/year**: arXiv preprint, 2019; also presented at the AAAI 2020 Reinforcement Learning in Games workshop.
- **URL**: https://arxiv.org/abs/1902.10565
- **Status**: full-text §4.1, §5.2 and Appendix F checked.
- **Topics**: auxiliary score distribution; territory ownership; mixed search utility.
- **Abstract**: Multiple search, architecture and target improvements accelerate Go self-play.
- **Notes**: Ablations support ownership/score auxiliary targets. Appendix F also adds bounded nonlinear score utility to win/loss utility in search. These are separate mechanisms, neither simply fitting current displayed score.

### [suphx2020] Suphx: Mastering Mahjong with Deep Reinforcement Learning
- **Authors**: Junjie Li, Sotetsu Koyamada, Qiwei Ye, Guoqing Liu, Chao Wang, Ruihan Yang, Li Zhao, Tao Qin, Tie-Yan Liu, Hsiao-Wuen Hon.
- **Venue/year**: arXiv preprint, 2020.
- **URL**: https://arxiv.org/abs/2003.13590
- **Status**: full-text §3.1–3.2 and Eq. (4) checked.
- **Topics**: four-player imperfect information; global reward prediction; round credit assignment.
- **Abstract**: Mahjong policy learning combines several techniques to address hidden information and multi-round outcomes.
- **Notes**: A predictor trained on expert human logs estimates final game reward using round history. Differences between consecutive predictions supply round rewards. The paper explicitly explains why raw round points need not align with overall rank.

### [douzero2021] DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning
- **Authors**: Daochen Zha, Jingru Xie, Wenye Ma, Sheng Zhang, Xiangru Lian, Xia Hu, Ji Liu.
- **Venue/year**: ICML, PMLR 139:12333–12344, 2021.
- **URL**: https://proceedings.mlr.press/v139/zha21a.html
- **Status**: full-text §3, evaluation objectives/footnote 7 and Appendix D.2 checked.
- **Topics**: three-player competition/cooperation; terminal payoff; Deep Monte-Carlo.
- **Abstract**: Full-episode returns supervise action-value networks for large legal card-combination spaces.
- **Notes**: WP uses terminal ±1; ADP uses point payoff including bomb multipliers. Both are terminal rewards. The paper trains both and observes different play styles and objective-specific strengths; Deep Monte-Carlo here is not MCTS.

### [hanabihle] Hanabi Learning Environment
- **Authors**: Google DeepMind contributors.
- **Venue/year**: official software repository; inspected 2026-11-01.
- **URL**: https://github.com/google-deepmind/hanabi-learning-environment/blob/master/hanabi_learning_environment/rl_env.py
- **Status**: `HanabiEnv.step` implementation checked.
- **Topics**: cooperative multiplayer; incremental score reward.
- **Abstract**: Reference environment exposes observations, legal actions and rewards for Hanabi agents.
- **Notes**: Reward is next score minus previous score; the code notes potentially large negative terminal rewards. This evidence describes the environment, not every downstream paper or variant.

### [pluribus2019] Superhuman AI for multiplayer poker
- **Authors**: Noam Brown and Tuomas Sandholm.
- **Venue/year**: Science 365:885–890, 2019.
- **URL**: https://noambrown.github.io/papers/19-Science-Superhuman.pdf
- **Status**: author-hosted full text; Fig. 2 and MCCFR section checked through PDF text extraction.
- **Topics**: six-player imperfect information; payoff utility; counterfactual regret.
- **Abstract**: Self-play and real-time search yield strong six-player poker play.
- **Notes**: Learns from payoff differences rather than hand-win labels. It is an MCCFR/search system, not a PPO actor-critic or evidence for dense intermediate rewards.

### [scorewin2022] Score vs. Winrate in Score-Based Games: which Reward for Reinforcement Learning?
- **Authors**: Luca Pasqualini, Gianluca Amato, Marco Fantozzi, Rosa Gini, Alessandro Marchetti, Carlo Metta, Francesco Morandin, Maurizio Parton.
- **Venue/year**: ICMLA 2022; inspected arXiv v2 dated 2023-01-09.
- **URL**: https://arxiv.org/html/2201.13176v2
- **Status**: introduction, LZ-Score setup and conclusions checked.
- **Topics**: expected score versus outcome; uncertainty and risk; objective choice.
- **Abstract**: Studies why score optimization and winning can diverge under stochasticity or approximate play.
- **Notes**: Includes a score-trained Go system and analysis of variance preferences. This cautions against equating score maximization with win-rate maximization; it does not refute auxiliary score prediction.

### [rinascimento2020] Rinascimento: using event-value functions for playing Splendor
- **Authors**: Ivan Bravi and Simon M. Lucas.
- **Venue/year**: IEEE Conference on Games, 2020.
- **URL**: https://arxiv.org/abs/2006.05894
- **Status**: introduction and §IV experimental methodology checked.
- **Topics**: sparse score signals; event evaluation; forward planning.
- **Abstract**: Event-value functions provide alternatives to score-change evaluation for Splendor planning agents.
- **Notes**: Main experiments tune BMRH/event functions with NTBEA, including four-player tests. Keep distinct from deep RL or PPO ablations.

### [rainbowarena2025] RainbowArena: A Multi-Agent Toolkit for Reinforcement Learning and Large Language Models in Competitive Tabletop Games
- **Authors**: Yingzhuo Liu, Shuodi Liu, Hongsong Tang, Yubing Ma, Zikang Li, Junge Zhang, Liuyu Xiang, Zhaofeng He.
- **Venue/year**: AAMAS extended abstract, 2025.
- **URL**: https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2639.pdf
- **Status**: §2 and §3 checked through PDF text extraction; reward details not established.
- **Topics**: unified tabletop interfaces; multiplayer self-play; benchmark tooling.
- **Abstract**: A toolkit combines six tabletop environments, RL/LLM agents, self-play and multi-agent evaluation.
- **Notes**: Allows custom rewards but the inspected abstract does not specify per-game formulas. Included for framework relevance, not as positive evidence for a particular reward choice.
