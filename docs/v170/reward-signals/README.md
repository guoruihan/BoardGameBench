# 桌游与多人游戏 RL：奖励和辅助监督检索

检索日期：2026-11-01。范围：原始论文与作者官方实现；没有新增训练或改动游戏规则。

## 结论

仅使用终局胜负是一条成功路线，但不是桌游 RL 的统一标准。
现有工作还采用结算分数、逐步得分变化、最终结果的分解预测，以及预测最终收益的变化。
必须区分环境奖励、actor 优化目标、critic/共享表示的辅助损失、搜索效用和最终评测指标。

| 工作 | 已核实的信号 | 不能据此声称 |
|---|---|---|
| [AlphaZero](https://ar5iv.labs.arxiv.org/html/1712.01815) | value 学终局胜负/和；policy 学 MCTS 访问分布 | 只有一个二值标签、没有其他策略训练目标；或这证明所有游戏都只需胜负 |
| [KataGo](https://ar5iv.labs.arxiv.org/html/1902.10565) | §4.1 最终分差分布、棋盘最终归属辅助目标；附录 F 搜索效用同时包含胜负与非线性分差效用 | 只有 score critic，或其全部加速来自分数辅助头 |
| [Suphx](https://ar5iv.labs.arxiv.org/html/2003.13590) | §3.2 四人麻将用预测全场排名收益的变化给各小局分配奖励 | 直接把每小局点数当作最终目标；每个原子动作都有独立奖励 |
| [DouZero](https://ar5iv.labs.arxiv.org/html/2106.06135) | 分别训练终局胜负 WP 与结算点数 ADP；附录 D.2 比较目标与打法差异 | ADP 是逐步密集奖励，或更高期望点数必然提高胜率 |
| [Hanabi 官方 HLE](https://github.com/google-deepmind/hanabi-learning-environment/blob/master/hanabi_learning_environment/rl_env.py) | `HanabiEnv.step` 返回 `state.score() - last_score`；终局可能有负向修正 | 所有后来 Hanabi 论文都沿用同一包装器或相同失败计分 |
| [Pluribus](https://noambrown.github.io/papers/19-Science-Superhuman.pdf) | 六人扑克，以支付收益进行 MCCFR 自弈与搜索 | 它是 PPO/actor-critic，或证明了密集奖励的优势 |

## 对本项目的推论：尚未实验验证

最直接可借鉴的是 KataGo 的结果分解监督，而不是把所有内部数值相加成奖励。
Harmonies 的动物/地形得分、双方终局分数和分差可作为候选辅助标签。
需要更细空间监督时，可以考虑每张动物卡的最终完成情况，但这是本项目的新设计，非 KataGo 已验证结论。

同一训练框架可以支持可选标签与缺失掩码，计分游戏按训练集确定尺度。
输入遵守当前玩家观察范围；终局可揭示的结果可以用于回溯监督，不把未来结果作为决策输入。
没有原生数值胜利分的游戏不能伪造 score；阶段点数、局数胜利和全场排名须分别解释。

建议保留三个受控实验：

1. 原胜负 PPO。
2. 同样的胜负 PPO + 归一化终局分数/分差/分项辅助预测；胜负头继续负责胜负 GAE。
3. 在同样表示下，明确加入分数或分差目标，匹配对应的 reward、value 和 advantage；承认其改变训练偏好。

三组固定对手池、初始模型和 learner 决策预算，单独记录辅助任务计算成本。
先诊断 actor 行为是否改变、critic 预测是否校准，再检查新开发 seed 上的胜率、分差、总分和分项得分。
当前 Harmonies actor 行为几乎不变的问题并未因文献检索而解决；新增损失不能替代梯度与先验约束诊断。

终局分数比胜负携带更多信息，但标签仍可能只在终局生成；这与逐步密集奖励不同。
分数 critic 不能直接替换胜负 GAE 中的 value。若采用保持原目标的 potential-based shaping，
须另外核对折扣、终局边界及中断处理；不能把任意逐步加分称为保持策略不变的塑形。

## 相关反证和检索边界

- [Score vs. Winrate](https://arxiv.org/html/2201.13176v2) 直接研究纯分差与胜负目标的区别，讨论不确定性下的风险偏好，并报告围棋中的实验；不能把该结论推广成“分数辅助任务无用”。
- [Rinascimento](https://ar5iv.labs.arxiv.org/html/2006.05894) 对 Splendor 的事件价值提供了相关思路，但主要实验是统计前向规划与 NTBEA 调参，不是 PPO 证据。
- [RainbowArena, AAMAS 2025](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2639.pdf) 提供多款桌游的统一自弈接口，并允许自定义 reward；本次读取的扩展摘要未披露逐游戏奖励公式，不能由其成绩反推奖励设置。
- 检索覆盖 score utility、payoff、global reward prediction、auxiliary targets、credit assignment 和事件价值等角度，并沿 KataGo/DouZero 讨论与参考文献补查。不是系统综述，也没有统计“多数论文”采用哪种路线。
- Semantic Scholar API 本次不可访问；改用论文全文、会议页面及官方代码。没有虚构引用量，没有下载或提交 PDF。

条目与证据定位见 [文献索引](memory-bank.md)，主题关系见 [主题图](mind-graph.md)，引用信息见 [BibTeX](references.bib)。
