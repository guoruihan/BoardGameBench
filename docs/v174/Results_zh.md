# V1.7.4：联合策略池交付与已有标定结果

冻结已完成：16 个固定双人任务、101 个启发式／学习／搜索配置，可按固定身份加载。强度和风格只按现有证据填写，不强行凑满三档。方法及调用见 [Joint_Policy_Pool_zh.md](Joint_Policy_Pool_zh.md)。

## Toy Battle 后续授权的 32 局已完成

用户随后明确要求将原来的 Toy Battle 搜索对旧 RL 这 32 局全部同步启动。本次仅执行这一组：相同冻结策略、16 个原环境种子交换座位、每局累计 900 秒。b35 上 32 个单核 worker 在 1.75 秒内全部启动，**15.11 分钟全部结束**，没有超时、超预算或未知结局。

搜索 **27 胜 5 负，胜率 84.375%**；按原协议的环境配对名义区间为 **60.41%–95.03%**，满足“高于旧 RL 锚点”的原分档条件。双方平均勋章数为 6.6875:2.34375；Toy Battle 还允许直接占领获胜，不能以勋章均值代替胜负。搜索平均决策用时 869.94 秒，实际搜索计时最大 898.002 秒。本次增加 7.80 worker 核时，与第一次停止前累计合计 159.40 worker 核时；没有使用 GPU。

本组结果保存在 `toy32_resume/summary.json`；更新后的完整对手目录为 `toy32_resume/opponent_catalog.json`，仅调整 Toy Battle 搜索的强度及高档角色，其他游戏和全部策略身份保持原样。`completion_verification.json` 记录 32 局结果哈希、策略身份及预算核对。旧 `report_final/` 保留第一次停止时的快照；下表 Toy Battle 一行已经更新，其余行仍来自该快照。

## 范围纠正与实际完成量

本次将用户要求的一轮并行检查扩大为多预算、多种子重新标定，这是执行范围错误。原计划含 608 场 900 秒对局，189 个同时运行的 worker 无法在一波完成。用户指出后立即停止三个队列，保留完成结果；提议的额外 Spice Road／边界面板从未启动，现已取消。剩余队列不恢复，不因分档未决而追加实验。

- 原计划 11,872 局；已保存 10,294 局；状态：`{'complete': 10290, 'budget_exceeded': 1, 'unresolved_rule': 3, 'not_run': 1578}`。
- 停止时有 189 个活动子任务；未保存完整结果的项目保留为未知。`not_run` 同时包含未启动和被中断的局，不冒充技术失败或终局输赢。
- 三个队列从启动到全部停止约 48.3 分钟；累计 **151.60 单 worker 墙钟小时**，包含加载、I/O 和被中断工作。已落盘比赛 CPU 合计 134.50 小时，是排除未落盘中断工作后的较小口径，不能与前者相加。
- 每台保留半数物理核心；运行快照中整机 CPU 约 25–32%，未发现 worker 亲和性违规。未使用 GPU，未训练新模型。协调、汇总及开发检查未纳入上述 worker 核时。
- 900 秒是每局累计搜索决策上限。此轮 30／180 秒对局未完成前即停止；配置已冻结，但不能冒充本轮已测结果。历史短预算结果仍保留在 V1.7.2。

## 900 秒搜索对冻结学习锚点

下表胜率为已完成有效局的胜点比例（平局半分），不是包含所有未知结局的无偏估计。完整清单、未知结果上下界及配对区间保存在 `report_final/calibration.json`。共用同一批初始环境并交换座位；实际后续随机事件可随行动分歧。

| 游戏 | 有效完成/原计划 | 搜索胜率 | 本轮强度判断 |
|---|---:|---:|---|
| calico | 32/32 | 87.50% | 高于锚点 |
| cascadia | 32/32 | 53.12% | 未拉开差距 |
| century_a_new_world | 31/32 | 96.77% | 观察到超预算，不纳入合格分档 |
| century_spice_road | 32/32 | 71.88% | 未拉开差距 |
| clever | 32/32 | 96.88% | 高于锚点 |
| dominion | 32/32 | 31.25% | 未拉开差距 |
| el_dorado | 32/32 | 53.12% | 未拉开差距 |
| gizmos | 32/32 | 62.50% | 未拉开差距 |
| harmonies | 32/32 | 90.62% | 高于锚点 |
| jaipur | 32/32 | 65.62% | 未拉开差距 |
| machi_koro | 32/32 | 59.38% | 未拉开差距 |
| patchwork | 32/32 | 100.00% | 高于锚点 |
| splendor | 31/32 | 90.32% | 证据不足 |
| stone_age | 32/32 | 90.62% | 高于锚点 |
| targi | 32/32 | 18.75% | 低于锚点 |
| toy_battle | 32/32 | 84.38% | 高于锚点 |

Harmonies 另保留自身分数目标的 habitat 搜索：有效 32/32 局，胜率 60.94%，双方平均分 94.25:90.44。与胜负目标搜索是不同配置，不混为一项。

## 保留的对手角色

锚点始终为 `legacy`。上下档要求完整面板、至少 16 个独立环境、胜点比例至少 60%／至多 40%，且探索性名义区间在 50% 同侧；未决不等于等强。区间未做多重比较校正。这是对一个锚点的局部关系，不是全局可传递排名。

| 游戏 | 锚点以下 | 锚点以上 | 有分项差异的互补候选 |
|---|---|---|---|
| calico | — | search_900 | — |
| cascadia | heuristic, progress | — | — |
| century_a_new_world | heuristic, progress | — | — |
| century_spice_road | heuristic, progress | — | candidate, search_900 |
| clever | heuristic, progress | search_900 | candidate, search_900 |
| dominion | heuristic, progress | — | — |
| el_dorado | heuristic, progress | — | — |
| gizmos | heuristic, progress | — | search_900 |
| harmonies | heuristic | search_900 | progress, search_900, search_score_900 |
| jaipur | progress | — | — |
| machi_koro | heuristic | — | search_900 |
| patchwork | heuristic, progress | search_900 | search_900 |
| splendor | heuristic, progress | — | — |
| stone_age | — | search_900 | progress, search_900 |
| targi | search_900 | — | — |
| toy_battle | heuristic, progress | search_900 | candidate |

`candidate` 是保留的 V1.7.3 新策略，并不保证在本次面板上显著优于锚点；旧版较大确认结果与本次样本分开保存。不能只凭算法名称把搜索设为高档、RL 设为中档。

## 风格与边界

- Harmonies habitat 启发式与学习锚点呈现动物／地形分项差异，可作为互补对手；这不是已经证明同等总能力或因果性策略偏好。具体分项均值及共同种子数见自动报告。
- New World 有搜索实际累计用时 907.93 秒的 900 秒配置记录，即使赢了也不计为有效预算内比赛；该成员标记 `budget_violation`，不进入推荐上下档。未修补或重跑这一冻结成员。
- Jaipur 的规则未决保留原状态；Five Tribes 保持研究态、不纳入此池。不能把不完整规则边界藏在平均胜率里。
- 部分长搜索面板在停止时未完成，因此对应角色暂不晋级。历史证据仍可查阅，不自动补局。
- 正式 test 未用于本轮。后续 agent 开发评测仍需固定交互／训练／推理额度及隔离测试种子。

## 文件与验证

交付根目录为 `outputs/v174/joint_pool_20261101/`：`pool/manifest.json`、`pool/models/`、`pool/source.tar.gz` 为可搬移的策略与真实运行源码；`report_final/` 为第一次停止时的角色索引和统计；`toy32_resume/opponent_catalog.json` 为纳入后续授权 32 局的最新角色索引；`results/` 为原始逐局文件；`user_scope_stop.json`、`finalization.json` 为第一次停止与成本记录。权重、原始轨迹和服务器信息留在本地忽略目录。

安装包：`outputs/v174/package/boardbench-0.6.7-py3-none-any.whl`。实际比赛运行源码与最终 CLI／分析源码分别保存；最终分析身份见 `analysis_release_final/identity.json`。

- 此前完成的 132 项相关检查已通过，包括嵌套先验搬移、哈希、公开信息、未决结果、搜索预算归因及队列停止。
- 已有 152 条实际保存轨迹、21,944 个决策经公开 `env.step` 重放一致；这是停止前完成的快速局检查，不声称最终所有搜索轨迹都已重放。
- 101 个真实冻结对手均可加载并给出合法初始动作／搜索根建议；最终独立 wheel 的对应输出与真实运行源码一致。安装包 345 个载荷文件已逐一核验。
- 三个节点的本轮队列及对局进程已清空。辅助收尾 watcher 曾因启动时队列状态文件尚未生成而退出；最终报告由停止后的直接汇总生成，不依赖该 watcher，也不再启动任何新比赛。

策略池 SHA256：`bff20b640b62f083d1902a03a29b4dbb2df3486d0b463c49f9d619aa5d5a5141`。
第一次停止时结果身份：`70bb043f512e8e272421bd1cc86f0472aa76f0908b21b53d928c3d0d5c26134b`。
后续授权 Toy Battle 32 局结果身份：`52e2a94ddb6648b8ce75b3103a9e3b6387a47cc6c53f6c46fbefabb70fc5175d`。
