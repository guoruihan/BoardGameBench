# V160 双人环境族：配置与覆盖边界

本环境族包含注册表中的 **17 款双人游戏实现**。每款固定一个原版官方双人配置，未将扩展、其他地图、其他计分卡或所有推荐设置一并纳入。这里的“官方配置”说明所选游戏内容的来源；实现仍可能包含已披露的规则解释或未决情况，不能据此宣称所有规则边界都已获得官方认证。

本文依据当前注册表和各游戏说明整理，不记录持续变化的训练成绩。环境正确性、运行完成率与对抗 bot 的能力是三个不同的检查项；安装了实现、通过规则测试或击败随机策略，都不等于已经获得强参考选手。启动、训练与恢复见 [Runbook](Runbook.md)，冻结对手、配对比较及结果统计见 [Evaluation](Evaluation.md)。

## 固定配置一览

表中游戏名称链接到详细规则覆盖、组件来源、测试和边界说明。配置名用于识别环境，不应在同一评测中被静默替换。

| 游戏 / `game_id` | 固定 `config_id` | 本轮覆盖与主要限制 |
|---|---|---|
| [Cascadia](games/cascadia.md) / `cascadia` | `standard_all_a_duo` | 标准双人、五种动物均用 A 计分卡；包含 85 块基础栖息地、五块起始板和 100 个动物，按双人规则抽取本局组件。每人 20 回合；其他计分卡、家庭规则和情景不在范围内。采用已核实的设计师动物计分与换牌澄清。 |
| [Calico](games/calico.md) / `calico` | `standard_teal_purple_sizecats_duo` | 固定青绿、紫色两块原版板及 Millie、Tibbit、Coconut 三只猫；猫的图案分配随机，使用全部 108 块布料，保留从四个目标中选择并排列三个目标的设置决策。覆盖六种目标，未包含其他板和其余七只猫。 |
| [Patchwork](games/patchwork.md) / `patchwork` | `classic_corrected_duo` | 经典双人、全部 33 块拼布、9×9 板、五块皮革和 7×7 奖励；时间轨使用出版方修正后的皮革位置。满板无法放皮革时将该皮革移出，证据边界见详细说明；不包含 Express、Doodle、促销组件或 Automa。 |
| [Harmonies](games/harmonies.md) / `harmonies` | `side_a_base32_no_spirits_duo` | 双方使用 A 面、32 张基础动物卡、不使用自然精灵；复用已有加速双人引擎和规则标识。通用环境族的地形启发式不等于此前保留的 Harmonies RL / 搜索参考策略。 |
| [Century: Spice Road](games/century_spice_road.md) / `century_spice_road` | `base_2017_duo` | 2017 独立基础版、43 张市场商人卡和 36 张订单卡；保留升级、重复交易、跳过卡的支付、休息和容量丢弃。交易至少执行一次；零次交易的无效果出牌未获明确规则支持。排除 Golem、组合模式、促销和 Golden Deals。 |
| [Jaipur](games/jaipur.md) / `jaipur` | `space_cowboys_2019_duo_match` | 2019 版完整先获两枚印章的比赛，含回合重置、交换、出售、奖励牌与骆驼结算。**卢比、奖励标记数、货物标记数全部相同的回合没有已核实裁决，保留未决，不制造赢家。** |
| [Splendor](games/splendor.md) / `splendor` | `classic_2014_duo` | 2014 经典基础版双人，完整 90 张发展卡、10 位贵族；本局随机选三位贵族。含暗抽保留、金币替代支付、超限返还和贵族选择。不是独立游戏 Splendor Duel，也不含扩展。 |
| [Gizmos](games/gizmos.md) / `gizmos` | `cmon_2018_duo` | 2018 基础版全部 108 张等级卡与四张起始卡实例；按标准设置使用 36/36/16 张等级牌库及双方起始卡。含转换器组合、触发顺序、研究与牌库底排序。分配器内部用颜色多重集的均匀无放回抽样建模；大量转换器的精确支付枚举可能耗时。 |
| [Machi Koro](games/machi_koro.md) / `machi_koro` | `classic_2014_base_duo` | 2014 英文经典版，15 类设施、84 张市场卡、四地标及标准初始资产。首个完成四地标者立即获胜。**新买游乐园不追溯触发本回合已接受的双骰结果**是明确标记的 provisional 解释；不含 Harbor、后续版本或随机市场。 |
| [That's Pretty Clever](games/clever.md) / `clever` | `original_sheet_duo_six_rounds` | 原版 Schmidt 计分纸，双人六轮；包含双方主动/被动机会、奖励链、重掷与末轮额外骰窗口。未启用 Challenge 纸、续作或单人变体。 |
| [Five Tribes](games/five_tribes.md) / `five_tribes` | `base_fakir_duo_four_markers` | 原版 Fakir 配置、30 地块、90 工人、22 精灵、54 资源；每人两个顺序标记和 11 骆驼。**研究态，规则版本 2 仍有可到达的未决边界**；不能作为无条件规则完整的正式环境，详见下文。 |
| [Century: A New World](games/century_a_new_world.md) / `century_a_new_world` | `original_A1_B1_C1_D1_duo` | 2019 独立版 A1/B1/C1/D1、玩家板 A 面；23 个位置、22 张符合双人筛选的得分卡、17 块独立版奖励与 10 块探索。保留工人挤出、帮助/工具、堡垒选择和探索。未覆盖 E1/F1、三四人、组合模式或大盒 G1/H1。 |
| [Stone Age](games/stone_age.md) / `stone_age` | `classic_base_duo` | 经典基础双人，完整 36 张文明卡及 28 块建筑，本局随机使用两叠各七块建筑；保留双人位置限制、掷骰后工具选择、喂养和反应窗口。人口/农业/工具已满时仍可合法占位但不再增加。未包含扩展、周年冬面或固定回合变体。 |
| [Targi](games/targi.md) / `targi` | `original_2012_duo` | 原版 16 边界、45 部落和 19 货物牌，完整三行布局、四次抢劫及两种自然结束。额外打出保留牌使用原版“多付任意普通货物、每轮一次”措辞；不宣称覆盖后续英文某些印次的“指定类型货物”措辞。少量边界解释见下文。 |
| [Dominion](games/dominion.md) / `dominion` | `second_edition_first_game_duo` | 第二版官方 First Game 王国：Cellar、Market、Merchant、Militia、Mine、Moat、Remodel、Smithy、Village、Workshop，加基础财宝、胜利与诅咒牌。覆盖该完整推荐设置，**不是全部 26 种第二版王国牌**；不含扩展。 |
| [The Quest for El Dorado](games/el_dorado.md) / `el_dorado` | `2017_first_route_duo_no_caves` | 2017 原版官方首局路线 B/C/N/I/K、18 类市场牌，每人控制两个探险者，双方各八张起始牌。无洞穴、其他路线或扩展；若干购牌/动作时序的保守解释已记录。 |
| [Toy Battle](games/toy_battle.md) / `toy_battle` | `castle_field_intro_duo` | 官方首次游戏建议：Castle Field 地图、八种兵种及全部兵种能力，**特殊基地当普通基地处理**。保留完整印刷地图、连通、叠放和区域勋章。不是开启特殊基地效果的标准 Castle Field，也未覆盖全部八张战场。 |

## 使用前必须理解的规则边界

**Five Tribes 保持研究态。** 规则版本 2 已采用设计师确认的紧急出售：三个零价竞价格已满且当前无钱支付时，可以出售货物来筹措费用。但同时没有钱和可出售货物时，仍返回 `no_affordable_bid_and_zero_slots_full`。Anun-Nak 在支付所用长老回袋后仍不足三个工人时，返回 `anun_nak_bag_fewer_than_three_requires_ruling`，在扣款与随机抽取前停止。后者已有从正常开局合法到达的回归轨迹，并非只存在于人为构造状态。这两类结果不是平局，也不能按当前分数指定赢家；其余精灵的若干解释亦列于 [Five Tribes 边界表](games/five_tribes.md#interpretations-and-remaining-boundaries)。

**Jaipur 的完整三重平手没有补写家规。** 官方回合比较依次为卢比、奖励标记数量、货物标记数量；三者均相同后，当前实现保留 `unresolved_round_tie`，不颁发印章、不结束为自然终局。参见 [Jaipur 未决情况](games/jaipur.md#explicit-unresolved-rule-edge)。

**Splendor 和 Gizmos 的规则版本 2 明确采用有出处的无动作裁定。** 只有所有正常主动作
都无法执行时才允许强制跳过。Splendor 的出版方裁定经公开转引核查，要求展示保留牌；
Gizmos 依据授权 BGA 实现的专门修复提交，不称为出版方文字 FAQ。双方同时永久阻塞仍
没有获核实的终局判法，运行器只会外部截断，不制造平局。两款标记 `ready_with_rulings`，
具体证据等级见各游戏文档；旧模型的显式复用流程见 [规则修正与迁移](Rules_Corrections.md)。

两者的未决状态均没有合法后续动作、没有赢家且 `terminated=false`；具体 outcome 字段分别为 Five Tribes 的 `unresolved_rule` 和 Jaipur 的 `unresolved`。统一运行器将它们记为 `unresolved_rule`。评测必须保留发生次数及原始记录，不应删除这些种子后只报成功对局的胜率。

其他已披露解释包括：

- [Machi Koro](games/machi_koro.md#evidence-and-one-explicit-interpretation) 的新游乐园不追溯触发双骰是事件时序解释，尚非出版方 FAQ 定论，可能影响普通对局。后续若改变，须升级规则版本。
- [Targi](games/targi.md#boundary-rulings) 采用 Fata Morgana 只移至面朝上空位、部落牌耗尽后洗弃牌、保留牌额外打出例外和本轮货物奖励生效等解释，并区分作者确认与推定部分。
- [El Dorado](games/el_dorado.md#explicit-boundary-interpretations) 明确说明起始格重入、Transmitter 与普通购买的关系、同回合重新抽到新购牌时不能使用、购买阶段紫牌与障碍移除等处理；其中少见交互未获得独立设计师 FAQ 确认。
- [Patchwork](games/patchwork.md#sources-and-audit) 的修正皮革轨道来自出版方；已满板皮革处理的证据则来自数字版和转述澄清，不能把两者的证据等级混为一谈。
- [Toy Battle](games/toy_battle.md#why-this-configuration-is-official) 使用官方明确允许的入门配置。关闭特殊基地效果是固定覆盖范围的一部分；未验证的标准基地连锁时序没有被悄悄简化后声称已实现。

各游戏文档与组件数据还记录了来源、版本、核对方法和适用范围。公开资料可用于恢复组件事实，并不表示其插图、扫描件或第三方引擎代码获准再发布；本轮代码与数据交付不依赖私有 API 调用说明。

## 接口、信息权限与恢复

入口为 [`boardbench.families.registry`](../../src/boardbench/families/registry.py)。`GAME_IDS` 定义请求的游戏族；`available_games()` 只表明模块已安装，**不认证规则完整性或 bot 训练完成情况**。用 `game_class(game_id)` 获取实现后，按以下流程运行：

`describe()` 和命令行 `list` 另外提供机器可读的 `coverage_status` 与
`known_unresolved_edges`。Five Tribes 标记为 `research_only`，Jaipur 为
`known_unresolved_edges`；Splendor/Gizmos 为 `ready_with_rulings`；其余 `fixed_scope`
表示本文界定的范围，也不是官方认证。

1. `reset(seed, first_player)` 初始化；随机组件与随机事件由环境 RNG 管理。
2. `current_player` / `current_actor()` 指定当前决策者；一个 API 步骤是实际玩家选择，可能是反应、支付、奖励选择或额外回合，因此不能假定每步机械轮换双方。
3. `observe(player)` 返回该玩家按规则可见的局面；`legal_actions(player)` 返回当前阶段的动作字典。环境合法动作与 bot 的选择偏好分开，启发式分数不是合法性条件。
4. `step(action)` 检查合法性并推进；非法动作不会消耗状态或 RNG。自然终局由游戏规则决定，运行器的时间/步数上限产生外部截断，不增加家规终局。
5. `snapshot()` / `serialize()` 与 `restore()` 保存、恢复内部状态和 RNG，支持 JSON 回放；恢复校验游戏、配置与规则版本。跨规则版本不能假定兼容。

观察、裁判数据和回放快照的用途不同。观察隐藏对手规则上不可见的手牌、私有抽取、未来牌序等；先前公开的获取或出牌历史仍可用于合法记忆，具体可见内容以各游戏文档为准。**完整快照、RNG、裁判的隐藏计分详情不应传给策略。** 记录轨迹时也应保留观察者席位，不能把两位玩家各自可见的信息合成一份策略输入。

通用 bot 使用 `features(observation, action)` 的 24 维有限数值；特征零提供初始启发式，其余特征用于可训练的修正。这是一组初始策略表示，不是完整局面编码，也不是官方计分的替代。已有 Harmonies 冻结 RL 对手通过专用策略类型接入，不能用同名环境下的通用启发式冒充。

要复现评测，除种子外还需固定源代码、组件数据、规则/配置身份、对手模型及推理设置。换位配对使用相同初始环境种子；不同动作可以使后续随机调用分叉，因此“同种子”不表示整局随机轨迹逐事件相同。冻结实际模型文件、来源归档、继续运行和独立确认流程见 [Evaluation](Evaluation.md)。

## 官方分数、胜负奖励与不完整结果

官方分数用于描述游戏成绩，胜负奖励用于训练或统计比赛结果，两者分别保存。

| 字段或情况 | 正确解释 |
|---|---|
| `scores` / `score_breakdown` | 游戏自身的计分及分项，未统一归一化。非终局计分的含义因游戏而异；例如 Stone Age 当时的公开分数轨不含未公开的最终文明牌计分。裁判分项可能含特权信息。 |
| Machi Koro、El Dorado | 竞速胜利，没有构造统一胜利点；官方 `scores` 为 `null`，看赢家及结束原因。 |
| Jaipur | 整场官方 `scores` 是印章数，卢比属于回合/分项信息。 |
| Toy Battle | 官方分数为勋章；攻占总部者可以勋章更少，因此不能简单以分数大小推断赢家。 |
| 正常非终局步骤 | 共享奖励协议为双方 0，不把中间启发式值伪装成官方奖励。 |
| 已完成且唯一赢家 | 终局训练 reward 为赢家 `+1`、输家 `-1`；按游戏规定共同获胜或平局时双方为 0。评测的胜率点数另计为胜 1、平 0.5、负 0。 |
| 未决、超时、外部截断、执行错误或未启动 | 没有可靠终局奖励；运行结果保留状态，不制造输赢，也不把中间分数代入终局成绩。 |

正式比较应同时报告计划场数、完成场数、各类不完整结果和完整配对种子数量。仅完成对局的胜率须带分母；全计划场次的已知胜率点数仅给出下界，未知结果仍是未知。确认集应使用冻结候选和新环境种子，不能再根据确认成绩挑选模型。具体区间、分母和恢复约束以 [Evaluation](Evaluation.md#outputs-and-interpretation) 为准。
