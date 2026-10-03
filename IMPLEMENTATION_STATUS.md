# V1.2 实现与验收状态

2026-10-04，包版本 0.4.0。V1.2 基础设施已实现，当前四小时试验的状态以
`outputs/v12/latest_status.json` 为准，最终以对应 run 的 `report.json` 为准。
尚不能把“代码可运行”称为“达到高手人类”或“训练已收敛”。

- 规则未变；Harmonies 新增 4564/884/240 动作版本，同口径 STOP 后为 4565/885/241。
  槽位映射、合法 mask、32 张卡的旋转后继等价，以及隐藏未来独立性有回归覆盖。
- 增加完整 Adam/RNG/数据位置恢复，PPO 行为 log-prob 比率、真实梯度更新与恢复等价检查。
- 限时协议由父进程结算、子进程执行 hooks/策略，覆盖永久阻塞、部分 IPC 帧、错误动作、
  主动停止、截止部分得分、启动失败和原子步越界尾部；所有尝试进入统计。
- 单一四小时 watchdog 覆盖诊断/教师/训练/验证/测试/发布；固定八个物理核与一张 GPU。
  源码快照隔离后续编辑。三种子、开发/验证/最终测试分离，发布前真实恢复并原子换指针。
- 网页支持人工计时、STOP、过期存档结算及新局刷新策略；网页比较不冒充正式限时评测。
- Chromium 的新增 STOP/截止/恢复/390px 检查与原 32 步动物交互检查均通过；
  证据在 `outputs/v12_checks/browser_v12_01/` 与 `outputs/v12_checks/browser_animals_01/`。
- 全量 **161 项通过**，见 `runs/v12_acceptance_final.xml`；首轮 157 通过、1 个源码摘要检查触发，因为
  测试期间又更新了运行时代码。保留该失败记录，停止修改后重新验收，不关闭版本检查。
  另修正了一个测试夹具把正常子进程启动限制为 0.3 秒导致的忙机偶发失败；正式启动
  30 秒限额未放宽，永久阻塞启动的短超时测试仍保留。

详见 [运行手册](docs/V1_2_RUNBOOK.md)。人类参考检索未找到可核实的同 setting 分布，
故 human_target 留空；已记录不同棋盘面/计算上界等不匹配来源及不采纳理由。

---

# V1.1 历史实现与验收状态

日期：2026-10-03；包版本 **0.3.0**。V1 review 中五项工作已落实；没有新增游戏或自主
agent 框架，没有更改动物牌、末袋裁定或其他游戏规则。下文保留 V1/V0 历史结果。

## 本轮修改及验证

- 候选从各自配置及权重 metadata 构建网络；hidden=32/64 的混合 run 通过训练、验证、
  导出和新进程推理。候选评测前检查全部来源的任务、规则及训练集对主验证/测试集的污染。
  来源摘要构成稳定 ID，同名阶段不再静默丢失。4 项缺陷回归先失败、修复后通过。
- Micro 先做固定 8 个训练局的 flat MLP 诊断：8,192 局训练约 81.80 秒，训练集 argmax
  从 5.125 到 10.375，验证从 5.828 到 9.031；采样得分也改善。说明存在学习能力，不能
  将 V1 弱结果简单归因为断梯度或 argmax 错误。保留 [诊断结果](outputs/v11/micro_diagnostic_a/report.json)。
- Micro/TIE 引入共享动作评分网络，使用公开邻接/线关系计数；另加明确标为监督学习的
  启发式模仿对照。Harmonies PPO 仍是原 flat MLP；不把人工表示收益全部算作 RL 收益。
- 搜索显式区分原子动作与完整回合深度，带硬模拟步预算；Harmonies 增加动物模式潜力
  叶估值，记录实际 rollout 深度与分项结果。预算不足时不保证到达目标完整回合。
- **132 项全量测试通过**：[JUnit](runs/v11_acceptance_final.xml)。覆盖候选污染、混合宽度、
  来源 ID、训练/推理 metadata、有限训练集、真实参数学习、搜索预算、并行验证与分片评测。
- **39 个固定策略 × 128 局 = 4,992 局**全部 completed，0 非法动作、0 失败、0 截断。
  [汇总](outputs/v11/report.json) 重新核算原始 Runner 分片，而非复制控制台均值。
- [动物定向 Chromium 验证](runs/v11_browser_animals_a/report.json)：32 步实际网页操作，
  在资源放置中间拿牌，放置 5 只动物完成卡片，释放槽位并再拿牌，两个中途状态精确存取。
  初始供给是库存守恒的受控 fixture，每一步均经过正常引擎合法检查。
- [三游戏完整浏览器验证](runs/v11_browser_trained_final/browser_check.json)：手动完整局、
  提示/拒绝、存取、真实 PPO 接管/暂停、挑战回放、390px 手机无溢出，无 JavaScript 错误。
  首轮发现长策略标签撑宽下拉框，已精简本轮展示标签；完整训练信息仍在 manifest。
  失败截图保留于 runs/v11_browser_trained/，最终通过记录另存，不覆盖失败证据。
- [产物审计](runs/v11_artifact_audit.json)：15 个训练 run、39 个导出策略内容摘要全部通过；
  各三种子组的实际学习模块相同，训练 actor 参数实际更新，原 V1 权重和 ZIP 均未改变。
- [迁移恢复](runs/v11_portable/report.json)：全部 15 个已训练策略，在新 venv、新目录、
  隔离 Python 中重放首个测试局，动作与完整观测一致，父 checkpoint 和副本均未被改写。
  复用现有 Torch/NumPy，不声称完全空系统依赖复现。
- [wheel 验证](runs/v11_wheel_validation.json)：0.3.0 wheel 在标准库独立 venv 中跑通
  三游戏及卡牌/网页资源，无 Torch 也可用引擎和非神经策略。

## 新冻结测试结果

训练种子 811/812/813；每个 run 独立用 64 个验证局选 checkpoint，保留全部三个训练重复。
搜索只用验证选配置。新测试为 6000000..6000127，全部选择冻结后才查看测试分数；没有
根据新测试成绩追加训练或调参。原 V1 测试集不用于本轮选模型。

表中学习方法为 **三个模型的测试均分 ± 三个均分的样本标准差**，不是环境分数标准差，
也不是 95% 区间。非学习方法只列同一批 128 局的均分。

| 游戏 | 随机 | 启发式 | 搜索 | PPO | 模仿学习 | 同表示未训练 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Micro Tiles | 5.96 | 14.64 | 15.04 | 14.83 ± 0.13 | 14.84 ± 0.06 | 11.83 ± 2.25 |
| Take It Easy! | 12.16 | 135.13 | 147.34 | 144.24 ± 3.52 | 136.90 ± 0.81 | 41.60 ± 24.40 |
| Harmonies | 22.63 | 46.15 | 85.20 | 66.73 ± 5.87 | 未运行 | 27.42 ± 1.09 |

Micro PPO 已达到启发式附近，但不宣称显著超过启发式；三个模型对启发式的配对区间均
包含 0。未训练网络在新表示下本来就较强，seed 812 的训练收益区间还略含 0；另外两个
种子的匹配训练收益为正。TIE 三模型对启发式的配对均值差为 +6.77/+13.16/+7.38，区间
下界均为正，说明这批固定产物有实质收益，但三个训练重复仍不足以宣称普遍稳定或 SOTA。

配对区间按 128 个环境种子上的差值均值 ± 1.96 × 样本标准差 / sqrt(128)，只描述固定
模型的环境不确定性，未经多重比较校正。每模型原始差及区间见 [report.json](outputs/v11/report.json)。
不能把 384 局当成 384 个独立训练重复，也不能将 V1/V1.1 不同测试集当作配对实验。

Harmonies 的 PPO 优势仍以地形为主；增强搜索的动物规划更强：

| 方法 | 地形分 | 动物分 | 动物放置数/局 | 完成卡/局 | 总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 搜索 | 55.98 | 29.23 | 6.93 | 1.14 | 85.20 |
| PPO（三模型均值） | 59.57 | 7.16 | 1.88 | 0.15 | 66.73 |

## 成本与停止点

训练用 b35 RTX 4090，每个 run 单卡，独立任务并行；单 run 上限 900 秒（batch 边界检查），
所有任务均达到预定局数。表中“选中耗时”是首次生成被验证选中 checkpoint 的累计时间，
不表示已证明它是最优策略；“完整训练”包含最终未被选中的较晚阶段。

| 游戏 | PPO 局数/run | PPO 完整训练秒（811/812/813） | 选中耗时秒（811/812/813） | 模仿完整训练秒/run |
| --- | ---: | --- | --- | --- |
| Micro | 16,384 | 205.07 / 199.08 / 196.11 | 205.06 / 99.86 / 49.35 | 4.11–4.18 |
| TIE | 16,384 | 447.09 / 440.40 / 441.98 | 224.40 / 330.63 / 330.24 | 11.17–11.28 |
| Harmonies | 8,192 | 812.82 / 791.38 / 828.34 | 812.80 / 791.36 / 828.32 | 未运行 |

九次 PPO 的 run 时间相加为 4,362.28 秒，六次模仿为 46.09 秒；固定集诊断另计 81.80 秒。
这些不是并行运行的总墙钟时间，也不含验证/搜索测试/部署。未擅自停止任何既有服务；
任务结束后 b35 的 GPU 1–5 已释放，原 GPU 0 服务保留。

CPU 推理在 b35、每 worker 2 线程；Micro/TIE 4 workers，Harmonies 8 workers。
纯 Solver 调用累计均值如下，不含加载、Runner 日志和浏览器网络：

| 游戏 | 搜索秒/局 | PPO 秒/局（三模型均值） | 搜索实际模拟步/局 |
| --- | ---: | ---: | ---: |
| Micro | 0.0391 | 0.00535 | 376 |
| TIE | 0.3818 | 0.01062 | 1,944 |
| Harmonies | 13.5994 | 0.04426 | 10,920.60 |

Harmonies 搜索单局 Runner wall time 为 13.7358 秒；三个 PPO 为 0.1697–0.1918 秒，
加载另约 4.16ms/局。高搜索分数有明确代价，不能声称同成本提升或跨版本硬件归一化加速。
选中搜索分别为 turn_384、atomic_128、turn_384；名字中的预算是每次决策的模拟步上限，
不是保证用满的步数。V1 24 步和其他候选的验证记录也完整保留。

## 交付与边界

- 本轮评测及部署源码摘要：
  `a541607f5ad121b35c12a38d4c04f23ba691dff171c0c1e25a9ccbae0a4125bb`。
  各训练启动时的完整源码单独保存；训练期间评测/打包工作使完整摘要不同，实际学习模块
  组内一致性已审计，不将后来的源码冒充训练时源码。
- outputs/v11/ 包含计划、15 次训练全部权重、验证、39 个策略及 4,992 局原始分片。
  Git 发布代码与报告文档，ZIP 另含权重、原始数据、截图和逐文件 SHA-256。
- 当前仍无优化器级续训；可用新配置、新输出目录做下一轮预先约定的实验。搜索每步重新
  规划，不会自动跨局积累策略；继续训练也不保证单调改善。
- 任意自定义超长策略标签未做通用布局重构；本轮交付的紧凑标签已通过手机检查。
  手机点击堆叠详情仍未增加，按 review 属非阻塞体验项。
- 无本轮独立外部 reviewer；review 文档提到的 review_v1_checks/ 不在当前工作区，
  没有声称重新执行那套独立核查。游戏牌面数据沿用现有规范及来源边界。
- 旧 V1 源码绑定 checkpoint 必须用自己的源码副本恢复，未关闭任何哈希检查。

---

# V1 实现与验收状态（历史）

日期：2026-10-03；包版本 **0.2.0**。以下 V1 结果是本轮实际运行，不是交接包的计划。
V0.1 的历史记录保留在本文后半部分；其 ZIP 未被覆盖，摘要与此前交付一致。

| 游戏 | 环境完成 | 实验完成 | 展示完成 |
| --- | --- | --- | --- |
| Micro Tiles | 是 | 是：单训练种子 pilot | 是：真实权重与四类交互 |
| Take It Easy! | 是 | 是：单训练种子 pilot | 是：真实权重与四类交互 |
| Harmonies solo A | 是 | 是：单训练种子 pilot | 是：真实权重与四类交互 |

## V1 实际验证

- **107 项测试通过**，包含原 V0 的 40 项。记录：
  [最终 JUnit](runs/v1_acceptance_tests_final.xml)。
- 规则测试覆盖独立 Micro 30/15/0、Take It Easy! 93 分样例，Harmonies 全部叠放、
  32 动物 × 6 方向、库存守恒、河流直径/环/分叉、取牌交错/延迟揭示、末袋见证和半局恢复。
  反射拒绝使用明确标注的合成手性模板测试；基础 32 牌本身未发现需要反射才能匹配的模板。
- 三个环境均验证非法动作不改变真实状态/RNG、搜索分支不读取真实未来、公开对象不污染规则、
  RL 固定有限数值/mask、终局不再抽取或采样。
- PPO 确实更新参数；保存未训练和两个训练进度。训练、验证、测试种子不重叠。
  参数哈希、环境步数、梯度步数、配置和训练时源码均保留。
- **1,920 局冻结测试全部完成**：3 个游戏 × 5 个固定策略 × 128 个环境种子，
  0 非法动作、0 失败、0 截断。全部经同一个 Runner 的独立 checkpoint 评测路径。
- [真实浏览器验证](runs/v1_browser_trained/browser_check.json)：每游戏手动点击完整一局、
  提示/拒绝、存档恢复、策略接管/暂停、终局比较与双方回放、手机宽度无横向溢出；
  三个选中的 RL checkpoint 均在网页实际加载，无 JavaScript 错误。
- [迁移恢复验证](runs/v1_portable/report.json)：复制最终 checkpoint 和其源码，在临时 venv
  通过隔离 Python、新工作目录恢复三款 RL 策略，首个测试局的动作和完整观测逐项一致，
  父 checkpoint 和副本哈希均不变。边界：复用安装好的 Torch/NumPy，不声称完全空系统安装。

## 冻结测试得分

每格为最终真实分数均值 ± 总体标准差，128 局；所有方法使用相同测试种子。
候选选择仅用 32 局验证，测试开始后没有继续调参或训练。

| 游戏 | 随机 | 启发式 | 搜索 | PPO 已训练 | 同网络未训练 |
| --- | --- | --- | --- | --- | --- |
| Micro Tiles | 5.70 ± 3.50 | 14.84 ± 3.66 | 15.05 ± 3.71 | 5.84 ± 3.60 | 5.77 ± 3.58 |
| Take It Easy! | 7.61 ± 12.16 | 135.84 ± 25.06 | 139.45 ± 28.17 | 10.55 ± 14.62 | 7.72 ± 11.88 |
| Harmonies | 22.93 ± 8.21 | 44.30 ± 9.86 | 62.58 ± 12.81 | 55.23 ± 11.15 | 21.02 ± 7.49 |

解释：Harmonies 的训练后策略在本次固定实验中明显高于未训练/随机及启发式，但低于搜索；
Micro 尚未验证学习收益；Take It Easy! 的 RL 仍远弱于启发式与搜索。
这不是多训练种子的稳定性结论，不宣称 SOTA。页面显示方法和验证均分，没有硬凑难度三级。

完整逐局记录与汇总：
[Micro](outputs/v1/micro_tiles/test_pilot_b/results.json)、
[Take It Easy!](outputs/v1/take_it_easy/test_pilot_b/results.json)、
[Harmonies](outputs/v1/harmonies/test_pilot_b/results.json)。各方法子目录有 episodes.jsonl、
events.jsonl、summary.json；决策、资源和分数可追溯。

## 成本与训练投入

线上推理测于同一 CPU 主机、2 线程，以下为纯 Solver 调用 p95 毫秒；不含 checkpoint
加载、Runner 日志或网页网络。加载耗时、每局决策总时长、实际搜索步数另存结果 JSON。

| 游戏 | 随机 | 启发式 | 搜索 | PPO 已训练 |
| --- | --- | --- | --- | --- |
| Micro Tiles | 0.017 | 0.193 | 5.764 | 0.446 |
| Take It Easy! | 0.014 | 0.325 | 9.034 | 0.330 |
| Harmonies | 0.011 | 1.755 | 35.676 | 0.587 |

验证为三款游戏均选中 width=4、rollouts=2、depth=3、最多 24 次模拟步/决策。
实际每局平均模拟步数分别为 188、412、1167.125。搜索远非无限预算。

训练实际使用 RTX 4090 + PyTorch 2.4.1+cu121，三个独立任务分别用单卡并行；
每游戏只有一个训练随机种子 739。Pilot A 后仅做一次 4×预算扩大，所有旧产物保留。
下表记录被验证选中的 checkpoint，不把未选中训练进度冒充选中模型的成本。

| 游戏 | 选中训练局数 | 环境步数 | 梯度步数 | 达到该点秒数 | 参数量 |
| --- | --- | --- | --- | --- | --- |
| Micro Tiles | 2,048 | 18,432 | 512 | 22.40 | 24,979 |
| Take It Easy! | 8,192 | 155,648 | 3,072 | 167.93 | 33,172 |
| Harmonies | 2,048 | 105,858 | 1,756 | 179.52 | 685,653 |

全部实际训练投入（包含未选中的较晚权重及重复起跑）：Micro A 11.10s + B 42.31s；
Take It Easy! A 42.46s + B 167.93s；Harmonies A 40.48s + B 179.52s。
验证、测试、加载和网页时间没有被混入训练秒数；三任务并行时间不可直接相加作总墙钟时间。

## 交付与已知边界

- 当前运行源码摘要：
  `9e84a54f9f49cfc7aedb21997cdc2a4db4b7889ed1121a9009a0daa38864eaf4`。
  训练源码按启动时独立保存；之后 UI/导出/元数据修补没有被伪装成训练时源码。
- 15 个策略 manifest 和源码绑定 checkpoint 在 outputs/v1/*/policies/，包括真实模型文件。
  静态网页构建在 build/ui/；安装包包含 JSON 卡牌及两个 HTML 页面。
  [产物审计](runs/v1_artifact_audit.json) 校验全部 15 个 checkpoint，确认 A/B 同初始权重及同训练代码。
- 已构建 boardbench-0.2.0 wheel，并在无第三方运行依赖的新 venv 完成三款游戏，
  检查卡牌 JSON 和网页资源可读取。见 [wheel 验证](runs/v1_wheel_validation.json)。
- 首个 V1 ZIP 在独立解压恢复中发现空 workspace 目录遗漏；已补显式空目录条目及
  真实 ZIP 解压恢复回归测试。失败候选保留在 deliverables/rejected/，不作为交付包。
- 游戏存档与系统 checkpoint 分开。PPO 支持推理恢复，当前训练 CLI 不提供优化器级断点续训。
- 标准库基础环境可以跑引擎/随机/启发式/搜索；训练与 RL 推理需要可选数值依赖。
- 因本机旧 setuptools 不满足 pyproject 构建约束，wheel 必须使用标准隔离构建；
  没有把失败构建产生的 UNKNOWN wheel 当成交付物。
- 旧源码 checkpoint 仍应使用自己的源码副本恢复；没有关闭源码或内容哈希验证。
- 无独立审稿人/外部规则引擎作为 oracle；已运行独立样例、单元/集成/浏览器/迁移验证。
- 原版动物美术未被复制；32 卡结构化数据保留第三方来源与核查边界。

---

# V0 实现与验收状态（历史）

验收日期：2026-10-03。规范版本：0.3。实现版本：boardbench 0.1.1（V0.1 修补版）。

**A01–A12 已通过。** 全部核心路径使用 CPU 和确定性 mock，无 API key。
实际执行 `python -m pytest -q`：**40 passed**。完整命令和输出见
[commands.json](runs/acceptance_v0_1/commands.json)、[pytest.log](runs/acceptance_v0_1/pytest.log)、
[pytest.xml](runs/acceptance_v0_1/pytest.xml)。源码标识和环境见
[environment.json](runs/acceptance_v0_1/environment.json)。

新版审阅包已完成全新环境恢复：解压后创建独立 venv，仅安装 checkpoint 源码副本，
通过 `python -I` 从不同目录评测，动作与分数和新验收记录一致。见
[portable_restore/verification.json](runs/acceptance_v0_1/portable_restore/verification.json)
及同目录命令和输出；临时验证目录用后清理，父 checkpoint 保留。

## 本轮 review 处理

R1–R3 均采纳，详见 [逐条回复](docs/BoardBench_V0_Review_Response.md)。
`tests/test_review_regressions.py` 在修改前 6 failed、修改后 6 passed；完整套件 40 passed。
回归覆盖带写缓存及后续磁盘引用的 load（评测与恢复适应）、reset 失败尝试记录（两种模式）、
有限数值 RL 编码及 Integral 动作。新增行为并入 A04、A09、A10 的验收。
证据：[修复前](runs/acceptance_v0_1/review_regressions_before.log)、
[修复后](runs/acceptance_v0_1/review_regressions_after.log)、
[NumPy 实测](runs/acceptance_v0_1/numpy_smoke.log)。

本轮实际重新执行安装、40 项测试、三个示例、两次评测和 Chromium 操作；全部通过。
原 ZIP SHA-256 未变，旧验收目录中的 241 个文件与原 ZIP 逐字节一致，见
[旧产物核对](runs/acceptance_v0_1/prior_artifact_integrity.log)。新版产物使用新目录，无覆盖旧 checkpoint。

## 逐项验收

以下测试均包含在上述完整测试命令中；测试名可用 `pytest -k NAME` 单独重跑。

| 验收 ID | 状态 | 实际命令或检查方式 | 证据路径 | 已知问题 |
| --- | --- | --- | --- | --- |
| A01 | 已通过 | `tests/test_rules.py`：受控随机源依次抽尽收益牌；BANK／失败终止；reward 总和等于终局得分 | `tests/test_rules.py`；`runs/acceptance_v0_1/pytest.xml` | 无 |
| A02 | 已通过 | `test_rejection_preserves_state_and_rng`；`test_invalid_actions_count_and_do_not_fake_transitions` | `tests/test_rules.py`；`tests/test_runner.py` | 非法动作使用显式异常协议 |
| A03 | 已通过 | `test_seeds_and_simulation_do_not_touch_real_rng`：20 个种子，每个真实决策前 100 个模拟分支；真实完整轨迹对照 | `tests/test_rules.py`；`runs/acceptance_v0_1/search_demo/` | 复现验收不触及墙钟预算 |
| A04 | 已通过 | 引擎／RL／交互一致性；实际 RL rollout；有限数值编码、有效性标志与 Integral 动作 | `tests/test_rules.py`；`tests/test_ui.py`；`tests/test_review_regressions.py`；`runs/acceptance_v0_1/rl_example.log` | RL 为 Gym 风格薄接口，无算法库依赖 |
| A05 | 已通过 | `test_resident_lifecycle_and_terminal_feedback`；协作日志检查 history_size=0…5 且局内步数每局归零 | `tests/test_runner.py`；`runs/acceptance_v0_1/collaboration_demo/events.jsonl` | 无 |
| A06 | 已通过 | `test_collaboration_updates_consumed_and_budget_exhaustion_continues`：固定输入预测变化，后续决策读取新版本；真实子进程拟合 | `runs/acceptance_v0_1/collaboration_demo/jobs/`；同 run 的 `events.jsonl` | 不以得分提升作为验收条件 |
| A07 | 已通过 | 两个本地示例零调用；协作一次成功、五次拒绝后继续；失败 client 仍占额度 | `tests/test_runner.py`；`runs/acceptance_v0_1/audit.json` | 真实 API 未实现／未验证，mock 延迟不代表 LLM |
| A08 | 已通过 | `test_jobs_success_failure_timeout_and_no_new_jobs_after_deadline`；真实更新 job 产物被加载 | `tests/test_runner.py`；`runs/acceptance_v0_1/collaboration_demo/jobs/` | 只支持同步本地 argv 与 timeout，不做资源分配 |
| A09 | 已通过 | 新进程与新目录恢复、实际源码不一致报错、RNG 保存恢复；私有 load 目录及后续磁盘引用 | `tests/test_checkpoint.py`；`tests/test_review_regressions.py`；`runs/acceptance_v0_1/collaboration_demo/checkpoints/ep_000006/` | 源码快照需复制到独立目录再 editable install；不恢复半局 |
| A10 | 已通过 | 每局新副本、重复评测；带写缓存的 load 不串局；原产物摘要不变且可再校验；临时副本删除 | `tests/test_checkpoint.py`；`tests/test_review_regressions.py`；`runs/acceptance_v0_1/eval_demo/`；`eval_repeat/`；`audit.json` | V0 评测只支持禁止学习 |
| A11 | 已通过 | 同一 Runner 跑三个 Solver；日志计数、分数、决策耗时核对；另用不同观测／动作结构测试任务独立性 | `tests/test_runner.py`；`runs/acceptance_v0_1/audit.json`；三个 `*_demo/` | 示例策略和风险采集适配器本身仍是任务专属 |
| A12 | 已通过 | 实际 Chromium 页面操作 DRAW、BANK、新局、回放前后步／换局；检查无 JS 错误及窄屏溢出；重新安装和完整测试 | `runs/acceptance_v0_1/install.log`；`ui/browser_check.json`；`ui/play.png`；`ui/replay.png`；`ui/play_mobile.png` | Playwright 1.51.0 为本机兼容的可选浏览器测试工具 |

## 实际示例结果

这些是功能验收记录，不是策略强弱或学习有效性的研究结论。

| 运行 | 自然完成局数 | 得分总和 | 真实动作 | 模拟步数 | 发出的模型调用 | 额度拒绝 | 更新 job |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 6/6 | 1 | 7 | 0 | 0 | 0 | 0 |
| reference_search | 6/6 | 18 | 17 | 886 | 0 | 0 | 0 |
| collaboration_demo | 6/6 | 16 | 16 | 0 | 1 mock | 5 | 3 |
| checkpoint 独立评测 | 4/4 | 0 | 5 | 0 | 0 | 4 | 0 |

评测四局都抽到失败牌而自然终止，分数为真实的零，不是预算截断或缺失值填零。
原样重复评测的动作和分数一致，原适应目录和 checkpoint 摘要不变。
协作模块均值得分从 0.5 更新为 3.5、3.0、2.6666666666666665。
参数变更及其后决策读取新版本均有事件证据。

在线延迟和此前适应投入分别记录在 `eval_demo/summary.json` 的 `resources` 与
`historical_adaptation`。mock 调用耗时只说明包装路径执行，不能用于报告真实模型延迟。
全部计划保存点为 ep_000002、ep_000004、ep_000006，三个适应示例均保留全部保存点。

## 实现复查与边界

已本地复查规则、Runner、预算、checkpoint 和评测代码；没有使用独立代理审查。
复查修正了决策耗时口径、评测初始化尝试与已开始局数区分、收尾异常后的保存、
评测副本清理、恢复适应后的历史资源累计，以及源码安装对 checkpoint 不变性的影响。
相关边界有直接行为测试；本次没有遗留的核心验收失败项。

- 实现了确定性 mock，未接真实 provider；这是明确保留的未验证项。
- Python 钩子为协作式超时，job 支持进程组超时终止；不承诺硬实时调度或恶意代码隔离。
- 环境／求解器通过小型注册字典扩展，不提供插件发现、分布式执行或动态版本加载。
- 核心代码和参数格式无 pickle；外部源码依赖、虚拟环境和浏览器二进制不放进 checkpoint。
- 安装、功能测试、浏览器检查均有实际输出；UI 桌面截图已查看检查，移动布局有浏览器宽度检查。

完整接口与偏差说明见 [docs/interfaces.md](docs/interfaces.md)、[docs/decisions.md](docs/decisions.md)。
