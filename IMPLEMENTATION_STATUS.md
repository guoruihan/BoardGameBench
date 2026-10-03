# V1 实现与验收状态

日期：2026-10-03；包版本 **0.2.0**。以下 V1 结果是本轮实际运行，不是交接包的计划。
V0.1 的历史记录保留在本文后半部分；其 ZIP 未被覆盖，摘要与此前交付一致。

| 游戏 | 环境完成 | 实验完成 | 展示完成 |
| --- | --- | --- | --- |
| Micro Tiles | 是 | 是：单训练种子 pilot | 是：真实权重与四类交互 |
| Take It Easy! | 是 | 是：单训练种子 pilot | 是：真实权重与四类交互 |
| Harmonies solo A | 是 | 是：单训练种子 pilot | 是：真实权重与四类交互 |

## V1 实际验证

- **105 项测试通过**，包含原 V0 的 40 项。记录：
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
