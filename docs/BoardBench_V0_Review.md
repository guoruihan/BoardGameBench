# BoardBench V0 审查与修改清单

审查对象：`boardbench-v0-review.zip`，实现版本 `boardbench 0.1.0`。  
验收依据：随包 `docs/V0_Implementation_Spec.md`，规范版本 0.3。  
审查方式：源码检查、独立重跑、上传产物核对、针对性缺陷复现。  
本次没有修改提交的业务源码，也没有将缺陷复现写入上传的原 checkpoint。

## 1 结论

**V0 的主流程和架构基本通过审查。建议完成一个小型 V0.1 修补后冻结基础框架，再进入 V1。**

当前实现确实具备：单一规则引擎、常驻 Solver、跨局模块更新、统一模型调用包装、额度耗尽后的本地路径、同步 job、完整 checkpoint、新进程恢复及独立评测。没有必要为 V0 重构成服务平台或提前训练强 RL。

需要优先修补的是：恢复过程虽然复制了 workspace，却把原 checkpoint 的 `solver/` 路径直接交给 `load()`。现有三个示例只读加载，因此全部既有测试通过；带加载缓存或磁盘模型的后续 Solver 会暴露共享产物问题。这个问题已用普通加载缓存复现，修复只需要完善副本路径，不需要权限系统。

另有两项非 V0 阻断问题：环境 reset 失败的局级记录不完整，以及 RL 观测向量含 `None`。建议分别在接入新环境和正式 RL 训练前解决。

## 2 独立验证结果

### 2.1 本次实际执行

- 在独立 Python 3.12.14 环境中安装项目及测试依赖。
- 完整执行测试：**34 passed in 1.94s**。
- 使用独立 Python CLI 进程重跑 random、reference_search、collaboration_demo。
- 校验上传包内全部 **9 个 checkpoint** 的文件摘要与源码版本。
- 从上传的协作 checkpoint 直接恢复评测两次，使用不同于项目的工作目录。
- 核对新运行与上传运行的完整动作／环境转移记录一致。
- 核对真实步数、模型调用、拒绝次数、job 次数与事件账本一致。
- 对上传的原运行目录做前后摘要比较，内置 Solver 的标准评测未修改原产物。
- 查看上传的 play 和 replay 截图，检查 UI／回放代码及随包浏览器记录。

首次测试在受限执行环境中得到 33 通过、1 个本地 HTTP 建连权限错误；允许 loopback 网络后完整重跑 34 项全部通过。这是审查环境限制，不是项目缺陷。

本次没有重新运行 Chromium 页面点击流程；浏览器层依据随包操作记录、截图和源码审查。HTTP 交互与回放后端已经随测试重跑。真实 LLM provider 尚未实现、未验证，与当前 V0 的可选接入范围一致。

### 2.2 重跑与提交记录一致

| 运行 | 自然完成局数 | 总得分 | 真实步数 | 模拟步数 | 发出的模型调用 | 模型拒绝 | job |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 6/6 | 1 | 7 | 0 | 0 | 0 | 0 |
| reference_search | 6/6 | 18 | 17 | 886 | 0 | 0 | 0 |
| collaboration_demo | 6/6 | 16 | 16 | 0 | 1 mock | 5 | 3 |
| 上传 checkpoint 的独立评测 | 4/4 | 0 | 5 | 0 | 0 | 4 | 0 |

以上是功能复现结果，不是策略强弱或参数学习有效性的比较。评测的 0 分来自四局自然失败，不能据此判断学习系统是否有效。强策略、充分种子和训练曲线属于 V1。

协作示例的更新确实进入决策：在相同 `pot=2` 观测、关闭可用模型调用的条件下，初始模块选择 BANK，加载第 2 局 checkpoint 后选择 DRAW。它证明了“更新—加载—使用”的链路，而不是只创建了一个参数文件。

## 3 R1：恢复时应复制 Solver 产物

**优先级：建议 V0.1 修复，进入可扩展学习系统前完成。**

定位：`src/boardbench/runner/engine.py` 第 150 行的 `initialize_solver`；关键调用在第 155 行。

当前代码：

```python
restore_workspace(checkpoint, context.workspace)
invoke(context, "load", solver.load, Path(checkpoint) / "solver")
```

虽然 evaluation 为每局新建临时 workspace 和 Solver 实例，但 `load()` 接收的始终是原保存点内同一个 `solver/` 目录。恢复适应也使用这条路径。

### 实际复现

在上传 checkpoint 的复制件上，对 `CollaborationDemo.load` 增加普通加载缓存行为：先执行原有加载，再在传入目录写入 `compiled_cache.bin`。评测两个种子后观察到：

```json
{
  "evaluation_status": "completed",
  "cache_present_before_first_load": false,
  "cache_present_before_second_load": true,
  "original_checkpoint_changed": true,
  "later_validation_error": "checkpoint contents do not match manifest hashes"
}
```

这说明副本并未覆盖全部运行产物：后续评测局可以看到上一局加载留下的文件，原 checkpoint 的完整性也被破坏。复现不涉及恶意代码；加载时编译缓存、转换模型格式或保留磁盘参数路径都可能遇到这类问题。

当前内置 Solver 的 `load()` 只读，因此其已提交成绩和独立评测记录没有被这个问题污染。

### 最小修改

为每次恢复建立私有目录，同时复制 `solver/` 与 `workspace/`；将私有目录的 `solver/` 传给 `load()`。评测每局重新复制，恢复适应则为新 run 创建一次副本。副本应在相关 Solver 使用期间保持存在，不能在 `load()` 返回后立即删除仍可能被引用的文件。

源码快照仍按现有版本校验及安装契约处理，不必每局复制或重新安装源码。无需容器、权限系统或动态源码加载。

新增一项有实际意义的测试：加载时创建缓存，第一局和第二局加载前均看不到对方缓存；原 checkpoint 摘要不变，评测后仍能通过 `validate_checkpoint()`。同时覆盖一次恢复适应，保证新 run 不改动父 checkpoint。

## 4 R2：reset 失败应记录一次失败尝试

**优先级：低；接入 V1 新环境前补齐即可。**

定位：`src/boardbench/runner/engine.py` 第 56 行，`env.reset(seed)` 位于局级 `try/finally` 之外。

在测试 Task 中令 `reset()` 抛出 `RuntimeError("reset failed")`，adaptation run 的实际结果为：

```json
{
  "status": "failed",
  "attempted_episodes": 0,
  "started_episodes": 0,
  "failed_episodes": 0,
  "unstarted_episodes": 2,
  "error": "RuntimeError: reset failed"
}
```

`episodes.jsonl` 为空。错误已在 run 级报告，不是被静默吞掉；问题在于尝试数与失败局账本没有体现已经发生的初始化尝试，而 evaluation 初始化失败会保存 `started=False` 的记录。

最小修改：统一记录一次失败尝试，保留 `started=False`、`score=null` 和 `reason="reset_error"` 或等价原因；明确 `attempted_episodes` 与 `started_episodes` 的口径。尚未开始的局不必调用 `end_episode`。增加一个抛错 reset 的测试即可。

风险采集正常 reset 不触发此问题，不影响本次三个示例的得分和轨迹。

## 5 R3：正式 RL 输入需要纯数值编码

**阶段：V1 训练接口准备，不作为 V0 框架阻断项。**

定位：`src/boardbench/environments/adapters.py` 第 13 行的 `RiskRL.encode`。

当前 reset 返回的编码为：

```python
(0, 2, 2, 2, 2, 0, 0, None)
```

最后一项是未终局的 `score=None`。这可以用于现有 Python 调用示例，但不能直接作为有限数值的策略网络输入。本次用标准库 `array('f', observation)` 转换，得到 `TypeError: must be real number, not NoneType`。

V1 开始训练前，可将最终 score 留在 `info`，使观测保持固定长度的数值向量；若必须保留在观测中，则使用明确数值和有效性标志。不要用 NaN 填补未知值。

届时再固定 shape、dtype、动作空间与训练库适配。当前第 22 行只接受原生 Python `int`，若训练库产生 NumPy 整数标量，也应在适配器边界进行合法转换。无需在 V0 因此安装完整 RL 框架。

## 6 已确认的架构与范围

| 部分 | 审查判断 |
| --- | --- |
| 游戏规则 | 抽牌不放回、BANK 得分、失败归零与奖励口径一致。 |
| 引擎复用 | 真实交互、RL 薄适配、参考搜索和 UI 共用规则实现。 |
| 搜索 | 复制局面并使用独立模拟随机流；模拟量与真实交互分别计数。 |
| 常驻 Solver | 跨局保留任务经验和参数，局内状态重置；Runner 不负责算法或调用判断。 |
| 学习示例 | job 真正更新模块，更新版本在后续决策中被使用。 |
| API 额度 | 成功和失败调用计量，未发出拒绝单独记录；耗尽后本地动作继续。 |
| 时间与异常 | 协作式超时后不执行过期动作，已发生转移反馈一次，终局分数不被收尾异常抹掉。 |
| checkpoint | 包含源码、依赖声明、配置、参数、经验、RNG 和 workspace；版本校验与新进程加载有效。 |
| 独立评测 | 现有示例每局从同一保存点恢复，标准路径不回写；R1 补齐文件副本契约。 |
| 可视化 | 简单页面够用；回放基于记录的观测，不重新采样游戏。 |
| V0 范围 | 没有提前实现强 RL、多游戏平台或复杂安全基础设施，符合已定目标。 |

## 7 可以交给本地 Agent 的下一轮任务

> 完成一个小型 V0.1 修补。优先修复 R1：恢复时将 Solver 产物与 workspace 一起放入每次恢复的私有副本，load 不接收父 checkpoint 的原目录；覆盖评测和恢复适应。补充加载时写缓存的回归测试，验证原保存点仍可校验、各评测局不共享缓存。顺手修复 R2 的 reset 失败尝试记录。R3 可以在 V1 接正式 RL 时完成，若现在修改则保持观测编码与测试一致。不要扩展服务、权限或分布式系统。源码变动后重新生成 checkpoint 和验收证据，更新 IMPLEMENTATION_STATUS.md，再打包回传。

源码变化后原 checkpoint 与新源码不匹配，是当前版本绑定设计的正常结果。旧 checkpoint 可以继续通过自己的源码快照恢复；不要通过删除哈希校验来“修复”版本不一致。

## 8 审查定位信息

上传 ZIP SHA-256：

```text
6f974bdaaa3eb9e01b1f99da5769cd7550839218a2a613b168915f41a707c697
```

本次验证的项目源码标识：

```text
cfb54260d371439c486c8f099aa5901d2208c4a9c04721a8e13af4715eac5837
```

本文行号以该上传版本为准。本审查提供了功能与接口层面的证据，不把测试通过解释为自主开发能力、学习收益或成本优势已经得到验证；这些研究结论需要 V1 的真实任务实验。
