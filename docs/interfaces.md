# 实际接口与扩展点

## V1 增量

三个新任务提供 observe、legal_actions、score_breakdown、task_spec、save_state/load_state、
fork(sim_seed)。精确 snapshot 包含隐藏后续，只用于局面恢复；fork 仅根据公开信息重采未来。
BoardRL 返回 features（固定 float32 兼容向量）和 action_mask；终局 mask 全零且禁止采样。
Harmonies 采用 4564 个稳定动作 ID，合法列表只对相同动物/锚点的等价方向去重。

V1 Solver 的 Context.config 不含根 seed，end_episode 的 result.seed 为 None；Runner
自己的日志保留真实环境种子。V0 风险采集接口保持兼容。这不是恶意代码隔离。
decision_end 新增 solver_seconds（纯调用时间），原 elapsed_seconds 与资源口径不变。

网页新增 POST /api/hint、/api/bot-step、/api/save、/api/load、/api/challenge、/api/export；
/api/new 接受 task_id 与可选 seed，64 位种子可作为十进制字符串传输。/api/state 不返回
完整 snapshot；只有人类自然结束后才返回双方挑战回放。旧 play/replay CLI 保留，serve 为 play 别名。
Replay 必须从 run_start.task 或配套 config.json 读取任务，不按观测字段猜测或重跑引擎。

PolicyRegistry 校验配置/权重摘要和 checkpoint 源码，加载私有副本并缓存常驻模型；
manifest 记录相对路径、验证均分、训练投入与代码版本。standalone training 经 benchmark
导出完整系统 checkpoint，再调用共同 evaluate，未用另一个隐含评测规则引擎。

以下原有契约继续有效。

公共类型位于 `src/boardbench/contracts.py`。Observation 与 Action 均为任务定义的
JSON 可序列化对象，Runner 不限制具体字段或动作编号。钩子收到的观测／转移为独立副本。

## Environment

```python
reset(seed: int) -> tuple[Observation, dict]
step(action: Action) -> StepResult
```

StepResult 包含 observation、reward、terminated、truncated、info。
自然终止必须在 info 中提供 score，可提供 reason；正常引擎不主动做预算截断。
RiskCollect 额外提供 `observe()` 和 `fork(sim_seed)`；未来任务可通过自己的适配层实现模拟。

新增环境时，在 `environments.TASKS` 注册 `Task(factory, spec, aggregate)`。
`spec` 至少包含 id、version，动作 schema／静态拓扑等由任务定义。
`aggregate(results, requested)` 返回任务自己的汇总字段；通用 Runner 不内置未完成局计分规则。
规则版本不匹配会拒绝启动。单人同步调度是 V0 的范围。

`RiskRL`：reset 返回 `(encoded_observation, info)`，step 返回
`(encoded_observation, reward, terminated, truncated, info)`；动作 0=DRAW、1=BANK。
观测元组固定为 8 个数值：
`(pot, count_1, count_2, count_3, count_fail, steps, terminated_int, score_slot)`。
未终止时 score_slot=0，terminated_int=0 表示该槽没有最终得分；自然终止时
terminated_int=1，score_slot 为真实得分（也可能是 0）。原始引擎观测中的未知 score
仍为 None，终局 info.score 不变。编码可直接转换成 `array('f', obs)` 或
`numpy.asarray(obs, dtype=numpy.float32)`，没有 None／NaN 或额外隐藏信息。
动作接受 `numbers.Integral` 标量（含 Python int、IntEnum、NumPy 整数），在边界转换
成原生 int；拒绝 bool、float、数组及越界值。运行时不依赖 NumPy，接口不是 Gymnasium Env 子类。

## Solver

```python
start_task(context)
start_episode(observation, context)
decide(observation, context) -> Action
on_transition(Transition(observation, action, result), context)
end_episode(EpisodeResult(...), context)
save(directory: Path)
load(directory: Path)
```

只需实现需要的生命周期钩子；decide、save、load 需要实际实现。
`version` 属性用于关联动作与 Solver 产物版本。构造工厂接收 seed 和配置 params，
在 `solvers.SOLVERS` 注册。适应 run 全程使用同一个实例；评测每局新建实例。
保存持久学习状态、训练状态和自身 RNG，局内临时状态在 start_episode 清空。
学习由 Solver 决定；`allow_learning=false` 时不得更新持久参数和任务经验。

恢复时传给 `load(directory)` 的是私有文件副本，不是父 checkpoint/solver。
允许加载缓存和后续磁盘引用。恢复适应的副本位于新 run 的 `restored_solver/`；
每局评测的临时根目录下有 `workspace/` 和 `restored_solver/`，在该局回调全部结束后清理。
`save(directory)` 仍由 Solver 负责导出后续恢复所需的全部持久状态。

## Context

| 接口／字段 | 行为 |
| --- | --- |
| `consult(request: dict)` | 返回 ConsultResult(status, text, usage, elapsed_seconds, error)。status 为 success／failed／budget_exhausted。 |
| `run_job(argv, resources=None)` | 同步执行 argv，无 shell，cwd 为 workspace。仅支持 resources.timeout_seconds，不分配 GPU／CPU 配额。 |
| `emit(metrics: dict)` | 写 solver_report，保留自报来源，不修改真实得分和资源。 |
| `remaining_budget()` | 返回 wall_seconds、action_requests、model_calls 剩余额度。 |
| `workspace` | 可写的绝对 Path；持久化文件中应使用相对路径，避免依赖旧工作区。 |
| `mode / allow_learning / task_spec` | 本次运行约定及公开任务信息。 |
| `reference_simulator` | 未授权时为 None；授权且游戏开始后可 fork(sim_seed)，模拟分支可 step／observe／fork。 |

JobResult 包含 status、returncode、相对 run 根目录的 log_path、elapsed_seconds、error。
状态为 success／failed／timeout／budget_exhausted。失败 job 不自动重试，已发生耗时照常计入。
示例 job 不调用外部模型；跨进程共享模型额度留给后续 agent 接入。

ModelClient 契约是 `consult(request, timeout) -> ConsultResult`，provider 字段标明来源。
Context 统一计量并捕获异常。当前工厂只允许 mock；后续真实 client 必须在网络层实施 timeout。

## 事件与产物

事件共有 seq、run、kind、episode、phase、decision_id。真实转移额外包含 step、request_id、
observation、action、result、solver_version、decision_seconds。
model_start/model_end 用 call_id 关联；job_start/job_end 用 job_id 关联。
局间事件 decision_id=null；同一文件的 seq 严格递增。

episodes.jsonl 提供 score、status、reason、error、started、action_requests、environment_steps、resources。
status 为 completed／truncated／failed；已自然终止但收尾异常时保留 completed 与真实得分，
error 非空，run 状态为 failed。最后的 run_stop 和 summary 提供完整局数及成本摘要。
reset 异常记录 `started=false`、`status=failed`、`reason=reset_error`、`score=null`，
并写一条 `episode_initialization_failed` 事件；不发 episode_start／transition，也不调用
start_episode 或 end_episode。attempted_episodes 包含该失败尝试，started_episodes 不包含。

新增算法不应修改 Runner。测试 `test_runner_task_independence` 用不同观测结构和字典动作
验证了任务独立性；规则、UI、RL 一致性和模拟 RNG 隔离均有直接行为测试。
