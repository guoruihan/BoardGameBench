# V1 与现有 V0 的集成说明

核查日期：2026-10-03。依据本次用户提交的 `boardbench-v0` 源码、README、接口文档与测试文件。本文是源码静态复核；本轮未重新运行整套测试。原包 `IMPLEMENTATION_STATUS.md` 记录 34 项测试通过，不将该记录表述为本轮新测结果。

V1 的目标是在现有项目上增加三个游戏、RL／搜索基线和人类交互页面。自主开发 agent 留到 V2。现有 README 将自主 agent 列入 V1 的旧表述需要同步修正。

## 1. 保留的骨架

无需重写 Runner、另建分布式服务或增加权限沙箱。保留风险采集任务和三个示例 Solver，作为回归测试与接口示例。

| 现有位置 | 已有能力 | V1 处理 |
| --- | --- | --- |
| `src/boardbench/contracts.py` | 任务自定义 JSON 观测／动作，`StepResult`，Solver 生命周期 | 保持基本契约 |
| `environments/__init__.py` | `Task(factory, spec, aggregate)` 与 `TASKS` 注册表 | 注册三个新任务 |
| `runner/engine.py` | 多局运行、适应／冻结评测、预算、事件、周期 checkpoint | 保留主流程，仅修复已定位的边界问题 |
| `runner/context.py` | 本地同步 job、mock 模型网关、模拟步计数 | V1 基线无需真实 LLM；保持现有能力 |
| `artifacts/store.py` | 日志、完整系统 checkpoint、源码和文件摘要 | 承接新 Solver 和卡牌数据；验证打包完整性 |
| `environments/adapters.py` | 风险采集 RL 薄接口、`SearchAccess` 和 `Simulation` | 为新任务增加薄适配，扩展模拟分支的合法动作访问 |
| `ui/server.py`、`ui/index.html` | 本地 HTTP 游戏界面、只读轨迹回放 | 增加任务视图、策略加载、提示／接管和同种子挑战 |

算法依赖可放在可选依赖组中；无需让运行风险采集的基础安装强制依赖 GPU 或完整训练框架。

## 2. 源码中的实际契约

### 2.1 环境与注册

```python
reset(seed: int) -> tuple[Observation, dict]
step(action: Action) -> StepResult

StepResult(
    observation,
    reward: float,
    terminated: bool,
    truncated: bool = False,
    info: dict = {},
)
```

自然终止时必须提供 `info["score"]`。不要用时间预算耗尽来伪造游戏自然结束；未完成的真实得分仍为 `None`，由 Runner 记录截断。

`InvalidAction` 的现有契约是：非法动作不改变状态，也不消耗环境 RNG。Runner 统计一次动作请求，但不计一次成功环境转移。新游戏继续遵守。

注册形态已存在：

```python
TASKS[task_id] = Task(factory, spec, aggregate)
```

- `factory()` 当前不接收参数。规则配置可先由注册项使用固定工厂闭包提供。
- `spec` 至少提供 `id`、`version`，通过 `context.task_spec` 到达 Solver。静态邻接表、动作格式和公开卡牌定义适合放在这里。
- `aggregate(results, requested)` 返回该任务的统计字段。不要把风险采集“未完成按零计入”的文字原封不动挂到新任务；明确报告完成率、完成局均分，以及采用何种缺失局处理口径。
- `get_task` 严格核对任务规则版本。`runner/config.py` 拒绝未知配置字段；当前 `task` 只支持 `id`、`version`，没有任意 `task.params`。

先实现每个游戏的一种固定 V1 规则即可。以后需要 Harmonies B 面或自然之灵时，再增加明确的配置字段或注册项，并记录独立规则版本；不要现在引入通用插件发现系统。

### 2.2 Solver

```python
start_task(context)
start_episode(observation, context)
decide(observation, context) -> Action
on_transition(transition, context)
end_episode(episode_result, context)
save(directory)
load(directory)
```

`solvers.SOLVERS` 保存工厂，工厂接收 `seed` 和 `solver.params`。适应 run 使用一个常驻实例；独立评测每局重新创建并载入同一 checkpoint。

现有 `random`、`reference_search`、`collaboration_demo` 不是三个新游戏的通用基线：

- `random` 只从 `task_spec["actions"]` 的静态列表采样，不支持变化的落点合法性。
- `reference_search` 明确使用 `DRAW`、`BANK` 和 `pot`。
- `collaboration_demo` 学的是演示用均值得分模块，不能作为 V1 强 RL 结果。

为新游戏增加独立 Solver 类或一层读取合法动作的共享适配。不得靠在 Runner 中判断 Solver 类型来切换策略。

### 2.3 参考模拟器

现有 `RiskCollect` 额外支持 `observe()`、`fork(sim_seed)`。授权时 Runner 通过 `SearchAccess` 暴露分支，模拟分支的 `step` 自动增加 `simulation_steps`。

新游戏建议沿用：

```python
observe() -> dict
legal_actions() -> list[Action]
fork(sim_seed: int) -> Environment
```

这些附加方法是 V1 要实现的扩展，并非已经出现在 `Environment` Protocol 的强制方法。实际代码可选择把合法动作放入公开观测，或在适配器暴露查询；保持一种清楚且统一的用法即可。

`fork` 复制当前公开状态，并用独立种子从公开信息允许的分布生成后续随机事件。不能把真实未来牌序交给搜索。模拟器也不能推进真实游戏的 RNG。Harmonies 的公开弃置历史／剩余数量需要足以定义这一采样分布。

现有 `Simulation` 仅包装 `step/observe/fork`。若搜索通过分支查询合法动作，增加对应的薄转发即可。

## 3. 最小增量改动顺序

### 3.1 先接一个完整游戏

1. 在 `environments/` 添加游戏引擎、规则数据和小型规则测试。
2. 注册 `Task`，添加能完成整局的合法随机 Solver。
3. 添加明确的新任务配置，直接走既有 `run` 和 `evaluate`。
4. 添加数值 RL 编码与动作解码；RL 与 UI 都调用同一个游戏引擎。
5. 加入前端任务视图，再增加基线训练与策略互动。

同一模式接入后续游戏。建议的文件组织可为 `environments/<game>/`、`solvers/<game>/`、`configs/<game>_*.json`；具体名称以实际实现为准。

### 3.2 必须调整新任务的预算

V0 默认 `max_actions_per_episode=16`，评测配置默认每局也只有 16 次动作请求。这不足以完成 Take It Easy! 的 19 次放置，更不适用于 Harmonies 的回合内多个动作。

每个新任务的示例配置必须显式给出足够的单局动作数、全 run 动作数和墙钟时间。Harmonies 区分：

- `episode`：完整一局；
- `turn_index`：游戏回合；
- `decision_id`／`environment_steps`：拿资源、放 token、拿牌、放动物、结束回合等决策。

不要将一个回合强行压成固定顺序，也不要为了避开动作预算自动替玩家放动物。任务回合编号可放进观测和 `info`，无需重新定义 Runner 的逐动作循环。

### 3.3 RL 训练可以独立运行，评测进入共同 Runner

现有 CLI 没有 `train` 子命令，现有 `run_job` 只是同步子进程执行工具。V1 可以增加独立训练脚本／模块，直接调用同一引擎的 RL 适配器；不必让每条训练轨迹都穿过完整 JSON 事件记录路径。

训练至少保存规则与编码版本、训练配置、种子、环境交互量、训练时长、权重和验证结果。选出的策略用 Solver 包装，再经相同 `evaluate` 路径测试和提供给网页。训练耗时与在线决策耗时分开报告。

3–5 个方法是每任务的目标上限，不是要求每个算法都重建一套运行器。不同难度主要来自实测策略／checkpoint 或搜索预算；命名和选择标准由基线文档约定。

### 3.4 UI 保留本地服务

现有可用路由：

```text
GET  /
GET  /api/state
POST /api/new
POST /api/action
```

`play` 的 `PlaySession` 当前显式拒绝非 `risk_collect`；HTML 也只认识 DRAW／BANK。需要增加按任务选择视图的能力。

建议最少增加：载入预训练策略、取得当前提示、切换自动接管、暂停自动播放。路由名称待实现确定；不要把它们写成已存在 API。

网页只发送结构化游戏动作并渲染返回观测。规则、卡牌匹配和计分仍在 Python 引擎；前端的合法动作高亮来自引擎结果。可编译静态前端后由本地服务托管，不要求把 Python 环境重写成纯浏览器实现。

现有 `Replay` 只读取日志中的观测，不重新运行引擎，这一点保留。多任务回放需从配套 `config.json` 或新增轨迹元数据取得任务 ID／版本，不能仅凭观测字段猜游戏类型。

同种子挑战需要显式选择相同任务、规则版本和局种子。当前 `/api/new` 只自动派生下一个种子；指定挑战种子的接口是 V1 新增。提示、候选预览和 bot 分支搜索不能改变真实随机流。

## 4. 系统 checkpoint 与局面存档是两个对象

| 对象 | 保存什么 | 用途 | 当前状态 |
| --- | --- | --- | --- |
| Solver／系统 checkpoint | 代码快照、参数、任务经验、训练状态、Solver RNG、依赖声明 | 从同一策略开始独立新局，继续训练／适应 | V0 已有目录与校验流程 |
| 游戏局面存档 | 棋盘、市场、待放资源、卡牌进度、回合标志、真实随机流状态、终止状态 | 网页暂停／恢复、接管同一局、复现局面 | V0 未实现通用保存恢复接口 |

V0 checkpoint 不恢复半局。不要只恢复模型权重却声称恢复了棋盘；也不要把上一局棋盘塞入 Solver 持久经验，在独立评测的新局继续使用。

V1 可增加小型 `export_state()/import_state()`，名字仅为建议。存档同时记录任务规则版本。Harmonies 必须能在三枚资源尚未放完时恢复；仅保存回合末棋盘不够。

完整局面存档中的 RNG／未揭示顺序供游戏服务恢复，不作为 Solver 观测。它与搜索 `fork(sim_seed)` 的语义不同：前者精确恢复同一局，后者采样符合当前信息的可能未来。

现有 checkpoint 严格绑定源码摘要。修改 V0 成 V1 后，旧 checkpoint 被新源码拒绝是当前设计行为，不是需要绕过的错误。旧结果仍可通过旧 checkpoint 自带源码的独立环境恢复；不要求本轮实现跨版本自动迁移。

卡牌 JSON 和必要静态资源必须进入可安装包及源码快照。`source_files()` 当前会收集 `src/boardbench/` 下所有文件，但 setuptools 的 package-data 目前只显式声明 UI HTML；新 JSON／前端构建产物应补充 package-data 声明并验证安装结果。

## 5. 前次审阅三个问题的本轮核实

### 5.1 “load_checkpoint 依赖原路径”：当前源码未复现该说法

当前源码没有名为 `load_checkpoint` 的函数。实际流程为：

```text
validate_checkpoint(path)
restore_workspace(path, new_workspace)
solver.load(path / "solver")
```

`CollaborationDemo.load` 从传入目录读取状态，不依赖原 run 路径。源码包含移动 checkpoint、不同 cwd、新进程恢复测试；`scripts/verify_bundle.py` 还建立独立环境，只安装 checkpoint 源码副本。原包保存了对应验收记录。

因此不把这一条列为当前必须重修的 bug。V1 新增 Solver 的门槛是：持久状态不写入不可迁移的原训练目录依赖；复制产物后仍能载入。若基线需要外部权重文件，须随 checkpoint 收集或在明确的产物包中提供，不能默默依赖开发机路径。

### 5.2 reset 异常计数：适应 run 仍有缺口

`run_episode` 在建立 `EpisodeResult` 和进入自己的 `try` 之前调用 `env.reset(seed)`。在适应 `run` 中，reset 抛错由外层捕获，run 会标为失败，但本局没有加入 `results`；`attempted_episodes`、`failed_episodes`、`failure_reasons` 因而没有记录这次失败尝试。

独立 `evaluate` 已有初始化异常处理，会写 `started=False` 的失败记录；不要将这一缺口泛化成两个路径都没有处理。

**修复门槛：**适应路径一次 reset 抛错应生成一次失败尝试，`score=None`、零成功环境转移，`started=False`，原因标明 reset／初始化；不得调用该局的 `decide` 或重复写局记录。保留 run 的整体失败状态，并增加直接行为测试。无需重写 Runner。

### 5.3 RL 初始输入含 None：仍然存在

`RiskRL.encode` 输出八元组，最后一项直接取 `obs["score"]`；未终止时是 `None`。它是演示薄接口，并非 `gymnasium.Env`，也没有 `observation_space`／`action_space`。现有一致性测试验证的是它与引擎的对应关系，不保证张量可用性。

**V1 修复门槛：**新任务 RL 编码在 reset、中途和终局均应是固定结构、有限数值、明确 dtype；缺失终局分数留在 `info` 或使用数值加有效标记，不能把 `None` 输入网络。若选择 Gymnasium／算法库，再提供相应空间与合法动作 mask。风险采集旧薄接口可保留兼容，也可显式升级并更新文档；不把旧编码照搬到新任务。

## 6. 命令：已存在与拟新增

以下命令在提交的 V0 中确实存在。运行目录必须不存在，重跑时使用新目录名。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q

python -m boardbench run --config configs/risk_random.json --out runs/random_demo
python -m boardbench run --config configs/risk_search.json --out runs/search_demo
python -m boardbench run --config configs/risk_collaboration.json --out runs/collaboration_demo
python -m boardbench evaluate --checkpoint runs/collaboration_demo/checkpoints/ep_000006 --config configs/risk_eval.json --out runs/eval_demo
python examples/rl_rollout.py
python -m boardbench play --config configs/risk_play.json
python -m boardbench replay --trajectory runs/collaboration_demo/events.jsonl --port 8766
```

已有证据生成命令为 `python scripts/collect_evidence.py --out NEW_DIRECTORY`。浏览器验收需要可选 `[browser]` 依赖和 Chromium；没有浏览器时不得将浏览器操作标为已验证。

下面是 V1 应交付的命令形态，配置、模块和路径均为占位示例，**现在不可直接运行**：

```bash
python -m boardbench run --config configs/<game>_random.json --out runs/<new_run>
python -m boardbench play --config configs/<game>_play.json
python -m boardbench.<training_module> --config configs/<game>_<method>_train.json
python -m boardbench evaluate --checkpoint <exported_checkpoint> --config configs/<game>_eval.json --out runs/<new_eval>
```

训练导出到完整系统 checkpoint 的连接逻辑尚不存在。实现者可增加小型导出入口／工具函数，使用已有 manifest 和 `save_checkpoint`，并在 README 写出实际命令；不要伪造一个当前不存在的 `boardbench train`。

## 7. 兼容与验收清单

- [ ] 原风险采集引擎、三个示例 Solver、现有 CLI 和事件格式继续可用。
- [ ] V0 既有测试通过；新增测试覆盖具体规则风险，不追求无意义的测试数量。
- [ ] 三个游戏的 `reset/step` 返回现有契约；自然终局提供真实 `info.score`。
- [ ] 新游戏的非法动作保持状态及 RNG 不变；搜索和网页预览不改变真实随机流。
- [ ] 新任务配置不会沿用 16 次动作上限导致系统性截断。
- [ ] 同规则、同种子、同动作，在直接引擎、RL 解码和网页后端中产生一致转移。
- [ ] RL 观测始终可数值化，动作 mask／解码一致，不含真实未揭示牌序。
- [ ] 每个策略均可经相同 Runner 独立评测；真实得分、模拟步数、时间与训练投入分别记录。
- [ ] Solver checkpoint 能迁移；游戏存档能恢复当前半局；两者不混用。
- [ ] 新卡牌数据和必要前端资源进入安装包与 checkpoint 源码快照。
- [ ] 回放继续读取实际记录，不重新抽牌或重算出另一条轨迹。
- [ ] 适应路径 reset 失败有完整局尝试记录。
- [ ] README 更新 V1／V2 边界，列出实有命令、实际验证证据和仍未实现的功能。

完成上述集成后，V1 的新能力主要落在游戏引擎、适配器、基线和页面中；V0 已有的运行、保存与独立评测流程继续承担共同实验记录。
