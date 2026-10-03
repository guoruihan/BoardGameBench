# BoardBench V0

用于研究常驻求解系统的本地实验骨架。当前实现风险采集测试环境、三个示例
Solver、多局适应、预算与日志、同步训练 job、源码绑定 checkpoint、独立评测、
RL／搜索薄适配，以及本地游戏页面和轨迹回放。

范围依据 [V0 实施规范 v0.3](docs/V0_Implementation_Spec.md)。
微型拼板、Harmonies、强 RL 与自主开发 agent 属于 V1。
`collaboration_demo` 是手写接口测试程序；模块有真实更新，但不代表自主编程或策略提升。

当前为 V0.1 修补版（包版本 0.1.1）：修复恢复时 Solver 文件隔离与 reset 失败记录，
并将 RL 观测改为纯数值编码。逐条判断见
[review 处理说明](docs/BoardBench_V0_Review_Response.md)。旧保存点仍绑定自己的源码，
新版 checkpoint 与证据另存于 `runs/acceptance_v0_1/`。

## 安装与快速运行

需要 Python 3.10+、Linux。运行时无第三方依赖；全部核心验收不需要 GPU 或 API key。
推荐在项目自己的环境中执行，当前工作区已有 `.venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q

python -m boardbench run --config configs/risk_random.json --out runs/random_demo
python -m boardbench run --config configs/risk_search.json --out runs/search_demo
python -m boardbench run --config configs/risk_collaboration.json --out runs/collaboration_demo

python -m boardbench evaluate \
  --checkpoint runs/collaboration_demo/checkpoints/ep_000006 \
  --config configs/risk_eval.json --out runs/eval_demo

python examples/rl_rollout.py
python -m boardbench play --config configs/risk_play.json
python -m boardbench replay --trajectory runs/collaboration_demo/events.jsonl --port 8766
```

运行输出目录必须不存在。再次运行时换一个目录名；程序不会覆盖或拼接已有记录。
CLI 打印状态、summary 和 checkpoint 路径。退出码：正常结束／预算停止为 0，
执行失败为 1，配置或加载错误为 2；是否完成全部局数以 summary 为准。

`play`／`replay` 默认仅监听 `127.0.0.1`，终端显示地址，Ctrl+C 结束。
远程机器可通过 SSH 转发对应端口后在本机浏览器打开。
UI 支持 DRAW、BANK、新局；回放支持选择局、前后步，只读取日志中的观测。

## 配置和示例

配置使用 JSON；未填写的默认值写入运行目录的 `config.json`。未知字段、非法预算、
未支持的模式组合会报错。默认值见 `src/boardbench/runner/config.py`。

| 示例 | 作用 |
| --- | --- |
| `risk_random.json` | 本地随机动作，模型调用为零，固定种子复现。 |
| `risk_search.json` | 获授权的独立分支模拟，有限 rollout，单独统计模拟步数。 |
| `risk_collaboration.json` | 6 局、一次 mock 额度；后续拒绝后继续本地决策，每两局更新一个均值得分模块。 |
| `risk_eval.json` | 固定 4 个评测种子，每局重新加载同一 checkpoint，禁止学习，独立额度。 |

协作模块在 `end_episode` 收集已完成局的得分，用同步子进程拟合均值得分。
后续动作读取 `pot - (1 + mean_score)`，版本和参数更新写入 `solver_report`。
该预测器只为演示更新与使用路径，不是风险采集的强基线。

V0 支持 `adaptation + allow_learning=true` 和
`evaluation + allow_learning=false`。后者通过 `evaluate` 入口运行，任务、Solver、
模型和能力配置来自 checkpoint，评测配置仅接受种子、每局预算和模式选项。
恢复适应可用 `run --checkpoint PATH --config CONFIG --out NEW_DIRECTORY`：保留任务、
Solver、能力与模型配置，在新 run 中开始新局；参数和任务经验继续，局号和预算重新计算。

## 结果与恢复

每次运行包含：

```text
config.json       生效配置（含默认值）
events.jsonl      顺序事件、真实转移、咨询、job、版本、保存与停止
episodes.jsonl    局种子、真实得分／null、状态与资源
summary.json      局数、分数汇总、资源、版本及产物路径
jobs/             子进程输出
workspace/        Solver 可变工作文件；评测副本用后丢弃
restored_solver/  恢复适应时的私有 Solver 文件副本（仅恢复 run 创建）
checkpoints/      按完成局数周期保存的完整产物
```

checkpoint 包含 `manifest.json`、`solver/`、`workspace/` 与可安装的 `source/`。
manifest 记录文件摘要、源码标识、配置、进度和保存前资源。恢复校验内容和源码一致性；
若当前实现已变化，请使用独立环境安装该 checkpoint 的源码：

```bash
cp -a /path/to/checkpoint/source ./snapshot-source
python -m pip install -e ./snapshot-source
python -m boardbench evaluate --checkpoint /path/to/checkpoint \
  --config /path/to/configs/risk_eval.json --out /path/to/new_evaluation
```

先复制源码再安装：可编辑安装会生成 egg-info，不能把这些新文件写入原 checkpoint。
需要可编辑安装以定位源码快照，V0 不支持仅安装 wheel 后创建 checkpoint。
不依赖原工作目录的绝对路径，不自动切换源码或依赖，不恢复被中断的半局。
参数和 RNG 状态用 JSON 保存，不使用 pickle。

`load()` 获得私有 Solver 文件目录，可在其中生成缓存或保留磁盘模型引用。
该副本与 workspace 一起保持到使用结束；每个评测局重新复制，局结束后删除，
恢复适应时的副本保存在新 run 中。父 checkpoint 不作为可写加载目录。

自然终止记录真实得分。预算截断／异常未完成时得分为 `null`，不把累计收益当作 BANK。
风险采集另报告覆盖所有请求局数的均分，未完成／未开始按零计入，分母是请求局数。
`attempted_episodes` 区分初始化尝试与成功 reset 的 `started_episodes`。
reset 报错计一次失败尝试：`started=false`、`score=null`、`reason=reset_error`，
不会调用尚未开始游戏的 `end_episode`；适应 run 停止，独立评测保留各次尝试。

真实动作请求包含非法动作；模型失败调用占额度，未发出的拒绝单独计数。
决策耗时已经包含其中的咨询或 job，不能再加分项作为总成本。
评测在线成本与历史适应投入分开；恢复适应后的 checkpoint 也累计之前的投入。
checkpoint 的资源捕获点是本次保存开始前，run 总耗时包含实际保存时间。

## 验收证据与审阅包

完整验收和证据索引见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。
接口及兼容决定分别见 [interfaces.md](docs/interfaces.md)、[decisions.md](docs/decisions.md)。

可重新生成独立证据目录：

```bash
python scripts/collect_evidence.py --out runs/acceptance_v0_new
```

该脚本实际执行安装、测试、三个示例、两次独立评测和 RL 示例，保留命令和输出，
核对日志计数／得分、评测复现及原产物不变。
如需自动实际浏览器操作和截图，再安装可选工具（已固定为本机支持的版本）：

```bash
python -m pip install -e '.[browser]'
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.browser-cache"
python -m playwright install chromium
python scripts/collect_evidence.py --browser --out runs/acceptance_v0_with_ui

python scripts/package_review.py --evidence runs/acceptance_v0_with_ui \
  --out deliverables/boardbench-v0.1-review-new.zip
python scripts/verify_bundle.py deliverables/boardbench-v0.1-review-new.zip \
  --out runs/bundle_verification_new
```

审阅 ZIP 内证据保留所选目录名，例如 `runs/acceptance_v0_1/`。打包不包含虚拟环境、浏览器缓存、
调试历史或凭据。`verify_bundle.py` 解压到临时目录，新建环境，仅安装 checkpoint 源码，
以隔离 Python 进程恢复评测并比较动作和分数；验证结束自动清理这个临时副本。

## 已知边界

- 当前只实现确定性 mock provider；真实 API 接入未实现、未验证。mock 耗时不能作为 LLM 延迟结论。
- Python Solver 钩子采用协作式超时；返回后停止继续行动，记录实际超出量。
  本地 job 超时会终止进程组。这里不承诺任意 Solver 的硬实时抢占。
- 固定种子复现要求墙钟预算不构成截断；不同机器在超时时可能停止于不同步骤。
- `allow_learning` 和参考模拟器授权是合作代码的实验契约，不是恶意代码隔离或权限沙箱。
- 默认每两局保存，只保留按周期完成的保存点；不足周期的末尾不额外自动保存。
  超时或收尾失败后不开始新保存；失败的 `.partial` 目录不属于有效 checkpoint。
- 风险采集的 RL／UI 编码及三个示例具有任务专属性；通用 Runner 不读取牌袋或棋盘字段。
