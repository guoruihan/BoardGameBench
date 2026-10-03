# BoardGameBench · V1.1

可游玩、可训练、可复查的单人棋盘游戏平台。三个游戏共用 Python 规则引擎、Runner、
源码绑定 checkpoint 和本地网页。V1.1 对应包版本 **0.3.0**；不增加新游戏或自主 agent 框架。

| 游戏 | 固定规则 |
| --- | --- |
| Micro Tiles | 3×3 三色、双候选、邻接与整行奖励 |
| Take It Easy! | 19 格、27 种牌、三个方向成线 |
| Harmonies | 单人 A 面 23 格、基础 32 动物、120 枚有限资源 |

[实现、结果与验收报告](IMPLEMENTATION_STATUS.md) ·
[V1 review](docs/BoardBench_V1_Review.md) ·
[原始规则规范](docs/v1_handoff/BoardBench_V1_Agent_Handoff/README.md)

## 安装与直接游玩

Python 3.10+、Linux。引擎、随机、启发式、搜索及基础网页只用标准库：

    python3 -m venv .venv
    .venv/bin/python -m pip install -e '.[dev]'

PPO、模仿学习及其推理需要 PyTorch/NumPy：

    python3 -m venv .venv-train
    .venv-train/bin/python -m pip install -r configs/requirements-train-tested.txt
    .venv-train/bin/python -m pip install -e '.[dev]'

本工作区已有 .venv-train，通过 --system-site-packages 只读复用已安装的
PyTorch 2.4.1+cu121 / NumPy 1.26.4，没有修改基础环境。使用显式 Python 路径。

在项目根目录运行：

    .venv-train/bin/python -m boardbench serve --config configs/harmonies_play.json

浏览器访问 http://127.0.0.1:8765；页面内切换三个游戏。
服务只监听启动主机的 loopback。远程使用 VS Code Ports 或 SSH 将**这台主机**
的 127.0.0.1:8765 转发到本地；转发端口被占用时可换本地端口，不能转发另一台训练节点。

页面支持手动游玩、提示/拒绝、策略单步与自动接管、暂停、精确存档恢复、
同初始随机条件挑战和双方回放。Harmonies 可在资源放置之间拿牌、放动物，
完成卡释放槽位；存档保留卡片进度、棋盘动物和待放资源。

默认读取 outputs/v11/<game>/policies/。完整交付包包含真实权重；Git 只发布源码，
单独克隆后需放入交付包的 outputs/v11/ 或重新训练/导出。
没有策略目录时显示“内置配置，未评测”的方法，不伪装成已训练模型。
存档在 runs/ui_saves/；服务端私有供给不会进入策略观测。

## V1.1 的方法与实验边界

保留五类以内的方法：随机、启发式、预算搜索、PPO，以及 Micro/TIE 的**启发式模仿学习**。

- PPO 不使用教师动作。Micro/TIE 的共享动作评分网络使用公开的相同/冲突/空位计数、
  线长度与当前牌数值；局部输入宽度 12/16，隐藏层 64。价值网络读取全局特征。
  这些是人工设计的表示，因此必须与同表示的未训练网络比较。
- Harmonies PPO 保留两层 128 的平坦 MLP。环境本身仍使用原来的有限数值接口和合法 mask；
  神经策略另有严格版本化的 flat_mlp/action_mlp 元数据，验证/导出不猜测网络宽度。
- 模仿学习用启发式生成软动作目标，训练状态由 80% 启发式/20% 随机动作采集。
  推理仅调用真实网络权重，不调用教师。它是监督学习对照，不记为纯 RL 收益。
- PPO 用完整局 Monte Carlo 优势，gamma=lambda=1；训练的 score-delta 奖励总和等于
  缩放后的终局分。评测只用真实终局分。训练采样，评测/UI masked argmax。
- 搜索每步重新规划，不会跨局自动学习。预算记录候选数、采样数、模拟步上限、
  深度单位 atomic_action/turn 和叶估值。turn 是目标完整回合数，不保证预算足够；
  验证日志记录实际达到目标的 rollout 数。
- 搜索从公开剩余资源独立采样未来；观测不含真实 seed/RNG/供给顺序。接口面向可信代码，
  不承诺对恶意 Python 策略隔离。

实验计划在 configs/v11_plan.json：训练种子 811/812/813；
验证 5000000..5000063，最终测试 6000000..6000127。
每次训练至多 900 秒（在 batch 边界检查），Micro/TIE PPO 16384 局、
Harmonies PPO 8192 局，模仿学习 1024 局。旧 V1 测试集只保留为历史参考。
不得把多个测试局当作独立训练重复，也不得将不同测试集的均分直接当配对提升。

验证保留所有来源 run 和 checkpoint，逐 run 选择最佳进度；不只报告最好的训练种子。
每个来源在评测前检查任务、规则、架构及 train 与主 validation/test 是否有交集。
ID 由 run 内容摘要和 checkpoint 阶段组成，移动目录不改变身份，同名阶段不会静默丢失。
选中的模型和对应未训练对照一起冻结，最后才在新测试集运行。

## 复现实验

以下输出路径必须不存在。已有交付包时无需重复生成；重跑要使用新目录或干净源码副本，
不要覆盖旧实验。训练前确认 GPU 无其他进程；启动器不会停止或驱逐既有任务。

    .venv/bin/python scripts/prepare_v11.py
    bash scripts/launch_v11_job.sh 1 micro-ppo-811 .venv-train/bin/python -u -m boardbench.training --config outputs/v11/experiment_plan/micro_tiles_ppo_811.json --out outputs/v11/micro_tiles/ppo_811

配置矩阵包含全部三个种子的任务。模仿学习使用 -m boardbench.imitation。
scripts/run_v11_queue.py 可在一张已检查空闲的卡上依次执行矩阵中的独立任务。
命名 tmux 和 tee 日志保存在 logs/v11/；多卡用于独立任务，不是分布式训练。

通用验证/测试入口（将路径替换为实际新 run）：

    .venv-train/bin/python -m boardbench.benchmark validate --task micro_tiles --training TRAIN_RUN --extra-training EXTRA_RUN --search-config configs/v11_plan.json --retain-runs --workers 4 --out VALIDATION_OUT
    .venv-train/bin/python -m boardbench.benchmark test --selection VALIDATION_OUT/selection.json --policies POLICIES_OUT --workers 4 --shard-size 32 --out TEST_OUT

--extra-training 可重复。并行验证有源码一致性检查；最终评测要求运行源码与冻结验证一致。
分片测试仍调用同一个 Runner，每局独立恢复策略；保留各 shard 的原始 episodes/events，
不把改写拼接的日志冒充原始日志。aggregation.json 明确列出分片和种子覆盖。
本轮导出后运行 scripts/label_v11_policies.py，将页面标签缩短为方法与种子，避免手机
下拉框被长标签撑宽；只改显示 label，完整局数、来源和权重摘要保留在 manifest 中。

产物布局：

| 路径 | 内容 |
| --- | --- |
| outputs/v11/experiment_plan/ | 实验计划、摘要及每个 run 的配置 |
| outputs/v11/micro_diagnostic_a/ | 固定 8 局诊断、失败/改善曲线及采样/argmax 对比 |
| outputs/v11/<game>/ppo_* / imitation_* | 配置、数据划分、全部权重与训练时源码 |
| outputs/v11/<game>/validation/ | 每个候选逐局结果与冻结选择 |
| outputs/v11/<game>/policies/ | 来源记录、配置、模型、源码绑定 checkpoint |
| outputs/v11/<game>/test/ | 新测试集的原始 Runner 分片与统计 |
| outputs/v11/report.json | 三训练种子汇总、配对差、分项得分与全部训练成本 |

scripts/report_v11.py 从原始日志重算报告。训练种子间标准差和固定模型的环境种子配对区间
分开报告；配对区间采用均值 ± 1.96 × 样本标准差 / sqrt(n)。
Harmonies 额外报告地形、动物分、动物放置和完成卡数。纯 solver 时间、Runner 单局
wall time、模型加载及前期训练成本不混为一项；并行度明确记录，不宣称整体加速。

## 验收、恢复与历史版本

    .venv-train/bin/python -m pytest -q
    PLAYWRIGHT_BROWSERS_PATH=.browser-cache .venv/bin/python scripts/browser_v11_animals.py --out runs/animal_browser_new
    PLAYWRIGHT_BROWSERS_PATH=.browser-cache .venv/bin/python scripts/browser_v1_check.py --server-python .venv-train/bin/python --out runs/browser_new
    .venv-train/bin/python scripts/verify_v11_portable.py --artifacts outputs/v11 --out runs/portable_new

浏览器工具需可选 browser 依赖和 Chromium。动物定向测试用库存守恒的受控初始供给，
所有后续动作都经过真实引擎合法检查；通过实际按钮操作验证，不新增后门 API。
迁移检查复制源码和 checkpoint，在新 venv、新目录、隔离 Python 中重放全部训练策略，
比较动作和完整观测，核对父产物与副本未变。复用已安装数值依赖，不声称空系统复现。

标准 wheel 构建需要 pyproject 声明的 setuptools，使用隔离构建；
无需 Torch 即可运行引擎和非神经策略。完整源码绑定 checkpoint 创建/恢复使用 editable 安装。

游戏存档与 Solver checkpoint 不同。当前训练入口仍是新训练，不提供优化器级续训；
已有 checkpoint 支持推理恢复。不要绕过源码或内容哈希检查。
V1 老策略须用各自的源码副本或旧交付包运行，不能直接塞入新版策略目录。

V1 已冻结的源码、权重、原始记录和 ZIP 保持不变：
[V1 用法](docs/V1_README.md)、[V0 用法](docs/V0_README.md)。
动物牌数据和末袋裁定沿用交接包，没有改写规则来提高成绩。无 SOTA 或普遍难度分级声明。

完整 V1.1 包包含真实权重、原始实验记录、浏览器截图和逐文件 SHA-256 清单：

    .venv/bin/python scripts/package_v11.py --out deliverables/boardbench-v1.1-review-new.zip
