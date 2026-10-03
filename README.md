# BoardGameBench · V1

可游玩、可训练、可复查的单人棋盘游戏实验平台。三个游戏共用 Python 规则引擎、
Runner、源码绑定 checkpoint 和本地网页；每个游戏提供随机、启发式、有限搜索和
真实训练的 masked PPO。包版本 **0.2.0**，保留 V0 风险采集及全部兼容测试。

| 游戏 | 固定规则 | 规则版本 |
| --- | --- | --- |
| Micro Tiles | 3×3 三色、双候选、邻接与整行奖励 | micro_tiles_v1 |
| Take It Easy! | 19 格、27 种牌、三个方向成线 | take_it_easy_standard_v1 |
| Harmonies | 单人 A 面 23 格、基础 32 动物、120 枚有限资源 | harmonies_solo_a_v1 |

V1 是预设方法与可信环境，不是自主开发算法的 agent；后者留到 V2。不需要 LLM、
API key、账号系统或云部署。规则依据 [V1 交接规范](docs/v1_handoff/BoardBench_V1_Agent_Handoff/README.md)。
实际结果、测试与限制见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。
旧版用法保留在 [V0_README.md](docs/V0_README.md)。

## 安装

Python 3.10+、Linux。引擎、启发式、搜索、Runner 和基础网页只用标准库：

    python3 -m venv .venv
    .venv/bin/python -m pip install -e '.[dev]'
    .venv/bin/python -m pytest -q

RL 训练和已训练策略推理需要 PyTorch/NumPy。可创建独立环境：

    python3 -m venv .venv-train
    .venv-train/bin/python -m pip install -r configs/requirements-train-tested.txt
    .venv-train/bin/python -m pip install -e '.[dev]'

本次实测 Python 3.10.4、PyTorch 2.4.1+cu121、NumPy 1.26.4；训练用 RTX 4090，
网页和统一评测用 CPU。工作区现有 .venv-train 通过 --system-site-packages
只读复用既有 Torch/NumPy，项目与 pytest 装在独立 venv，未修改基础环境。
运行时使用显式环境路径，避免悄悄切换 Python 或 CPU。

## 直接玩已有策略

在项目根目录任选一条，不需要重新训练：

    .venv-train/bin/python -m boardbench serve --config configs/micro_tiles_play.json
    .venv-train/bin/python -m boardbench serve --config configs/take_it_easy_play.json
    .venv-train/bin/python -m boardbench serve --config configs/harmonies_play.json

打开终端打印的 http://127.0.0.1:8765。serve 是 play 的别名。
远程使用 SSH 端口转发；默认只监听 loopback。页面可以切换三个游戏。

界面支持手动整局、策略单步/自动播放/暂停、提示采用或拒绝、中途接管并交还、
精确存档恢复，以及同初始随机条件的挑战和双方回放。Harmonies 的拿牌/放动物
可与资源放置交错；图案和分数来自引擎，不在前端重写规则。

- 策略目录是 outputs/v1/<game>/policies/。本地交付和 V1 审阅包内含实际权重、
  配置、源码 checkpoint 和验证摘要。Git 源码仓库不默认提交实验产物；
  单独克隆源码时，需要放入交付包的 outputs/v1/ 或按后述命令重新生成。
- 缺少策略目录时，只显示“内置配置，未评测”的随机/启发式/搜索，不冒充 RL。
- UI 按方法名和验证均分显示，不强行分成 easy/medium/hard。
- 存档在 runs/ui_saves/；保存按钮返回 ID，重启后仍可恢复。
  存档含精确隐藏供给，仅用于服务恢复，不进入策略观测。导出按钮保存带任务元数据的 JSONL。
- 同种子不代表 Harmonies 每回合市场相同；取牌/弃牌选择影响揭示时机。
  人类结束前，服务不返回 bot 终局或未来回放。

网页是无外部依赖的静态 HTML/CSS/JS，安装包直接包含同一份资源。构建副本：

    .venv/bin/python scripts/build_frontend.py --out build/ui

## 四类方法与数据边界

随机方法在 Harmonies 中先均匀选合法动作类型，再均匀选该类型参数，没有暗藏策略偏好。
启发式在 solvers/board.py：Micro 偏好邻接与成线潜力，Take It Easy! 偏好未破坏的
高价值线，Harmonies 使用地形分、叠放潜力、动物分与栖地匹配。
搜索在 Solver 内截取启发式排名候选，以独立未来采样和有限深度 rollout 比较；
不裁剪引擎合法动作。最终搜索预算由验证集选择。

PPO 使用两个 128 单元 tanh 层、策略头和价值头；三款游戏的动作空间为 18/19/4564。
Harmonies 编码动作类型、资源、格子、动物实例和方向；动态 mask 保留所有合法后继，
每个动物/锚点只去除等价方向。board_features_v1 向量长度依次为 46/109/626，可转 float32、
无 NaN/None 输入，终局 mask 全零且不再采样。

引擎只在自然终局发完整真实分数。实测训练显式使用
reward=(next_score_preview-current_score_preview)/reward_scale，gamma=lambda=1，
按原子动作计算；总回报望远镜相消到缩放后的终局分。验证/测试只统计真实终局分。
训练使用 PPO 随机采样，评测和网页使用同一网络的 masked argmax，不混入手写动作。

真实环境、训练和搜索随机流分开。普通观测和 Solver Context 不包含环境 seed、
RNG 或未揭示顺序；Runner 日志保留复现种子。fork(sim_seed) 从公开剩余资源重采未来，
不是精确存档的复制。这是可信代码的接口契约，不是恶意代码安全沙箱。

## 训练 → 验证选择 → 冻结测试

每游戏一个训练种子 739；训练环境种子从 100000 起，验证 200000..200031，
测试 300000..300127，严格不重叠。Pilot A 后仅做一次 4×预算扩大，保留旧结果。
仍是单训练种子实验，不能用多个测试局代替训练稳定性测量。

以下以 Micro 为例，其余任务替换名称。输出目录必须不存在，重跑请换后缀。
训练前检查所选卡无人占用；明确要求 CUDA，不会自动降级 CPU。

    CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python -m boardbench.training --config configs/micro_tiles_ppo_train_extended.json --out outputs/v1/micro_tiles/train_new
    .venv-train/bin/python -m boardbench.benchmark validate --task micro_tiles --training outputs/v1/micro_tiles/train_new --out outputs/v1/micro_tiles/validation_new
    .venv-train/bin/python -m boardbench.benchmark test --selection outputs/v1/micro_tiles/validation_new/selection.json --policies outputs/v1/micro_tiles/policies_new --out outputs/v1/micro_tiles/test_new

验证保留未训练和两个训练进度，选择一个真实训练 checkpoint 和一个搜索预算；
--extra-training <old_run> 可把更早候选纳入验证选择。测试比较四类方法与同网络
未训练对照；选择文件提前冻结，不依据测试结果回头挑选。
导出的策略通过已有 boardbench evaluate，每局重新加载同一冻结 checkpoint。

本次产物按游戏保存在 outputs/v1/<game>/：

| 子目录 | 内容 |
| --- | --- |
| train_pilot_a / train_pilot_b | 权重、配置、划分、训练曲线、训练时源码 |
| validation_pilot_b | 验证候选与冻结 selection.json |
| policies/<policy_id> | manifest/config/validation 和完整可迁移 checkpoint |
| test_pilot_b | 128 局/方法的 Runner 日志、结果与成本 |

命名 tmux 作业输出在 logs/v1/。scripts/launch_v1_pilot.sh、
launch_v1_validation.sh、launch_v1_test.sh 在已选主机上运行，使用 Bash + tee，
不覆盖输出或停止既有任务。训练启动器逐卡检查进程/显存并执行 CUDA 小测试。
多个任务使用多张单卡并行，不是分布式训练。

## Runner、存档和复现

    .venv/bin/python -m boardbench run --config configs/harmonies_search.json --out runs/harmonies_new
    .venv-train/bin/python -m boardbench evaluate --checkpoint outputs/v1/harmonies/policies/rl_trained_002048/export/checkpoints/ep_000000 --config configs/harmonies_eval.json --out runs/harmonies_eval_new
    .venv/bin/python -m boardbench replay --trajectory /path/to/events.jsonl

新任务显式采用 256 原子动作上限，避免沿用 V0 的 16 步而错误截断。
非法动作不改局面或 RNG；未完成局保留 null 分数，单列失败/截断/完成率。

Solver checkpoint 跨新局恢复策略，游戏存档精确恢复半局，两者不同。
系统 checkpoint 严格绑定源码；代码改变后，复制 checkpoint 的 source/ 到独立位置，
再 pip install -e COPY。不要直接可编辑安装原 checkpoint，也不绕过哈希校验。
加载时 solver 文件和 workspace 均进入私有副本，父产物不被缓存污染。
PPO 产物支持推理恢复；当前训练入口是新训练，不提供优化器级断点续训命令。

## 验收与边界

    .venv-train/bin/python -m pytest -q
    .venv/bin/python -m pip install -e '.[browser]'
    PLAYWRIGHT_BROWSERS_PATH=.browser-cache .venv/bin/python -m playwright install chromium
    PLAYWRIGHT_BROWSERS_PATH=.browser-cache .venv/bin/python scripts/browser_v1_check.py --server-python .venv-train/bin/python --out runs/v1_browser_new

测试覆盖独立计分样例、供给守恒、全部叠放、32 动物 × 6 方向、末袋见证、
中途恢复、不可见未来、RL 数值/mask、一致转移、真实更新、新进程加载、迁移导出与网页。
Harmonies 的“不足九枚不建立部分市场”是交接包项目裁定，不宣称官方 FAQ。
动物数据的第三方来源与核查范围保留在 JSON/handoff；网页使用程序化卡面，没有复制外部美术。

工程完成不等于 RL 胜过启发式，不宣称 SOTA 或普遍难度分层。
决策延迟只计 Solver 调用，加载、浏览器网络和绘制另列。
单机 pilot 不能证明跨硬件稳定延迟或多个训练种子的统计结论。

本地完整交付包可用以下命令重新打包（不覆盖已存在的文件）：

    .venv/bin/python scripts/package_v1.py --out deliverables/boardbench-v1-review-new.zip

包内包括源码、实际权重、训练/验证/测试原始记录、浏览器截图和逐文件 SHA-256 清单。
迁移恢复可运行 scripts/verify_v1_portable.py；该验证复用已安装数值依赖，但在临时 venv
使用复制的 checkpoint 源码、新工作目录和隔离 Python，不依赖原工作区的代码或权重路径。
