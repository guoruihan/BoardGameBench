# 双人环境族：基准 bot 的训练与选择方法

依据源码 `6bcb711` 的 `families/{train,neural,ppo,play,evaluate}.py` 及实际
`outputs/v160/bots/<game>/<run>/config.json`。参数以各运行配置和模型身份为准；
本文不声明所有实验已经结束。启动恢复见 [Runbook](Runbook.md)，统计协议见
[Evaluation](Evaluation.md)，规则迁移见 [Rules Corrections](Rules_Corrections.md)。

## 公开候选表示与 CEM

每款游戏独立定义 24 维有限候选特征 `features(observation, action)`，第 0 维
为启发式评分，其余描述公开局面与动作。这是紧凑人工表示，不是完整棋盘编码。
策略只接收当前玩家观察和合法动作；模拟器持有的对手私牌、隐藏牌序与 snapshot
不会传给策略。各游戏独立训练，不共享一个跨游戏模型。

CEM 优化线性策略的 **24 个权重**，以完整比赛评估参数，每步按特征点积选动作；
它不是逐动作 rollout 或 MCTS。实际 `cem_r1` 常用 16 候选、3 个环境 seed，
每候选分别对启发式和当前 best 换座，共 12 局。保留 best／默认权重，其余从
高斯分布采样，取前四分之一精英更新均值和方差。适应度为胜率（平局计半胜）；
含不完整比赛的候选批次记选择分 `-1`，这是参数筛选信号，不是伪造该局的负奖励。

## 候选残差 actor-critic

24 维特征经带符号对数缩放后，进入两层 64 单元 Tanh 编码器；actor 输出标量
残差，加在冻结启发式／CEM 先验的标准化 logits 上，残差输出层初始为零。
critic 将合法候选编码的均值和最大值拼接，经 64 单元 Tanh 层预测价值。

CPU worker 先采完整游戏，GPU 做 PPO 更新。训练随机采样动作，评测默认 argmax。
两个玩家各有自身的决策／GAE 链，反应动作和额外回合也不会把两方轨迹串起来。
变长候选的 padding 在 actor 和 critic 池化中均被 mask 排除；训练检查记录的
动作位于合法 mask。PPO 使用 clip `[0.8,1.2]`、价值 MSE 系数 0.5、优势标准化、
Adam 和梯度范数上限 1；下表示例均为每轮更新 4 epochs。

新环境族训练奖励是自然终局胜／负／平的 `+1/-1/0`，不是直接最大化官方分数。
官方分数另行保存；无官方分数的竞速游戏保持 null。

| 实际配置示例 | 起点 | γ / λ；自博弈比例 | 每批局数 / mini-batch上限 | 学习率 / entropy / prior-KL系数 |
| --- | --- | --- | --- | --- |
| 默认：`cascadia/ppo_r1` | 启发式 prior，新网络 | .995 / .95；.25 | 32 / 64 | 3e-4 / .01 / .01 |
| MC：`calico/ppo_mc_r2` | warm-start 既有网络 | 1 / 1；.50 | 32 / 64 | 2e-4 / .005 / .003 |
| prior：`cascadia/ppo_prior_r2` | CEM prior，新残差网络 | 1 / 1；.50 | 32 / 64 | 2e-4 / .005 / .003 |
| 大 batch：Dominion／El Dorado r3 | 前者 warm-start，后者 CEM prior | 1 / 1；.50 | 64 / 256 | 4e-4 / .005 / .003 |

大 batch 的实际目录为 `dominion/ppo_batch_r3`、`el_dorado/ppo_prior_batch_r3`。
**默认 PPO 也采完整对局**；`γ=λ=1` 使每个自身决策的 return 等于终局胜负，
advantage 等于终局回报减当前价值，更新仍是 PPO。非自博弈局交替安排 learner
席位，对手为冻结 prior 或默认启发式。`patchwork/ppo_free_r3` 把 prior-KL
系数设为 0，但仍保留 prior logits。

上述常规配置用 7 个 worker，大 batch 用 15 个；mini-batch 还受 65,536 个
padding 后候选位置上限约束。r1 更新上限为 512，这些变种为 1024，实际可因
预算提前停止。warm-start 保留网络及其 prior，新建优化器；恢复同一运行则
还恢复优化器／RNG，要求源码与学习参数匹配，并沿用保存的预算截止时间。

## 数据、筛选与确认

CEM 保存候选逐局结果、gate、权重分布和 RNG；PPO 保存逐批结果、指标、轮次
模型及恢复 checkpoint。只有自然完成的 PPO 游戏产生梯度目标，没有完成数据
则停止报告。评测保留 `truncated`、规则未决和错误，不宣告赢家或补终局奖励。
PPO 每批只对首局写完整公共观察／动作轨迹；CEM 预检、训练器内部确认和显式
`record` 的评测也可保存轨迹。其余比赛有逐局结果，不能当作所有轨迹／候选
特征均已归档的离线数据集。生成权重、轨迹和运行目录不进入 Git。

训练 gate 是小样本晋升：候选对当前 best 换座，全部完成且胜率至少 62.5%
才晋升。实际常规配置用 12 个 gate seed，PPO 每 16 次更新检查；大 batch
改为每 32 次更新、32 个 seed。`best` 是初始化参考或通过 gate 晋升的模型；
`last` 仅表示较晚 checkpoint，不保证更强。
配置中的 64-seed 训练器内部确认只在预算允许时执行，不改变 best，也不等于
发布前的独立确认。

开发联赛冻结模型字节、源码、配置、seed 与赛程；当前常见设置为 64 个共同
环境 seed，每对 bot 换座共 128 局。开发阶段参考对启发式胜率、非随机 peer
均值、直接互战与不完整率选择主策略及互补池。95% 区间按环境 seed 整组
bootstrap，两次换座不视为独立样本；区间没有多候选比较校正。

最后用 **512 个未参与调参的新环境 seed** 确认冻结池，每对 bot 共 1024 局；
不能根据确认成绩再反选模型。训练／gate／内部确认有独立 seed 命名空间，
发布确认另须核对开发及历史使用记录。报告完成率、完成局胜率和未知结果上下界，
不能静默删除截断或直接计为失败。

固定启发式属于训练中见过的对手类型；新 seed 确认检验新开局上的表现，不是对
未知对手的泛化测试。保留策略之间的互战也不能替代独立强搜索或人类对照。

保留池可包含 random、启发式、CEM、不同 prior／训练设置及显式 bot guard，
不要求保留每个 checkpoint，也不声称所有游戏都因 RL 提升。Harmonies 通过
`kind: harmonies_rl` 复用既有双人专用 `SelfplaySolver` 模型和加速环境，
不属于本轮新训练的 24 维候选网络；其历史结果不混计为本轮训练产出。
