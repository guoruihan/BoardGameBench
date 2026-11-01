# V1.7.3：八任务搜索示范与完整局面 RL

本轮用户授权三小时自主探索，窗口为 2026-11-01 00:18:13 至 03:18:13 UTC。
训练在结束前预留确认评估时间。覆盖 Century A New World、Century Spice Road、Clever、
Patchwork、Splendor、Stone Age、Harmonies、Toy Battle。游戏、公开信息权限和官方计分不变。

## 方法与对照

- 复用 V1.7.2 完整玩家观察／动作树、静态公开组件信息和等价模拟内核。
- 新 actor-critic 同时读取完整局面、动作和旧候选特征；冻结旧 RL 作为加性策略先验。
  初始残差为零，以旧策略为行动起点；近似并列动作的 FP32 边界见下文。
- 使用有效的高预算搜索轨迹做行为模仿、终局价值和计分分项初始化。搜索有至少四次
  完整候选评估时，混合实际专家动作与候选软标签；低样本动作降低监督权重。
- 完整对局 PPO 使用同一玩家的决策链、gamma=1、实际行为概率和裁剪目标。
  专家数据只进入监督损失，不冒充 on-policy 样本；不完整对局保存但不产生终局学习标签。
- 首轮比较胜负目标与胜负／终局分差混合目标，二者都有计分分项辅助头与示范回放。
  Toy Battle 的勋章不是唯一胜负条件，因此两个配置都保留真实胜负目标。
- 每个任务独立策略；不要求全部任务使用同一个最优超参数。后续变体由实际结果决定。

旧通用策略可以提供连续 logits；Harmonies 旧适配器目前只提供选中动作，因此采用
该动作偏好作为冻结先验，并非旧策略的完整概率分布。所有候选均通过实际对局比较；
统一训练流程不意味着这些先验具有相同的探索分布。

首轮快速评估后，针对退化／停滞任务增加选择性监督：教师保留旧动作或候选样本不足时，
保留旧策略的完整动作概率；有足够样本支持改动作时才使用硬动作标签。同时测试更长的
模仿阶段和 PPO 的旧策略 KL 约束。首轮运行从原始源码快照继续，配置和结果不被覆盖。
Toy Battle 的第二个首轮分支增加熵探索并减小示范正则，仍使用纯胜负奖励。

方法参考：[PPO](https://arxiv.org/abs/1707.06347) 的策略更新、
[Expert Iteration](https://arxiv.org/abs/1705.08439) 的搜索与快速策略分工。
[DAgger](https://proceedings.mlr.press/v15/ross11a.html) 提醒我们还需检查学生实际局面的分布偏移；
本轮初始教师补数是新专家对局，不能将它称为已经实现 DAgger。

后续学生局面标注使用单独的 `guided_search_evaluation --config ...` label 模式：
学生实际执行动作，教师仅给当前局面提供另一份动作标签与候选评估。两种动作分开保存；
终局回报仍属于实际执行的学生轨迹。只有将这批数据纳入下一轮学习，才构成一次学生分布补数。
同一模块的 match 模式用于冻结学生与冻结搜索的短预算对照，和训练进程完全分离。

## 数据和评估

上一轮四个开发环境产生的专家轨迹现在属于训练数据，不能再用于独立确认。
后台另外生成 128 个高预算教师对局，模型训练不等待其完成；新数据编码后加入监督回放。
所有校准尺度从训练轨迹固定，评估分数不能反过来修改标签尺度。

训练模型按版本输出不可变快照。独立 CPU 评估进程对固定旧 RL 进行新环境、双座位比较，
同时记录官方总分、计分分项、完整率和实际时间。快速开发选择与最终确认使用不同种子区间；
正式 test 不参与任何调参。较短预算的搜索对照作为异步诊断，不能代替高预算教师的强度证据。

训练和评估分别记录配置、源码、模型哈希、轨迹、损失和成本。内部编码缓存使用本机产生的
压缩 pickle，不接受外部不可信文件；公开策略接口只接收观察和合法动作。
词表随模型保存，训练数据也保留同一份符号表。旧模型可使用经过原始源码导出的
`vocabulary.json` 侧文件恢复原编号；严格校验词表哈希，不通过忽略版本检查加载。
默认词表排除 Python 导入路径等元信息，源码目录迁移不再改变编码。

## 资源与恢复

每台机器的自身分配不超过物理 CPU 核的一半，a 系机器同时不超过 128 核。
训练使用空闲 GPU；只在确认容量后叠加本项目任务。b35 用于 CPU 教师搜索。
每个训练进程及采样子进程限制在分配的 CPU affinity 内，PyTorch／BLAS 单线程。
CUDA 必须通过真实运算检查，不允许请求 GPU 后静默降级。

本地实验根为 `outputs/v173/search_guided_rl_20261101/`，权重、轨迹、机器配置不进入 Git。
训练恢复使用原配置与 `checkpoint.pt`，校验配置哈希并恢复优化器／随机数状态；不得以恢复名义
延长原截止时间。已有模型始终保留，开发集优胜不自动替换已发布 baseline。

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-train/bin/python -m pytest -q tests/families/test_guided_training.py

# 在已分配资源的远端运行，config 指向对应任务的不可变配置。
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-train/bin/python -m boardbench.families.guided_training --config /path/to/config.json
```

### 第二轮的先验与数据

新运行可设置 `anchor_scale`，对示范里的冻结先验和实际行动使用相同比例；零初始化的策略修正在实数计算下保持原 logit 排序，FP32 近似并列例外见下文。`warm_start_bc=true` 可以从明确指定的旧快照重新进行示范训练，词表、基础模型和目标标定必须一致。小开发面板的最优值仅用于发现候选；最终采用 32 个开发环境双座次选优，再用未参与选择的 256 个环境双座次确认。

### 可恢复的异步评估

运行 `python -m boardbench.families.guided_evaluation --root RUN_ROOT --workers 32 --deadline-unix END --confirm-at FREEZE --development-split wide --development-seeds 32 --development-seed-start 173050000 --confirmation-split wide_confirmation --confirmation-seed-start 173060000 --confirm-seeds 256 --poll-seconds 600`。开发与确认种子不能重叠，已有输出不能静默更改环境数；确认前写入冻结的 `wide_confirmation_selection.json`。它不阻塞训练。8 环境的快速面板仅用于发现候选，不能作为提升证据。

浮点注意：缩放先验保留其 logit 排序，但 FP32 `log_softmax` 在接近并列的动作上可能引入相同数值，因此不能声称每条完整轨迹严格保持原模型。本轮检查发现 Clever/New World 有约 3e-8 至 6e-8 的差值被舍入；原始初始模型与缩放初始模型均保留，所有胜率来自实际配对对局。

### 学生局面纠正数据

`guided_search_evaluation` 的 `mode=label` 默认执行学生动作，同时保存教师建议；可显式设置 `teacher_execute_probability`，用独立策略随机数选择是否执行教师动作。每条记录保留实际行动来源，终局价值标签来自实际混合轨迹，而不是假设教师全部执行后的回报。带动作历史保护器的游戏暂不支持该混合模式，以免保护器记忆与实际执行不一致。示范只进入监督损失，不冒充 PPO 的 on-policy 样本。

### 保守的策略修正强度

`python -m boardbench.families.guided_variants --game GAME --source MODEL --output NEW_MODEL --scale 0.5` 将两个 actor 输出层同时乘以给定比例，因此实际 logits 为“原冻结先验 + 比例 × 已学修正”；critic 与辅助头保持不变。它产生独立文件和父模型哈希，用于检验过大的策略修正是否导致退化，不声称是额外训练或对局搜索。

可分别配置 `bc_learning_rate` 与 `ppo_learning_rate`，让示范拟合与后续在线改进采用不同步长；未配置时均沿用 `learning_rate`。本轮后续纠正数据试验采用较快的示范拟合与更小的 PPO 步长，检验在线更新是否过度改变已学策略。

### 采样温度与 PPO 一致性

同模型、同环境的贪心/随机行动对照显示部分模型的随机行动远弱于贪心策略。`sampling_temperature` 只作用于随机 rollout：动作采样、旧 log-prob、PPO 新 log-prob、熵和先验 KL 都使用同一温度；每条样本保存温度用于审计。贪心评估不受该参数影响，示范监督也保持原目标。默认 1 兼容此前模型；低温度仅是待验证的探索方案。

低温可能让有限 log-prob 对应的 FP32 概率下溢为零。先验 KL 直接使用 log-prob 计算，
仅排除显式 padding 动作，并检查先验与策略的合法支持集一致；不能把数值下溢当成
真正的零概率支持。非有限损失检查仍保留。实际失败的 Patchwork/Toy Battle 批次、
修复前后诊断及恢复分支均单独保存，不覆盖失败证据。

在线阶段还可设置 `replay_value_coef=0`、`replay_auxiliary_coef=0`，仅回放专家动作，
让 critic 与分项辅助头直接学习当前策略实际轨迹的终局标签。默认值保留原方法；
这是检验不同策略的价值目标是否冲突的诊断，不预设它一定改善成绩。

本轮已完成，独立确认与资源口径见 [结果报告](Search_Guided_RL_Results_zh.md)。
