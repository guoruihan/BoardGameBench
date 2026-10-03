# V1.2：四小时后台试验与限时计分

本轮规则不变，优先诊断 Harmonies；Micro/TIE 保留旧模型并做限时兼容检查。
这是第一轮有界试验，不承诺训练收敛、超过搜索或达到高手人类。

## 0.4.1 动物 RL 续轮（2026-10-04）

原 `pilot_20261004_a` 在运行 582.800 秒后因 PPO 概率一致性检查失败而停止，
并非仍在训练。旧权重、失败日志、原教师和 current 指针均保留。
下面的原始路线只描述第一轮；本次批准的改进使用
`configs/v12_animal_rl_plan.json` 和 `boardbench.v12.animal_experiment`。

本机实测发现：CUDA 12.1 / Torch 2.4.1 / b35 4090 上，即便固定 logits，
float32 概率归一化仍有超过 1e-4 的波动。不是通过放宽门槛解决，而是统一用
float64 计算采样/更新概率和旧 log-prob；网络与梯度仍是 float32。
固定批次对 CPU64 参考误差小于 1e-15，32 个真实 PPO 批次均通过原门槛。
这是已观测到的运行栈行为，不泛化为所有 PyTorch/GPU 的问题。
复现及修复证据在 `outputs/v12_checks/probability_*20261004_a/`。

续轮固定相同的 `cards_v1` 输入和 `slots240_v1 + STOP`、两层 128 MLP：

- 从 96 局旧教师的公开轨迹重新编码，不重新支付搜索成本、不读 benchmark holdout。
- 动物残局从合法动作前缀重放得到；按来源整局和布局隔离训练/诊断留出集。
  一步放置、2–3 步构造、拿牌后放动物分别记录覆盖数。只能使用不会抽牌/补资源的
  最后回合；精确枚举所有合法动作及 STOP，超节点上限则弃用，不伪造精确标签。
  BC 拟合第一动作最优值，真正 on-policy PPO 学习短轨迹总分；两者独立报告。
  样本不足或未拟合成功都明确记载，短残局成功不等于整局能力。
- 三个种子 911/912/913：冻结 BC 各 90 秒；纯 PPO 与 BC→PPO 各累积 180/600 秒。
  原生 PPO 学习率 3e-4；BC→PPO 为保守微调 1e-4，均使用 KL 0.02 提前停止本批更新。
  BC→PPO 显式校验父权重 hash/source/网络/划分/种子，只复制权重，重置 Adam 并采集
  新 on-policy 数据；不能把它伪装成纯 PPO，也不能绕过 resume 的配置一致性。
- 如果 BC 连一步放动物都拟合不了，停止放大训练，保留诊断报告。
  如果只有 PPO 拟合失败，保留它的 180 秒对照，不延长纯 PPO；单独考察教师辅助路线。
  不乘动物奖励系数，不强制放动物，不删除 STOP 或合法动作交错顺序。
- 仍使用原先已选定的 3 秒在线时限和相同开发/验证/测试划分。旧基线验证复用原始记录，
  不冒充新重复；训练阶段只由开发集选择，测试集在冻结后才运行。
  晋级要求真实总分配对门槛通过，且动物分/放置/完成卡均超过旧 PPO。
  BC→PPO 还必须在总分上优于冻结 BC，动物分不下降。整个家族与实际部署的单个候选
  分别检查；无改进或执行/恢复失败则不换 current 指针。

启动方式与原来一致，只替换 plan 和新的输出目录：

```bash
tmux new-session -d -s boardbench-v12-animal-rl 'bash -o pipefail -c "PYTHONNOUSERSITE=1 .venv-train/bin/python -u -m boardbench.v12.launch --repo /datapool/data3/storage/ruihan/code/boardgame --plan configs/v12_animal_rl_plan.json --root outputs/v12/animal_rl_UNIQUE 2>&1 | tee logs/v12/animal_rl_UNIQUE.log"'
```

这是原预算的继续，不是再加四小时。启动器读取原 allocation/result，扣掉旧失败轮
582.800 秒，并保守计入复现/隔离/验证命令的完整 timeout 上限共 350 秒（含导入启动）；
同时绝不越过原始 **2026-10-04 06:21:06 +08** 截止。过期后此配置拒绝启动，
需要新的资源授权。工程 CPU 单元测试不记作策略学习实验；真实策略诊断全部入账。
新结果在 `diagnosis/report.json`、`learning_curve.json`、`frozen_selection.json`、
`test_results.json` 和最终 `report.json`；`allocation.json` 保留旧账本摘要。

运行时代码改变后，仍须使用 current 发布对应的冻结源码启动网页。
在尚未晋级时对应的是 `outputs/v12/pilot_20261004_a/source/src`；
晋级后对应新续轮的 `source/src`。已有旧服务不会自动热加载新源码，不能跨源码关闭校验；
策略文件的自动晋级与服务器进程重启是两回事，后台实验不会杀掉用户服务。

## 预算与启动

### 已运行续轮的多 GPU 接管

用户后续授权 b35 全部可用卡并行。`scripts/parallel_v12.py` 是单独记录 hash 的
调度器，训练/推理仍导入原 run 的冻结源码，不改变模型版本或放宽续训校验。
GPU 0 有既有任务时不使用；每张健康空闲卡最多一个独立训练任务。
它会暂停原控制器继续派发，等待当前子训练/评测完整写出结果，再由原 watchdog
正常结账退出。若检查失败且尚未切换，则恢复原控制器。旧文件不改写、已完成阶段
按配置/source/权重摘要复用，未完成队列才交给新调度器。

在 b35 启动命名会话，所有路径需对应实际旧 run，输出目录必须是新的：

```bash
tmux new-session -d -s boardbench-v12-parallel 'bash -o pipefail -c "PYTHONNOUSERSITE=1 PYTHONPATH=outputs/v12/animal_rl_20261004_a/source/src .venv-train/bin/python -u scripts/parallel_v12.py launch --repo /datapool/data3/storage/ruihan/code/boardgame --prior outputs/v12/animal_rl_20261004_a --root outputs/v12/animal_rl_parallel_UNIQUE --gpus 2 3 4 5 --extra-gpu-charge 450 2>&1 | tee logs/v12/animal_rl_parallel_UNIQUE.log"'
```

每个训练阶段按“方法 × 种子”并行，依赖 BC 的 PPO 在对应 BC 完成后才启动。
同一阶段全部训练结束后再进行 CPU 限时评测，避免训练争用 CPU 改变在线分数。
仍共享原来的八个物理 CPU 核；保持相同超参数、训练秒数、三组种子、3 秒在线时限，
以及原有验证/冻结测试/自动晋级门槛，不把本次调度变更当成新的性能结果。

GPU 秒改为各卡实际分配给训练任务的时间之和（包含该子进程导入和 CUDA 初始化），
CPU-only 评测期间释放 GPU。新 `gpu_intervals.json` 记录每个任务的起止和设备，
旧轮费用累加到 `cumulative_charged_gpu_seconds`。本次仍同时保留 14,400 GPU 秒上限
和原 06:21:06 截止；不能把五卡一小时报成一 GPU 小时。外层 watchdog 独立检查
累计 GPU 时间与墙钟，异常时清理本实验的后代并保留日志。此脚本不停止任何无关任务。

本次实际设备复核发现 GPU 1 特有的数值不稳定：相同输入反复计算的 float32
前向/归一化误差波动到约 1e-3；GPU 2–5 的 100 次重复完全稳定，最大误差约 1.9e-6。
GPU 4/5 上用原 BC 权重各做三批 PPO，均通过原阈值，并得到相同更新统计与分数。
因此排除 GPU 1，不通过继续放宽阈值掩盖异常；这还不是硬件故障的确诊。
证据为 `outputs/v12_checks/device_replay_20261004_a/gpu*.json`。
五个独立设备检查按每条 90 秒 timeout 上限保守入账共 450 GPU 秒，含导入初始化；
上面命令的 `--extra-gpu-charge` 是这次已有检查的费用，不应脱离具体账本机械复用。
串行续轮在 BC→PPO 首批检查处自行失败，故本次不发送接管信号，直接复用已完成的
三个 BC、三个纯 PPO 第一阶段及其开发评测；失败批次没有 optimizer update，不作为
可续训阶段。全局历史费用为 2187.976 秒，加上述 450 秒后再累计新并行任务。

`configs/v12_plan.json` 固定一张 GPU、八个不同物理 CPU 核、最多 14,400 秒。
启动器在资源检查前开始单调时钟；源码复制、GPU 初始化、小批诊断、失败尝试、
教师数据、训练、验证、测试、导出和清理都计入。同一时间至多一个训练进程；
评测/教师最多四个 worker，所有后代继承八核 affinity 与单卡可见性。
GPU 秒按整段占用计，CPU core-seconds 按 8 × 占用秒计，**不是实测活跃利用率**。
外层 watchdog 独立于训练/评测进程，留两秒清理余量；如仍发生系统调度尾延迟，
`watchdog_result.json` 如实记录，不能声称 Linux 是硬实时系统。

在 b35 的项目目录启动一次（先按照本机规则检查 GPU 空闲，不驱逐别人的任务）：

```bash
mkdir -p logs/v12
tmux new-session -d -s boardbench-v12-pilot 'bash -o pipefail -c "PYTHONNOUSERSITE=1 .venv-train/bin/python -u -m boardbench.v12.launch --repo /datapool/data3/storage/ruihan/code/boardgame --plan configs/v12_plan.json --root outputs/v12/run_UNIQUE 2>&1 | tee logs/v12/run_UNIQUE.log"'
```

`run_UNIQUE` 必须换成新的目录名；不能重复覆盖已有试验。启动器使用非阻塞独占锁，
拒绝同时启动第二轮；GPU 1 如忙碌就停止，不自动换卡。源码复制到 `RUN/source`，
后台与子进程从该副本加载，之后的对话/编辑不会悄悄改变运行中模型的代码版本。

观察状态，无需一直挂着聊天：

```bash
.venv/bin/python -m json.tool outputs/v12/latest_status.json
tail -30 logs/v12/run_UNIQUE.log
```

每个阶段日志在 `RUN/logs/`，训练逐批日志在 `RUN/training/*/training.jsonl`。
最终写 `RUN/report.json`；异常/超时保留部分结果并在 status 标明停止。
自动收尾更新文件与策略入口；不依赖聊天继续执行，也不承诺主动发送聊天消息。

## 预注册路线与门槛

1. GPU 实际 forward/backward/step、旧权重显式导入、开发集测旧 PPO 时延。
   主预算取 `ceil(10 × p95)`，限制在 1–8 秒；仅使用开发种子，不用测试成绩挑时限。
2. 对随机、启发式、旧搜索、旧 PPO 和 solver 自己分配时间的搜索进行共同限时验证。
3. 96 个训练局的搜索教师，以及有合法动物动作的无随机未来终局两步精确 oracle。
   小数据拟合失败或样本不足，不扩大训练；训练 gap、独立状态 gap 分开报告。
   小残局不是完整游戏的最优解；holdout 是状态划分，不宣称独立玩家/游戏分布。
4. 五个 60 秒单种子诊断，依次拆开动作头压缩、槽位输入与卡牌语义。
   这些短跑用于定位问题，不据此声称哪一种普遍更优。
5. 主 PPO/搜索蒸馏各跑 911/912/913 三个种子。PPO 累积目标 300/900/2100 秒，
   BC 为 60/180/420 秒，阶段继续恢复 Adam、全部 RNG、计数器和数据游标。
   两阶段提升不足一分或最终评测余量不足时停止扩展；保留各阶段和未训练对照。
6. 开发集挑每个种子的阶段；独立验证集做替换门槛；冻结后一次性运行新测试集。
   三个新模型在同一验证种子上的平均分，要对每个非随机参照的配对均值差
   `mean ± 1.96 × sample_SD / sqrt(n)` 下界大于零，且没有策略失败，才可晋级。
   这是条件于固定产物、未经多重比较校正的**部署选择门槛**，不是显著性研究结论。
   测试成绩不反过来改变选择；测试执行异常可否决发布。
   固定配置的新限时搜索也可晋级：不伪装成三个训练重复，只对其余非随机参照做相同
   配对验证和执行/恢复检查。纯 RL 并不必须成为最终默认方法。

训练集为 8000000 起 65536 局，教师为 8100000 起 96 局；开发 9000100 起 16 局，
验证 9000000 起 32 局，测试 10000000 起 48 局。检查互斥以及与旧来源全部历史划分互斥。
不同规则任务可复用相同数字种子，不把不同游戏的成绩混为一个配对统计量。

主训练是完整局 score-delta PPO warm start 和搜索动作监督，不声称是时间条件最优控制。
编码接口支持真实时间特征，但 warm start 的时间列始终为零，避免部署输入未训练列。
可选 `time_conditioning=true` 必须配 PPO `episode_seconds`，使用真实截止、截止零 bootstrap；
此模式恢复优化器/RNG，但真实墙钟本来不能保证恢复后逐局轨迹与不停机运行按位一致。

## 计时语义与证据

`boardbench.v12.timed` 的父进程独占真实引擎，子进程只接收公开观测；ready 后才启动整局
时钟并发送首份局面，不给真实 seed/未来序列。启动/加载不算主在线分数时限，但算总预算。
所有 solver hooks 和通信都在可中断边界内；不在可信父进程执行策略函数。
动作在截止前完整收到才可提交；已接受的原子引擎步可跨过截止，单独记录尾部时间。
截止/主动 STOP 按最后已提交合法局面得分；程序错误/非法动作记零，不给免费兜底。
所有尝试进入均分，单列自然完成率、失败率、地形/动物、拿牌/放动物/完成牌等。
终局 hooks 是最多 50ms 的 best-effort 清理，不改变分数；不承诺回调完成。
模拟步计数是已返回响应里的可观测计数，进程被杀时的在途工作由墙钟预算覆盖，不能将
该计数说成全部计算量。该边界针对可信本地策略，不是恶意 Python 沙箱。

`final_observation.terminated` 表示规则终局，`episode_finished` 表示评测结束，两者不同。
每局独立策略对象，不共享上一测试局学到的状态。训练和验证保留源码、权重及配置摘要。
普通 PPO 的 batch 边界恢复有不停机/恢复后模型与 Adam 状态完全相同的单元测试。

## 自动策略更新与网页

```bash
.venv-train/bin/python -m boardbench serve --config configs/harmonies_v12_play.json --port 8765
```

新配置在**开新局**时重读 `outputs/v12/current.json`；不会在当前人类局中途换模型。
网页人工对局 1200 秒，支持剩余时间、STOP、自然/截止/主动结束及存取；恢复过期存档
不会重新给时间。页面挑战是未限时演示，**不是正式 headless 评测成绩**。
已有旧版服务不会热更新 Python 代码；需要用户自行重启到上述入口，不能擅自终止旧服务。

首次导入将 V1.1 权重/配置重新导出为本轮源码绑定 checkpoint，保留旧来源摘要和规则。
这不是绕过原 checkpoint 的源码检查；原 `outputs/v11` 和交付 ZIP 均不改写。
晋级前验证所有 manifest/权重/hash，并在独立发布进程真实加载所有策略；通过后原子替换
current 指针。旧指针保存在 `RELEASE/publication.json` 的 previous 字段，旧目录不删除。
发布被中断时最多留下未引用的新目录，不会出现半个 current 文件。

后续如修改运行时代码，应使用 `RUN/source` 对应源码启动页面，或再次显式重新导出；
源码摘要不匹配会拒绝加载，不能关闭安全检查。使用 snapshot 的示例：

```bash
PYTHONPATH=outputs/v12/run_UNIQUE/source/src .venv-train/bin/python -m boardbench serve --config configs/harmonies_v12_play.json
```

## 人类参考

本轮搜索结果记录在 `docs/v12_human_references.json`。未找到可核实且与本项目
Harmonies 单人 A 面、无精灵、有限资源/弃牌/计时协议一致的高手成绩分布。
公开蓝面自报 165 分、电脑构造上界等只作背景，不作为目标或发布门槛。
“没找到”是有限范围检索结论，不是断言不存在记录；人类水平目标暂缺。
