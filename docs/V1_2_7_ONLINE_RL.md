# Harmonies V1.2.7：已有知识初始化后的在线 PPO

2026-10-20。目标是让**模型自身**通过真实对局回报持续改进，不再把替换搜索器动作/候选集当成 RL 主线。包版本0.4.11。

## 方法与判断标准

1. 复用V126已有模仿模型和训练数据，不重跑昂贵教师。完整记录父模型、数据和源码哈希。
2. 显式重置未训练的价值头；固定策略和共享表示，让它按当前随机策略玩256局，仅拟合价值头的完整轨迹回报。不能把教师回报误当成当前策略的无偏价值。
3. 新鲜on-policy完整对局 -> GAE -> PPO actor/critic更新。每次128局；gamma=1、lambda=.95、PPO clip=.2、lr=1e-4、4epochs、minibatch1024、entropy=.005、targetKL=.02。奖励是原始总分增量/150，整局加总等于最终总分/150，不另加动物奖励。
4. 冻结初始策略作为逐环境对照；主要指标是独立模型的贪心完整对局得分、动物分和完成卡数。随机训练分数不能与贪心开发分数混用。

复用`v121.collector.EnvironmentPool`、V12公开观测编码和`CompactSolver`。策略输出240个动作，排除STOP；只能自然终局，不改游戏规则。1346维输入、256×256 MLP与父模型一致。为沿用历史数值修复，模型/梯度/Adam和概率运算采用FP64，禁用TF32；更新前行为概率重算误差必须<=1e-4，失败不放宽门槛。

PPO依据[原始论文](https://arxiv.org/abs/1707.06347)。模仿初始化及逐渐减小示范影响的辅助组参考[DAPG论文](https://vikashplus.github.io/Projects/RLwithDemo/dapg.pdf)，这里只是PPO+衰减BC消融，不声称复现DAPG。

## 第一阶段：四组各两个训练种子

|组|初始化|在线目标|
|---|---|---|
|bc_a/b|bc256_a/b|PPO|
|soft_a/b|q256_a/b（搜索软标签模仿，不是Q-learning）|PPO|
|anchor_a/b|与bc组相同父模型|PPO + BC，系数.05在前50万PPO动作线性降至0|
|random_a/b|随机、同256宽架构|PPO|

训练种子12701/12702，四组使用相同环境顺序与采样预算。不同组行为不同，轨迹长度也不同；按动作比较时同时报告实际局数和动作数，不假称转移数据完全相同。每个检查点保存优化器、随机状态、计数、父版本；每个完整更新后更新`latest.pt`，保留`initial.pt`、`best.pt`、各里程碑和`final.pt`。

固定32个development环境画曲线，里程碑5万/10万/25万/50万/100万/200万动作；另外64个development环境只用于该轮结束后的配对核对，分别运行初始、曲线选出的best和latest模型。不是正式测试或永久未见数据。train中的模仿验证/选择环境不进入在线训练；formal test/confirmation完全不运行。

学习是否有效需要实际分数证据：至少相对冻结初始模型有配对收益，同时比较同动作预算的random PPO。初始化不同、只有两个训练种子，不把单个最好checkpoint的小幅领先称为可靠方法优势。

## 资源与运行

- a218独立8进程，一张A100一个实验，无DDP。每个进程最多15个物理CPU核（14环境worker＋learner），外层1核，合计121核<=128。
- 持有现有节点CPU互斥锁，启动检查GPU空闲和节点负载；b35不使用GPU。
- 每任务首阶段最多200万PPO动作目标或4小时（包含初始化、评估、保存）。完整批次会显式超出动作目标；时间边界允许当前批次收尾，外层最多90秒清理宽限，实际占用全部计账。任务结束保存并停止，不自动扩无限预算。
- 旧V126教师/模仿费用只作历史成本保留：112.499592 CPU核时、0.123467 GPU小时。不会重算进新任务；新预检用明确标注的保守额度2 CPU核时＋0.5 GPU小时覆盖，实际预检receipt另存，不能重复相加。
- 采样本质仍有CPU成本；GPU负责批量策略推理和优化，利用率以实测为准，不承诺八卡满载。

在a218项目根目录启动（SSH到192.168.2.11，环境用`.venv-train/bin/python`）：

```bash
mkdir -p logs/v127
tmux new-session -d -s boardbench-v127-ppo "bash -o pipefail -c 'cd /datapool/data3/storage/ruihan/code/boardgame && PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv-train/bin/python -u -m boardbench.v127.launch launch --repo /datapool/data3/storage/ruihan/code/boardgame --root outputs/v127/ppo_a218_20261020_a --config configs/v127_a218_online_ppo.json 2>&1 | tee logs/v127/ppo_a218_20261020_a.log'"
```

恢复时先核对原PID已经退出、资源空闲，使用新tmux/log名字，将`launch`子命令换成`resume`，同一个root，不再传`--config`。恢复入口使用root内冻结源码，不用修改后的工作树；原任务的时间上限不会自动重置。

运行目录内容：

- `REPORT.md` / `results.json`：每组进度与模型得分。
- `curves.svg` / `curves.json`：同一批32开发局面的纯模型学习曲线。
- `trials/<name>/training.jsonl`：真实采样、每局成绩、损失、KL、梯度、吞吐、PPO动作数。
- `trials/<name>/evaluations/*.json`：逐局评估；`final_comparison.json`记录新面板上的best-vs-initial配对差。
- `trials/<name>/*.pt` / 相应`.json`：可恢复检查点及哈希。
- `launch.json` / `config.json` / `jobs.json` / `source/`：冻结配置和源码。
- `state.json`：所有attempt、GPU和CPU分配、含失败的累计占用记账。

## 旧任务停机与验证

已按用户授权停止`outputs/v126/learning_a218_20261020_a`；只向验证过的外层PID6189发送SIGTERM，原watchdog收拢子进程。之后没有匹配的V126训练/实验进程，DAgger32局标记中断，不作为完整训练数据导入。已有教师144局、八模型、完整pilot144局和逐步日志全部保留。停机receipt在旧root下`user_stop_20261020.json`；原状态`stopped_with_error`是有意停止的KeyboardInterrupt，不解释成算法失败。

预检入口：`python -m boardbench.v127.preflight --repo <repo> --out <new project output>`。它在8张卡上分别做真实采样、PPO更新和恢复检查，再用实际模仿检查点运行128局/14worker的完整小流程。CUDA测试需要显式`BOARDBENCH_V127_CUDA=1`，常规测试不会占用GPU。

正式启动、最终测试数量和首批更新证据见任务记录及本文件后续启动记录；预检成绩只检查链路，不用作方法选择。

最终预检：`outputs/v127/preflight_a218_20261020_b`，源码哈希`eb54d8f82eec41ca4ae323417541c45963a2e1c2735c141e3e17aee6e3f71563`。8张卡均通过PPO采样/更新/恢复；实际BC父模型完成512局，其中13345个PPO动作、56次优化器更新，包含价值头预热、冻结初始对照、曲线与新面板配对。孤立128局批次约1.1–1.5秒，仅用于吞吐预检，不推断八实验并行时仍有同样速度。

## 已验证启动记录

- 2026-10-20 07:57:42 CST启动`outputs/v127/ppo_a218_20261020_a`，tmux `boardbench-v127-ppo`，外层PID43799；冻结提交`b5f6de43e4c825c44d7993a088367b9d38789122`，源码哈希同上。八任务均实际进入PPO，持续更新actor/critic；不是仅预留显卡。
- 全量回归338passed/11显式CUDA skips，277.56秒；V127 CUDA测试已另外逐卡通过。之后只增强预检工具的节点锁/空闲检查，追加测试：V12712passed/1CUDA skip；该工具改动不会热改运行中的冻结训练源码。
-08:00:27 CST读数：各任务约72万–90万PPO动作，最近30批每批约1.01–1.31秒。行为概率误差约1e-14量级，actor/critic均有非零更新。主机进程亲和性并集121CPU（0–119及127），包含8 learner、112worker、资源跟踪和外层，不等于129个进程各占一核。
- 此窗口GPU约1%–20%、显存1.1–1.4GB/卡；5秒CPU实际约46.08核忙。每批environment+wait约54%–62%、model+transfer约12%–14%、update约17%–22%。这是阶段计时，不是函数级热点证明。
- 用户要求保持本轮不动；下一阶段可评估GPU打包、减少同步/重复计分、固定batch下worker数对照。不能为占用率修改当前实验参数。
- 跨机读取正在追加的JSONL曾出现缓存空洞视图；在写入节点a218直接检查八个日志均0 NUL、0坏行。监控优先读取原子JSON快照；详细训练日志通过a218读取，不把本机并发读取异常直接当成日志损坏。
