# Harmonies V133：连续纯模型强化学习

实现版本0.4.18。目标是在同一个成熟策略上持续训练，检验独立critic和自身经验回放能否超过普通PPO。不新增搜索、不读取专家动作损失、不从专家中途局面接手。保留旧搜索与专家档案，但它们不参与本轮新交互或优化目标。

## 固定实验

三组各两个训练随机种子13301/13302，每条**连续新增20,000,000个真实动作**。起点固定为V132第8轮`continue_winner_a/final.pt`，该父模型在上次确认前已经指定，不按确认512重新选父模型。

|组|相对共同基线的改动|
|---|---|
|ppo|普通连续PPO，不再添加专家BC|
|separate_critic|复制父模型的主体作为独立critic主体；actor与critic起始输出一致，原有Adam状态保留，仅新增critic主体的Adam矩从零开始|
|self_imitation|共享主体PPO，加小权重自身经验回放辅助；不修改PPO当前批次概率比|

独立critic增加了价值网络参数量与计算量；这轮可以检验该整体改动是否有效，不能仅凭收益就完全分离“梯度不干扰”与“新增价值容量”的贡献。

共同设置：FP64、1346输入/256×256/240动作、合法动作mask、无STOP；128整局一批；gamma=1、GAE lambda=.95、学习率3e-5、最多4epochs、minibatch1024、KL保护.02、entropy系数.005。每个任务6个CPU环境worker，GPU批量推理。整局自然结束后才更新参数。

不回退训练状态：每条继续自己的latest参数、优化器和随机状态；best仅作为单独存档，不用开发峰值覆盖下一批的起点。增加1M/5M/10M/15M/20M动作时评测。固定20M终点是主要比较，不能仅报告某组曲线最大值。

## 回放与critic诊断

回放保存最近8个完整采样批次，包含全部普通、失败和高分轨迹中的状态—动作；按transition均匀采样256条，不按终局得分过滤。目标为该学生实际走到自然终局的剩余真实回报，不是搜索Q或专家回报。辅助权重.02：

```text
positive = max(actual_remaining_return - V(s), 0)
actor_aux = -mean(log pi(a|s) * stop_gradient(min(positive, 1)))
value_aux = 0.5 * mean(positive^2)
```

这是带权重上限的SIL式辅助，不声称逐项复现某篇论文。旧经验不进入on-policy PPO的概率比或优势数组。策略之外的价值辅助仍使用真实回报；随机未来造成的幸运轨迹是该方法的风险，必须以正常开局、多个训练种子和独立环境验证。

完整回放批次以不可变压缩NPZ写到各任务`replay/`，checkpoint只保存活动批次的路径、SHA与回放随机状态，避免每步重复重写整个缓冲区。归档块不随FIFO驱逐删除，便于恢复与后续离线研究。回放可能产生数GB新数据，全部放在data3项目输出目录。另保留每批固定前四局的seed、动作、行为log probability、奖励轨迹，不按成绩挑选。

新增`mc_value_mse`与`mc_explained_variance`是对**实际剩余回报**的诊断；旧`explained_variance`仍是对GAE构造目标的拟合，二者不能混称。历史通用日志字段`bc_loss/bc_weight`在回放组承载辅助值，显式`auxiliary_kind=own_replay_SIL`及`expert_bc_weight=0`用于消歧。

## 数据与评测隔离

- train pool与父模型逐项完全相同，原示范留出与排除环境仍不入训练。
- 新生成512局开发曲线、512局开发审计和512局最终确认，按原种子范围无放回抽样。
- 新面板与原train/development/confirmation/test/historical，以及刚完成的fresh512确认面板均不相交。
- 原split文件、原确认成绩和正式test不改动，不用于本轮选择。
- 所有评测是无搜索贪心策略，从普通开局到自然终局，按128局分批。
- 六条任务全部完成20M时，先冻结全部六个final checkpoint，再与共同父模型做一次新的confirmation512。报告两个种子分别成绩与方法均值；环境配对区间不等于对训练随机种子的充分置信保证。

## 资源、故障与恢复

使用a226前三张已授权GPU，两任务/卡；六任务各7物理CPU核，加控制器共43核，不超过该机器128物理核的一半。第四张卡无本轮任务，不创建无意义试验占卡。GPU4-7属于其他任务，禁止抢占。

每任务上限7200秒，达到20M或时间上限时保存退出。资源忙时最多等待600秒；新启动的外层保护窗口8400秒，只作为本轮任务与收尾的安全上限，**不续用上次10小时授权**。训练、预检失败、资源等待与最终确认分别记账。资源门禁只重试资源问题，不掩盖数值错误。确认需全部任务完成匹配交互量；不完整时保留checkpoint并明确报告。

入口与输出：

```bash
ROOT=/datapool/data3/storage/ruihan/code/boardgame
PYTHONPATH=src PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv-train/bin/python -m boardbench.v133.campaign \
  prepare --repo "$ROOT" --root "$ROOT/outputs/v133/continuous_a226_20261021_a" \
  --config configs/v133_continuous_a226.json
```

随后在a226运行`boardbench.v133.preflight --repo ... --root ...`。只有真实CUDA、初始策略对齐和完整更新恢复检查通过，且source_id一致，才运行`boardbench.v133.campaign launch --repo ... --root ...`。launch与resume放入命名tmux，并tee独立日志。

`resume`只在原外层截止之前恢复同一源码、配置和优化状态；不会把失败任务换成新实验，也不会延长时间。`guard.json`、`state.json`、各任务`progress.json`与`training.jsonl`提供活性证据；`curves.svg`与`curves.json`按**新增动作**画曲线；最终证据为`finalization.json`、`confirmation.json`和`completion.json`。完整复现还需要保留各checkpoint引用的回放NPZ块。

## 实际启动前验证

相关回归最终48passed、5个显式GPU跳过；另在授权a226实际运行全部三组CUDA断点恢复检查，3passed。三个已选GPU均完成FP64矩阵运算。三个成熟父模型分支在同8局上的初始策略输出完全一致，更新前概率比最大误差约2.49e-14，actor真实发生更新，保存后下一批轨迹和权重恢复一致。

这8局仅是预检，不作为成绩结论。预检真实MC explained variance约.8887，GAE目标指标约.9774，说明两种指标确实不同。最终源码对SIGTERM/最终评测子进程清理也有专门回归。两次版本匹配的CUDA预检和全部本地回归保留成本记录；本地回归采用270CPU核秒保守占用上界。

## 正式启动记录

- 实现提交：`ade1001`；冻结源码ID：`82ce044ba2e5fb98b9c974eb9d1239944e90e6e2e289afd61ced58307462613e`。
- a226命名会话：`bb133-continuous-a226-20261021-a`；日志：`logs/v133_continuous_a226_20261021_a.log`。
- 启动：2026-10-21 01:19:53 CST；外层硬截止：03:39:53 CST。每任务自身仍有7200秒上限，达到20M会提前完成。
- 六组初始development512均分均为107.044921875，证明正式评测起点对齐；这不是训练收益。
- 启动后约一分钟，六条已分别完成约19.7万至31.9万新增动作，actor变化非零，行为概率校验误差约2e-14至3.4e-14，回放辅助loss非零，日志与计数持续推进。
- GPU0-2各两进程，首次资源快照约1.8–2.1GiB/card、利用率16%–43%；GPU3空闲，GPU4-7原任务未动。不宣称GPU已经跑满。
- 早期常规批次约1.0–1.2秒，回放组约1.8秒；只用于估计时间，正式效率比较另包含保存、评测和全部资源费用。
- 输出：`outputs/v133/continuous_a226_20261021_a/`，其中`curves.svg`与`REPORT.md`自动更新。结果尚在产生，不能把启动检查描述成完成实验。
