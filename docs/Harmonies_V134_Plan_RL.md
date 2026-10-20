# Harmonies V134：布局意图与分层 RL

实现版本0.4.19。问题是显式空间目标和跨动作承诺能否改善成熟纯模型策略。
V133保持独立运行，不中断、不覆盖；本轮起点预先指定为V133 `ppo_a/final.pt`，
不是根据它的最终确认成绩挑选，也不声称它是V133最强模型。

## 四个方案

|方案|给worker的计划|更新时机|谁选计划|
|---|---|---|---|
|capacity|全零|不使用计划|容量匹配对照|
|reactive|一个主目标＋最多一个兼容次目标|每个原子动作|公开启发式|
|persistent|同上|最多两回合，完成或失效时提前结束|公开启发式|
|manager|同上|同persistent|独立小网络，高层PPO|

四组worker均加同结构残差adapter，以原主体表示和819维计划向量为输入，
输出240动作logit修正和一个value修正；末层初始化为零，因此起点输出与父模型一致。
capacity的计划列恒零，额外网络仍可从原状态表示学习，避免把普通扩容直接归因于计划。
manager另有独立参数和更新费用；它与persistent的差别不能解释为严格等参数量比较。

计划包含目标动物、锚点、具体目标堆叠、缺失资源、资源成本、是否需先拿牌和计划年龄。
候选0永远是自由行动；其余最多7个候选，交错保留不同动物与高分候选。
次目标要求与主目标同时兼容、锚点不同，并检查合并资源需求。
资源供给是公开信息的乐观上界，不保证未来一定能选到。
原始合法动作mask不受计划约束，worker可以偏离甚至放弃该计划。

首版不表示“先完成A再改地形做B”的完整依赖图，不是全局最优布局器。
候选几何来自现有habitat方法，向量化实现对照原函数检验。
agent的作用是设计并检验这些策略抽象；本轮不付费调用LLM，也不把LLM评价当奖励。
有价值后再探索更丰富的顺序模板和轨迹目标标注，不同时堆叠全部改动。

## 学习和可恢复性

保持FP64、1346维公开状态、240合法动作、完整自然终局、gamma1和原计分。
worker继续原PPO和已有Adam状态，不新增搜索、专家BC或人工目标奖励。
上层在每次新计划选择时记录候选集合、mask、选择概率及状态value；
高层transition横跨若干真实动作，其reward是这些动作原始reward之和。
高层GAE按option边界计算、终局bootstrap为0；不能把每个原子动作复制成一次上层选择。
上层有单独采样RNG，避免额外抽样直接扰动worker随机数流。

每个checkpoint包含worker、manager、全部optimizer/RNG和累计计数。
阶段晋级沿用自己的完整末态，不重置到峰值。best只存档。
只在完整更新边界保存；恢复必须匹配源码、配置、设备和checksum。
每批固定前4局保存动作、奖励、行为logp、逐步计划向量；manager还保存选择边界。
这些是自采样数据，不伪称专家真实意图。

## 自主实验协议

- a226 GPU0-3，每卡两个独立任务，不做八卡单实验。
- 第一阶段8任务×7CPU＋controller=57物理核，低于128核机器的一半。
- 四组各两个种子13401/13402，每条新增4M真实动作，任务上限3600秒。
- 在同一个成熟父模型上启动；公共新面板：development512、audit512、confirmation512。
  保持原train池及排除清单，正式test不变；新种子范围与旧fresh面板分开。
- 第一阶段以固定终点的audit512均分选出最佳非对照方案；不看曲线峰值和confirmation。
  它只是探索性晋级，不据此宣称显著胜出。
- 第二阶段保留capacity与晋级方案，各两个种子，继续8M动作；每任务上限4800秒。
  只用两张GPU、29CPU；未晋级策略和轨迹全部保留。
- 全部完成后冻结四个12M终点，加原父模型，统一confirmation512。
  环境配对区间只条件于这两个训练种子，不表示训练随机性的充分置信区间。
- 对晋级方案在audit512做移除计划实验。该干预有分布偏移风险，结合真实总分解释。
  确认和干预结果不再用于本轮调参。

外层硬窗口10800秒，包含两阶段及收尾；不完整则保存停止、不自动延期。
门禁忙时最多等待600秒，不抢占已有任务。所有预检尝试、训练、等待与最终确认记账。
正式目标以真实动作数为主，实际耗时和物理GPU区间另报，不能宣称等算力。

## 入口和证据

```bash
REPO=/datapool/data3/storage/ruihan/code/boardgame
RUN=$REPO/outputs/v134/plans_a226_20261021_a
PYTHONPATH=$REPO/src PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 $REPO/.venv-train/bin/python \
  -m boardbench.v134.campaign prepare --repo "$REPO" --root "$RUN" \
  --config "$REPO/configs/v134_plans_a226.json"
```

授权a226上先运行`boardbench.v134.preflight --repo "$REPO" --root "$RUN"`。
通过真实CUDA、成熟策略输出、下一批完整更新恢复和吞吐检查后，
在命名tmux中执行`boardbench.v134.campaign launch --repo "$REPO" --root "$RUN"`，tee独立日志。
`resume`只在原硬截止前恢复，不延长窗口；使用run/source冻结源码。

`stage1/`、`stage2/`分别保存config/jobs/state/results/curves.svg和每条trial。
`promotion.json`记录晋级依据，`finalization.json`在确认前冻结端点，
`confirmation.json`报告真实分数，`ablations/`记录计划干预，
`completion.json`及`guard_result.json`证明是否正常完成。

实现和预检成功不等于研究收益。本轮必须允许结论为“计划没有可靠收益”。

## 验证与实际启动

- 实现提交`b4be804`；冻结源码ID
  `2939f7d45303b7053b8376b89dd265f1adfe24be4f39be8bded0e907685cd413`。
- 本地相关回归46passed、8显式CUDA跳过；a226另外执行4个CUDA恢复测试，全部通过。
  四个成熟模型分支同8局初始108.625，更新前概率误差约3.20e-14，
  下一批实际轨迹及worker/manager权重恢复一致。
- 单任务预检128局/批：capacity2090、reactive2067、persistent3341、
  manager3070真实动作/秒。不同模式开销不同，不能把相同动作数解释为相同算力。
- 等待V133自然完成497秒后开始预检；V133的completion/guard均正常结束，
  没有中断它的训练或最终确认。预检121.60秒，所有等待和回归计入账本。
- 正式启动2026-10-21 **02:38:09 CST**；原始硬截止 **05:38:09 CST**。
  会话`bb134-plans-a226-20261021-a`，日志`logs/v134_plans_a226_20261021_a.log`。
- 正式八组initial development512均为108.759765625。
  02:39时八组均已完成多批更新，新增约5.7万至12.3万动作；
  manager参数变化非零，高层概率误差约8.88e-16，平均每个option约3.7个原子动作。
  这是运行健康证据，不是策略收益结论。
- 当前仍以CPU环境/候选计算为主，GPU0-3各两进程、约2.3–2.9GiB显存，
  单次利用率快照3%–4%；不宣称GPU已跑满。GPU4-7原任务保持不动。
- 研究结果仍在产生。归档未晋级策略，按协议自动进入第二阶段和最终确认；
  达到时限则保存，不能把未完成视为正常完成或悄悄续期。
