# V1.2.6 / 0.4.10：a218 八卡搜索指导学习

2026-10-20。用户授权立即启用a218八张卡，b35仅纯CPU。本版复用原教师数据，不改变规则、信息权限、900秒累计决策预算或正式测试隔离。

## 分工与实验设计

- b35现有 `learning_20261020_b_teacher_a1` 完成144局后保留：train2048–2175的128局拟合，train2176–2191的16局验证。跨机使用共享项目目录，不重复生成、不将开发集轨迹用于拟合。
- a218运行8个学生：BC／BC+候选Q条件软标签 ×隐藏宽度128／256 ×种子12601／12602，每张A100一个独立模型。每模型最多120epoch/1800秒，训练内验证NLL选检查点，25epoch无改善提前停止。
- 用户确认优先按独立实验分卡，允许单卡多任务；当前没有DDP或跨卡梯度同步。后续根据实际吞吐、显存和CPU余量决定是否在一张卡并发多个独立任务，不能把“分配八卡”误报成八卡一直满载。
- 纯学生诊断128局；900秒pilot在相同4个dev环境108–111、2个搜索重复上比较2基线+16个学习搜索变种，共144局。每个学生分别测试候选提议和后续rollout，不能用低预算分数淘汰900秒方法。
- 自动选择一个有希望的学习方向，在32个新train环境2304–2335做DAgger补标；用原128局+补标训练2个新种子，再在8个未参与本轮选择的dev环境112–119、2个搜索重复上比较7策略，共112局。所有学生保留，不自动替换生产策略。
- 本轮仍是搜索先验指导的学习与策略迭代，不是新一轮从零PPO。GPU训练阶段八卡并行；搜索和环境回放主要CPU-bound，不能保证所有阶段GPU利用率都高。

新的900秒搜索在a218完成288局（pilot144+DAgger32+fresh112），加复用教师144局，总432局/108个累计solver小时；纯学生诊断另160局。每个任务有独立超时，成本包含预检、失败尝试、旧教师及旧编排（仅继承一次）。新config中的2CPU核时是保守迁移/验证预留，不是测量的GPU占用。预计接续后约1.5–2小时，实际以状态和报告为准。

## 资源与安全

配置：`configs/v126_a218_search_learning.json`。a218最多101搜索核+1外层编排核，训练最多16核+1编排核；整个分配集合不超过128个物理核代表CPU。复用节点级CPU锁，GPU训练前重查空闲与真实CUDA。8卡必需的首轮若资源不可用，显式停止并保留进度，不静默回退CPU，也不抢占他人任务。

外部教师必须通过：完整非活跃状态、source/plan/splits哈希、精确训练环境、策略预算、逐局自然终局，以及数据构建时所有状态/动作的完整重放。教师与训练源码允许不同版本，但各自来源固定并记录。

b35旧外层PID18000于07:01:26CST被单独SIGSTOP，独立监督的教师工作进程继续运行。教师全部完成并退出后才允许向旧外层发送SIGTERM/SIGCONT，确保其不会进入CPU训练或重复派发。保留原始run/state，即使它以迁移中断状态结束也不改写成“实验完成”。后续正式run不得沿用旧b35的GPU UUID或CPU亲和性。

## 验证与运行记录

- a2188/8张卡通过真实CUDA前向/反向；每卡另通过一次实际BC/Q训练、选中模型导出、CUDA检查点恢复和NumPy/Torch输出一致性测试（约9.6–9.7秒/卡）。
- 完整回归 **327passed、10项显式GPU资源跳过**，275.31秒；新增V126 GPU测试已另在8张授权卡分别实际通过。a218自动流水线校准已完成29/29自然终局、0失败、3次小模型训练，预留0.08569CPU核时。六个CPU子阶段源码哈希均与外层一致：`e031735c1492f74ec4df65666a9403303c354a693de8c7e2fd8ae2fb618aa285`。完成后的恢复命令已验证不产生新任务。
- 新正式运行：`outputs/v126/learning_a218_20261020_a`，下述启动与实测记录已核验。

### 已启动并进入正式评估

2026-10-20 **07:10:47CST** 在a218启动，命名tmux `boardbench-v126-a218`，日志 `logs/v126/learning_a218_20261020_a.log`，外层PID6189固定CPU127。源码提交 `eb22c7b` 已推送origin/main，冻结哈希为上述e031735c。b35原外层PID18000于07:10:43在教师144/144完成、watchdog退出后终止，旧状态中的KeyboardInterrupt是预期迁移记录；没有任何教师局被杀或重跑。

完整重放得到10788条记录（拟合9619、训练内验证1169）。**8/8学生均已实际在不同GPU上完成训练**，每个参数发生更新，模型/优化器检查点与曲线已保存。选中检查点的验证NLL约1.691–1.722；这只是拟合指标，不是游戏得分提升。

测得模型训练循环约9.74–12.77秒，早停发生在35–48epoch，所选最佳在10–23epoch；每卡总显存约635–655MiB，5秒间隔采样利用率约3%–13%。证据支持后续将多个小模型任务安排到同卡，当前8个独立任务已释放GPU，未让空闲进程持续占卡。曲线后段验证损失回升，不能靠延长训练代替更多高质量数据或更好的表征。

截至07:12左右，纯学生128/128局完成，正式900秒pilot已自动启动100个工作进程，搜索分配101核+外层1核，合计102核（低于128）。后续自动补标、重训、fresh复核无需逐轮批准。a218小测GPU成本额外按240GPU秒保守预留写入receipt（不是实测kernel时间）；CPU验证已在config中记2CPU核时，不重复记账。继承旧教师+旧编排总56.5448CPU核时，仅计一次。

## 启动与恢复入口

在a218项目目录下使用命名tmux和日志：

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  .venv-train/bin/python -u -m boardbench.v126.pipeline launch \
  --repo /datapool/data3/storage/ruihan/code/boardgame \
  --root outputs/v126/learning_a218_20261020_a \
  --config configs/v126_a218_search_learning.json
```

`launch`仅用于新的输出目录；不能对正在运行的目录再启动。恢复前确认旧PID不存在、子campaign的watchdog已经完成、节点资源空闲，使用该run自身的冻结源码和原节点：

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=outputs/v126/learning_a218_20261020_a/source/src \
  .venv-train/bin/python -u -m boardbench.v126.pipeline run \
  --repo /datapool/data3/storage/ruihan/code/boardgame \
  --root outputs/v126/learning_a218_20261020_a
```

报告：run内 `REPORT.md`、`state.json`、`gpu_probe.json`；各模型 `models/initial/*/{curve.jsonl,progress.json,best.npz,latest.pt,summary.json}`。曲线反映拟合进展，真实策略强弱以pilot/fresh配对分数为准。
