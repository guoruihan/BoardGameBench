# V1.2.11：配对反事实与后续策略对齐

2026-10-20，包版本0.4.15，代码`boardbench.v131`。保留V130所有源码/模型/日志，不继续原配置的第三轮。

## 问题与已完成诊断

V130第二轮同64开发环境、两训练种子平均：纯PPO103.1328，静态示范102.8359，弱纠正99.8359，强纠正94.8125。强纠正训练标签匹配率提高，但真实整局分降低。标签估计的是habitat后续的收益，不是学生后续的收益。

从旧的128条学生轨迹、48个train环境中回放选择局面，不新增环境、不触碰test/confirmation。各环境按早中晚期/动作类型轮转选16个师生分歧状态，不按旧估值差挑选。所有公开前后状态、实际动作、最终分与原始引擎逐步一致。旧教师动作固定后，用新的公开未来重新评估，不能用选胜者的旧样本证明它更强。

冻结学生为V130第二轮`control_a/final.pt`，父检查点SHA256 `6ac3857ce587bef4cb2913e499099e9d4606ba27cbc308dbb104c248d16f6125`。该模型曾在重复使用的64dev取得103.6875，因此起点包含开发选择偏差，不是未见测试冠军。

首个诊断：16环境各4局面，共64状态；每状态32个新未来，每个未来执行教师/学生第一步 × habitat/学生后续，全部模拟至自然终局。未来只由`Harmonies.from_observation`按公开信息采样，不读取真实牌序、隐藏RNG或原局未来。

`outputs/v131/continuation_guidance_20261020_a/diagnostic`：64任务成功，8192完整分支，435506模拟步，28.69秒×65预留核=0.5180 CPU核时。

|教师动作−学生动作|环境均值|描述性95%区间|
|---|---:|---|
|之后habitat玩完|+1.8325|[1.2028,2.4623]|
|之后冻结模型玩完|−0.8682|[−1.5731,−0.1632]|
|模型后续优势−habitat后续优势|−2.7007|[−3.7945,−1.6068]|

29/64状态出现均值符号翻转，3个满足更保守的相反区间条件。按环境先聚合，不能把8192分支当成8192独立环境；这些是选择后的训练分歧局面，不是整局策略成绩或正式泛化结果。支持后续策略不匹配，不能说明所有专家知识无效。

## 数据与训练

后续768个状态使用另一套独立未来流，每个状态32次四分支采样。6个原有train内部验证环境完全不进本轮梯度，所有训练分支的在线环境也排除这些种子；父模型历史可能见过它们，不宣称从未见过。

保存每次分支终局总分/动物分/步数与随机未来种子，按同未来的收益差计算标准误。导出NPZ时重算配对统计、验证合法动作、公开状态哈希、样本数和父策略身份。

指导样本权重固定为`clip((mean_gain - 1.96 * paired_se - 1) / 5, 0, 1)`。这只是保守筛选启发式，不是多重比较/小样本的正式置信保证。近似平局、负优势和高噪声建议不强制模仿。每个指导分支至少32条非强制fit标签才允许运行，否则停止并报告，不静默变成另一个PPO对照。

三个分支各两个训练种子：

- control：只继续真实回报PPO。
- habitat：用habitat后续配对优势筛选、加权教师动作交叉熵。
- student：相同损失、相同超参数，仅筛选依据换成冻结学生后续优势。

指导最大系数0.02，所有分支同一父模型、同一原始奖励/网络/PPO设置/成对新RNG。保留actor、critic、两个Adam状态与累计计数；这属于新方法实验，不声称跨算法逐位续训。同run恢复需源码/配置/检查点哈希一致，保留全部RNG。

教师模拟分数不进入PPO回报、GAE或critic目标。另记录一次确定性小批量上的PPO策略梯度/加权教师梯度范数和余弦，只作干扰诊断；不消耗训练采样RNG。fit与按环境留出的validation标签指标分开报告。

每个任务最多新增200万动作/1800秒，曲线development[:32]，完整终点评估development[32:96]，自然终局、纯模型贪心，不搜索兜底。保留所有final及curve-best，主要比较固定终点latest与同种子PPO；32dev选择的保留模型与64dev观察最优不能混称。

## 资源与恢复

- b35：65物理CPU，保留至少16核；诊断600秒、标签3600秒任务上限，单状态240秒。复用原独立看门狗、全版本CPU锁、冻结源码和失败回收机制。
- 离线状态分支诊断的限额是单独任务预算，不把每局900秒偷换成每步900秒；此处不是900秒搜索策略测评。
- a218：原计划GPU0/1各3任务，但启动前被其他任务占用，资源守卫拒绝预检。按已有单卡多实验授权改为当前空闲GPU2承载6个独立任务。7环境worker+learner，每任务8核，加控制器共49核，不超过128核。每次启动重新检查GPU/CPU空闲并拿共享节点锁，不抢占其他任务。
- 预检保守预留14400 CPU核秒和600物理GPU秒，涵盖本地实现回归和两卡实际数据/native精确恢复检查，正式账本只计一次。诊断/标签额外按watchdog实际预留核×时间计费；同卡并发训练按GPU区间并集计物理卡时。
- GPU任务在命名tmux里运行；每批更新保存原子checkpoint/heartbeat。报告仅在summary与checkpoint完整时读取，修复V130先出现comparison、后出现summary导致控制器退出的问题；失败状态独立于报告持久化。

关键命令（均在项目根目录，设置`PYTHONPATH=src`及单线程BLAS；完整实跑命令和run IDs见tasks记录）：

```bash
.venv-train/bin/python -m boardbench.v131.prepare --repo "$PWD" --root outputs/v131/NEW_RUN --config configs/v131_continuation_guidance.json
# b35
.venv-train/bin/python -m boardbench.v131.campaign launch --repo "$PWD" --root outputs/v131/NEW_RUN/diagnostic --plan outputs/v131/NEW_RUN/diagnostic_plan.json
.venv-train/bin/python -m boardbench.v131.campaign launch --repo "$PWD" --root outputs/v131/NEW_RUN/labels --plan outputs/v131/NEW_RUN/labels_plan.json
# a218; paired.npz必须先通过data.build回放/哈希/样本验收
.venv-train/bin/python -m boardbench.v131.preflight launch --repo "$PWD" --out outputs/v131/NEW_PREFLIGHT --config configs/v131_continuation_guidance.json --paired-dataset outputs/v131/NEW_RUN/paired.npz
.venv-train/bin/python -m boardbench.v131.launch launch --repo "$PWD" --root outputs/v131/NEW_RUN/ppo --config configs/v131_continuation_guidance.json --paired-dataset outputs/v131/NEW_RUN/paired.npz
.venv-train/bin/python -m boardbench.v131.launch resume --repo "$PWD" --root outputs/v131/NEW_RUN/ppo
```

GPU恢复前需确认原控制器和其子进程均已结束，不能复制活跃任务。CPU阶段的已完成产物保留；失败任务需新attempt并扣除旧费用，不覆盖原attempt或自动重置预算。

方法参考：[LOLS](https://proceedings.mlr.press/v37/changb15.html)区分参考与学习策略后续收益；本实验是受其问题设定启发的PPO辅助实验，不宣称复现算法或继承其理论保证。
