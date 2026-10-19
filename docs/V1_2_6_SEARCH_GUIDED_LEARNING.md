# V1.2.6：搜索指导学习

2026-10-20，代码版本0.4.9。目标是提高**每局累计900秒决策时间**下的真实自然终局分，不要求3秒部署。规则、评分、公开信息边界和正式confirmation/test保持不变。

最新资源分工：**GPU+CPU混合任务放到a218，b35仅纯CPU；所有a开头节点每台合计最多128个CPU核**。b35沿用最多112核、保留16核。详见[资源约束](RESOURCE_POLICY.md)。下文b35 GPU探测、CPU回退及冻结配置为原始启动记录，不是继续使用b35 GPU的授权；现有任务未因本次约定自动迁移。

## 当前资源事实

b35（192.168.2.58）的GPU0/1/2/4/5能被nvidia-smi列出、空闲，GPU3报设备handle错误。但五张可见卡逐UUID实际执行PyTorch CUDA均失败，错误为CUDA unknown error。直接加载系统libcuda.so.570.86.10调用cuInit也返回999/CUDA_ERROR_UNKNOWN，不只是PyTorch检测失败。这不是已成功使用GPU。流水线训练前重新检查空闲和真实CUDA运算；无法使用的设备不分配，使用CPU小模型回退。不重置驱动、不停止其他任务。

前一轮 `outputs/v125/expanded_20261020_a` 继续运行。新CPU密集实验等待其watchdog退出、资源锁和负载检查通过后启动。外层编排固定物理核111，CPU搜索最多101核（含控制器），总数最多102；训练阶段最多5个进程各2核。所有CPU/GPU成本、失败尝试与实现预检都记账。

## 冻结的第一轮设计

|阶段|环境与重复|方法/目的|
|---|---|---|
|教师生成|train索引2048–2191，共144局，900秒/局|mixed_roots；128局训练，16个独立环境做训练内验证|
|学生训练|同一训练/验证数据|BC128两个种子、BC+候选Q软目标128两个种子、BC256一个种子|
|纯学生诊断|train2208–2223，共16环境|全部5模型；仅诊断，不按低耗时分数淘汰900秒方案|
|900秒首轮|dev108–111，4环境×2搜索seed|mixed_roots、habitat，以及每模型的候选提议/后续玩家两个独立改动，共96局|
|学生状态补标|train2304–2335，共32局，900秒/局|让首轮最有希望模型自己走，搜索只给所遇状态标动作，不把专家动作冒充学生行为|
|重训|原示范+新补标；保持原16环境验证|保留选中架构/损失，用两个新训练seed重训，原模型不覆盖|
|新开发环境复核|dev112–119，8环境×2搜索seed|两基线、首轮选中方案、两重训模型各两种搜索用途，共112局|

训练seed12601/12602，重训12611/12612；搜索seed12620261020/12620261021。环境使用既有split manifest，开发环境与V125选型96–107及历史扩测64–95分离。但这是**开发集复核，不是从未参与整个项目评测的正式测试**。

每个训练任务最多120epoch/1800秒，训练内验证NLL连续25epoch无改善提前停止；保存最低验证NLL模型。示范任务3600秒、首轮2400秒、补标1800秒、最终复核3600秒，均另有逐局900+120秒watchdog。资源等待也有上限。首轮按完整900秒配对均分选择一个方向做补标；即使没有超过基线，也保留负结果并完成一次有界纠偏，不无限追分。

预算是独立任务上限，不是900秒总研发限时。正式搜索总计384局、96个累计solver小时；CPU预留成本高于该数，训练、校准、等待、预检另计。预计前轮释放资源后约2–3小时获得完整一轮结果，实际以状态文件为准。

## 方法与防泄漏边界

- 复用现有cards_v1公开编码（1346维）和slots240_v1动作空间。禁用人工STOP，必须自然终局；不输入环境seed、搜索seed或隐藏未来。
- 小MLP使用两个Tanh隐层，先用BC学习老师所选动作。辅助版本额外学习**候选集合内部**温度5的Q软分布；只用至少2个样本的候选，未评估值null不当零分或负例。Q是habitat后续策略的终局分估计，不是最优Q。
- 实际行为的终局return与候选Q分开保存。本轮不训练或使用价值截断，也不运行PPO；先隔离策略学习的作用。
- 候选版本固定6个根动作，保留原前5个和原top1回退，最后一个尝试模型推荐的集合外动作。后续玩家版本保持原混合候选，单独用模型替换rollout策略。
- PyTorch训练，导出FP32 NumPy权重做单线程CPU搜索；逐状态验证两种推理的一致性，所有编码/推理/缓存成本在solver累计计时内。
- 数据构建逐局从公开环境seed重放，核对前状态/行为动作/后状态/终局分。DAgger标签取expert_action，实际状态推进仍用学生动作。按环境划分，不随机切相邻轨迹行。
- 训练内检查点选择只用train内部验证；策略方向在dev首轮选择，再用不同dev环境复核。不能把数据里的数万相关动作当成数万个独立环境。

## 运行和恢复

正式入口：

```bash
cd /datapool/data3/storage/ruihan/code/boardgame
mkdir -p logs/v126
tmux new-session -d -s boardbench-v126-learning \
 'bash -o pipefail -c "PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv-train/bin/python -u -m boardbench.v126.pipeline launch --repo /datapool/data3/storage/ruihan/code/boardgame --root outputs/v126/learning_20261020_b --config configs/v126_search_learning.json 2>&1 | tee logs/v126/learning_20261020_b.log"'
```

外层启动后冻结源码；后续每个子任务都从外层冻结快照复制，不读取变化后的工作区。子任务各有独立源码、计划、种子、计时、逐动作轨迹、最终结果、状态及watchdog。源码/配置变化不热更运行中的策略。主报告 `outputs/v126/learning_20261020_b/REPORT.md` 自动更新；各子任务还包含 `summary.json`、`curve.csv`、`REPORT.md`。

确认旧进程已退出后，从同一冻结源码恢复：

```bash
PYTHONPATH=outputs/v126/learning_20261020_b/source/src \
 .venv-train/bin/python -u -m boardbench.v126.pipeline run \
 --root outputs/v126/learning_20261020_b \
 --repo /datapool/data3/storage/ruihan/code/boardgame
```

实际长恢复同样放命名tmux并tee到新日志。恢复跳过已完成阶段；CPU失败阶段新建attempt，复用完成的完整局，保留失败成本；没有完整watchdog终结记录时拒绝盲目重启。训练保存模型/Adam/RNG/epoch，按epoch恢复，不宣称中途minibatch逐位恢复。

校准配置 `configs/v126_pipeline_calibration.json` 仅使用train环境、0.15秒短搜索、2epoch，验证整个闭环与恢复。其分数不可用于正式策略选择。正式启动前运行针对性测试与全流程校准。

## 已完成的启动前验证

- `outputs/v126/calibration_20261020_{a,b,c}` 各29/29局自然终局、0失败，且各完成3个小模型训练。c为前一冻结源码834e263ace72ec033bc6c1850f5397f72d25ef2403a5874af44a4ccd3da69be3，记账0.0891 CPU核时、GPU0；已完成任务恢复未创建新局。
- 初次全量318通过、9项显式CUDA资源跳过。最终变更相关47项通过；最新10项V126加1项旧源码一致性检查共11项通过。第二次全量测试与源码编辑重叠曾触发`child source mismatch`，保留该负记录，冻结后对应检查通过，没有关闭源版本验证。
- 新增检查包括候选mask/空Q、合法动作、真实终局rollout、Torch/NumPy推理一致、按环境隔离、数据全量重放、完整epoch恢复及GPU探测超时回退。
- 冻结后完整回归321通过/9 CUDA跳过。最终快照继承修复相关55项通过；`calibration_20261020_d` 使用最终源码5ce3ca2a16ab22eb2c851c19d6890c49a6dce11d259e372eefbd715743967df3，实际子任务继承外层源码哈希已核对。

首个待机尝试 `learning_20261020_a` 在任何子任务启动前停止：审查发现后续子任务快照需要强制继承外层冻结源码，而不是复制届时的工作区。该尝试没有训练局或模型，旧V125没有中断；保留其记录，5.027246 CPU核时（含5核时统一预检额度）计入替换尝试b的预检成本一次，不重复相加。

## 正式后台启动快照

2026-10-20 **06:32:12 CST** 启动 `learning_20261020_b`，tmux `boardbench-v126-learning`，冻结代码提交 `49cc2cd`、源码哈希 `5ce3ca2a16ab22eb2c851c19d6890c49a6dce11d259e372eefbd715743967df3`。编排PID18000，单线程固定CPU111；06:32:40心跳新鲜，处于 `waiting_for_prior_campaign_or_cpu_load`，尚未生成正式训练局。旧扩测400/480完成、80任务仍运行，未中断。

最终校准d已29/29自然终局、0失败、3个训练任务完成，0.08584 CPU核时/GPU0；六个CPU子阶段源码哈希全部与正式外层冻结哈希一致。后续阶段将自动接续，无需用户逐轮确认；正常完成后主报告保留全部策略、对照、成本和下一步判断，不自动更新生产策略。

GitHub已成功接收初版提交691fedb；后续修复49cc2cd已本地提交，发布遇到SSH banner间歇超时，暂不能宣称远端包含最终修复。运行依赖已保存的本地冻结源码，不依赖推送成功。
