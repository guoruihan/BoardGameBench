# V1.2.10：持续搜索指导的纯模型 PPO

2026-10-20，包版本0.4.14，代码目录`boardbench.v130`。

## 研究问题与对照

用户确认：目标不是让模型只替代搜索器的一步动作，而是让模型吸收已有搜索知识，再通过真实对局持续改进。搜索只在训练时提供标签；评测不调用搜索，不用搜索兜底。

起点固定为`outputs/v129/ppo_native_a218_20261020_a/trials/anchor_a/best.pt`：64开发环境均分102.03125。在与V125搜索重叠的32环境上，该模型99.6875/动物18.59375；`mixed_compiled`搜索114.85417/动物44.07292，约900秒/局。此处是相同环境的能力比较，不是同耗时比较；全为开发结果，不能称正式测试。

原始示范只有128fit局/9619动作，BC初始化约49–56分/64dev；原anchor在50万动作撤掉教师约束。当前实验直接从已训练的最佳模型出发，不从弱BC或随机策略重跑。

|组|教师辅助系数|辅助数据|重复|
|---|---:|---|---:|
|control|0|无，仅继续PPO|2|
|static|0.05|原128局fit示范|2|
|correct_low|0.05|25%原示范损失＋75%累计新补标损失|2|
|correct_high|0.20|25%原示范损失＋75%累计新补标损失|2|

两次重复用13001/13002种子，各组同种子、同环境顺序前缀、同父模型/Adam/critic。不同动作长度导致到达动作预算时局数不完全相同，按实际动作和局数记录，不假称轨迹相同。

教师辅助不再按固定动作数衰减到0。这里是PPO加持续BC辅助，并非声称复现某篇完整算法；补标遵循[DAgger](https://proceedings.mlr.press/v15/ross11a.html)的学生访问状态纠正思想。目标仍包含真实回报优化，不要求永远复制老师。

## 三轮闭环

每轮先生成64个完整学生轨迹，随后8个模型独立训练200万**新增**PPO动作，最后自动进入下一轮。三轮完成保存停止，各轮和每个任务有单独上限，不需要逐轮确认。

- 第一轮：同一父模型，greedy/stochastic两种roll-in，各32个相同train环境。
- 第二、三轮：上一轮correct_low/high的两个种子，各16个相同的新train环境；a重复greedy，b重复stochastic。补标池由所有四个模型共享，保留所有历史补标。
- 每轮留出1/8的环境作为补标内部验证，不进入梯度；所有组从开始即排除三轮预定验证环境。父模型历史PPO可能见过这些train种子，因此不声称是从未见过的泛化测试。原示范验证/选择环境也持续排除。
- 每轮沿各自**final**模型继续，避免通过挑选开发checkpoint改变固定终点对照。另独立保留每轮best和跨轮best，按固定32dev选择，不按64dev结果挑模型。

教师沿用`mixed_compiled`终局rollout，保留原六个候选；模型实际动作若不在其中，添加为第七个候选。各候选用同一批公开未来采样；只接受完整平衡采样轮。每局累计900秒，不是每步900秒。

日志同时记录实际学生动作、教师推荐动作、候选终局估计、学生/教师估值差、公开前后状态和真实奖励。教师永远不接管真实轨迹。候选估值是habitat continuation下的估计，不是最优Q，也不是当前学生的无偏价值。

补标数据逐步用原始环境回放，核对状态、合法mask、编码、动作和真实终局。非强制标签要求至少两个完整采样轮；无有效搜索样本的fallback不进入补标训练。只有强制动作的数据集直接拒绝，避免名义上有补标但实际没有梯度信号。

PPO仍使用本轮模型的新鲜完整对局、原始得分增量/150、gamma1/GAE0.95；教师分数/轨迹回报不进入PPO的奖励、GAE或critic目标。沿用FP64模型、概率、优化器和V129原生编码环境加速。

## 评测与判断

- 曲线固定development[0:32]；每新增25万、50万、100万、200万动作评估。
- 每轮结束在development[32:96]评估initial/latest/curve-best，保存逐局总分、动物分、完成卡和自然终局标志。
- 主要方法对照看相同训练种子的**latest−control latest**，并报告相对本轮父模型的配对差；best仅表示可保留策略。两个训练种子不足以证明普遍方法优势。
- 教师拟合NLL/动作一致率明确标为fit诊断，不是泛化准确率；不把动作不一致全部当作错误。最终看纯模型的完整局成绩。
- formal test/confirmation不运行。补标成本、环境数量、训练动作和实际资源独立报告，不能将PPO等动作对照称为总计算量相同。

## 资源、版本和恢复

- b35：纯CPU教师，每轮65物理核（64worker＋控制）；符合最多112核、保留16核的授权。每个episode1080秒外层上限，每轮teacher task1800秒，包括失败重试的已用时间。
- a218：只用GPU0/1，每卡四个独立实验；每任务7环境worker＋learner，总65核，加外层pipeline1核，共66核<128。GPU5/6等其他任务不触碰。
- 每个PPO任务最多新增200万动作或1800秒，当前完整批次收尾及清理照实计账；8任务未全部成功不会静默推进。
- CPU使用共享节点锁，GPU每阶段启动重新检查空闲。基础预检用保守7200 CPU核秒＋600物理GPU秒覆盖；三轮跨机器编排预检另按实测增加3.319665 CPU核时＋0.036096物理GPU小时，正式配置合计5.319665 CPU核时＋0.202763 GPU小时。各receipt保留，不重复相加；历史V126–129费用单独保留。
- 每个run冻结源码、配置、split/dataset/parent哈希，原生动态库固定哈希。导入父模型时保留actor/critic/两优化器/累计计数；第一轮重新设定成对RNG种子，之后各自保留RNG。不是声称修改算法后仍与旧算法逐位续训。

首次启动，在a218项目根目录：

```bash
mkdir -p logs/v130
tmux new-session -d -s boardbench-v130-guided "bash -o pipefail -c 'cd /datapool/data3/storage/ruihan/code/boardgame && PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src .venv-train/bin/python -u -m boardbench.v130.pipeline launch --repo /datapool/data3/storage/ruihan/code/boardgame --root outputs/v130/teacher_guided_20261020_a --config configs/v130_teacher_guided_ppo.json 2>&1 | tee logs/v130/teacher_guided_20261020_a.log'"
```

外层会通过已验证IP192.168.2.58，在b35创建`bb130-<run>-round_XX-teacher_attempt_YY`命名tmux；日志仍在项目run目录，不阻断对话。纯CPU补标阶段a218 GPU没有训练进程是正常的：不能在新标签生成前声称它们已经被学习。

恢复：先确认旧pipeline PID退出；用新tmux/log名字，在同root执行：

```bash
PYTHONPATH=src .venv-train/bin/python -m boardbench.v130.pipeline resume --repo /datapool/data3/storage/ruihan/code/boardgame --root outputs/v130/teacher_guided_20261020_a
```

入口自动加载原run冻结源码。已经完成的阶段不重做；仍在运行的远程teacher继续观察；已终结的不完整teacher用新attempt、复用完成局并扣除旧attempt时间。PPO精确恢复latest里的模型/Adam/RNG/计数；预算耗尽则保存停止，不因resume自动续预算。远程teacher自身有独立看门狗，外层中断不会让它无限运行。缺失远程receipt时需先检查对应tmux，不盲目重启。

关键产物：

- `state.json`、`REPORT.md`、`results.json`：阶段、资源、逐组成绩和配对差。
- `retained_best.json`：按32dev曲线选出的可恢复检查点指针，不覆盖旧策略。
- `round_XX/teacher_attempt_YY/episodes/*/decisions.jsonl`：完整公开轨迹、教师标签和候选评估。
- `round_XX/corrections.npz`及JSON：可审计训练数据及来源。
- `round_XX/ppo/trials/*/training.jsonl`、`evaluations/*.json`：真实训练与逐局评估。
- `round_XX/ppo/curves.svg`、`curves.json`：该轮学习曲线。

## 预检发现

短至2秒的教师冒烟测试虽然完成了对局，但筛掉低采样标签后只剩强制动作，不能证明新纠正信号生效。已增加“非强制fit标签”门槛，并将预检教师提高到30秒；正式实验保持900秒，短预算成绩不用于淘汰算法。

首个b35预检因外层预先只暴露7核，触发原资源守卫“必须保留16核”的拒绝，在任何teacher worker启动前停止。重试保留完整物理核可见性，由已有调度器分配7核预检；没有降低资源安全门槛。失败与重试目录均保留，计入预检覆盖费用。

有效预检：`outputs/v130/preflight_teacher_b35_20261020_c`完成6个真实学生对局，回放通过，保留34条非强制fit标签；它只是链路预检，不作为正式900秒训练数据。`outputs/v130/preflight_cuda_a218_20261020_b`在GPU0/1分别使用这些真实纠正标签及V129成熟父模型，完成原生采样、持续BC/PPO、整局评估和下一更新逐位恢复；行为概率误差分别2.49e-14/1.95e-14，BC损失与actor更新均非零。冻结源码标识`4a68d7ecf9e52723332605f45364bac9b81c1dc723a9b9250690ee6eb488bba1`。

全量回归363passed/13显式GPU skips，293.52秒；V130针对性8passed/1显式CUDA skip。真实CUDA能力由上述独立GPU预检验证，不把常规测试的skip算作通过。

跨机器编排预检`outputs/v130/pipeline_smoke_20261020_a`使用同一源码、三轮真实补标/PPO/评估，但每局教师30秒、每轮仅新增1万动作；只检验b35→a218→新学生→下一轮的数据和恢复链路，不用于判断算法收益，也不导入正式实验。

该预检已完整完成：192个teacher-labeled学生对局、三轮共24个PPO任务全部成功，各final检查点哈希通过，均生成64dev自然终局评测。最后GPU0/1释放。正式配置不会使用预检模型或低预算标签，仍从同一个V129最佳模型重新进行900秒补标。
