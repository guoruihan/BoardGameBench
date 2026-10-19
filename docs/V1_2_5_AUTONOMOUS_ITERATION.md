# Harmonies V1.2.5：小批量自主策略迭代

目标是在**每局累计至多900秒solver时间**下提高自然终局真实分。沿用原游戏、评分和公开信息权限，暂不要求3秒部署。用户授权按任务自主分配预算、提出新变种并依据结果推进；没有新的全局核时/08:00截止。第一阶段做3轮真实研究迭代，不保证每轮涨分。

## 运行约束

- b35权限上限112/128物理核、至少保留16核，日常目标约100；小批量没有足够独立任务时少用核，不增加无意义样本。首轮20worker+1控制核，GPU0。
- 每批独立`task_seconds`和每局`timeout_seconds`；首轮批次1500秒、单局watchdog1020秒，solver900秒。整体任务额外时间用于启动、记录和收尾，不增大solver预算。
- 所有源码/配置/划分/任务在启动时冻结，实际策略进程从冻结源码导入；命名tmux、tee日志、共享CPU锁、独立watchdog和失败保留复用v122。
- 物理核预留×墙钟的保守成本与进程CPU时间分开记录；工程验证、失败、复测同样计入。这里的预算按任务分配，不等于无超时或无限重试。
- 当前复用的基础调度器仍保守限制每批不超过主机一半物理核；首轮只需21核，不妨碍任务。需要超过64核时再以独立执行改动验证资源约束，不能静默放宽旧版本。

## 第1轮（先冻结，后读结果）

配置`configs/v125_round1.json`：development索引96–99，搜索seed12520261020，五策略各4局，全部900秒。这里的开发池历史上可能已被使用，不称未见测试；正式confirmation/test不参与调参。

1. `original`：保留原启发式后续玩家和top6根候选。
2. `habitat`：保留已有动物目标后续玩家和top6。
3. `habitat_fast`：在完整动作缓存之外，精确缓存card_goals真正依赖的棋盘、资源计数、卡ID及已放动物数。默认评分和动作排序不变。
4. `mixed_roots`：原启发式与habitat推荐交错去重取6个根动作；原top1保留为无有效采样时的fallback；后续使用等价加速habitat。
5. `resource_habitat`：对所需资源和可放置步数考虑机会成本，晚局降低长目标的影响，并保留更多地形收益；这些仅是启发式，不改变奖励或合法性。

三个变种是不同机制，不是同一权重网格。后续两轮必须读取这一轮及后续轮结果再决定方向，不能把预先写死的扫描称为自主决策。

## 第2轮：根据第一轮的实际结果选择

首轮20/20完整终局：original113.00、habitat103.25、habitat_fast104.25、mixed_roots114.00、resource_habitat104.75。mixed相对habitat+10.75，但相对original仅+1.00，区间很宽。动物分48.5 vs original35.5，同时地形65.5 vs77.5。缓存版模拟步数仅比habitat约+0.8%；资源版明显更慢。不能因为历史habitat平均更强，就忽略这批original更强的证据。

据此保留original、habitat、mixed_roots，新增：

- `mixed_compiled`：预编译合法静态堆叠扩展，减少每图案的Counter/product工作；与原card_goals及rank_actions逐项等价，仍是同一后续策略。
- `mixed_portfolio`：每一完整公共采样轮固定使用original或编译habitat，交替进行，各候选同一轮使用同一个玩家；估计的是该玩家混合分布的平均终局分，不是对不同玩家的噪声估计取最大。
- `mixed_finish_terrain`：公开剩余资源推算最多3回合或空格至多4格时切回original，其余采用编译habitat；检验动物布局后以地形收尾是否减少损失。

配置`configs/v125_round2.json`，development100–103，24局、25核。训练预检使用train索引1，不按其10秒分数筛掉任何正式候选。

## 第3轮：由第二轮证据改查候选覆盖和选择噪声

第二轮24/24完成：original93.50、habitat111.00、mixed_roots116.50、mixed_compiled114.75、mixed_portfolio110.75、mixed_finish_terrain105.00。两种不同后续玩家都没有超过mixed_roots；编译版模拟步数比mixed_roots约多9.9%，但没有清晰得分收益，不能把速度收益当得分收益。

mixed_roots的270次有效搜索决策中，170次合法动作超过6个、41次选择根排序第5/6个动作；168次前两名模拟均分差不足1。这支持进一步检查覆盖与不确定性，但不能证明被剪动作更优或近似平分源于噪声。

- `mixed_wide`：交错候选扩到10个，保留原6个前缀，编译habitat后续不变；检验更多方案能否抵消每个方案样本减少。仍以每局900秒直接比较，不按10秒预检分数筛选。
- `mixed_confident`：相同6候选与编译habitat；完整共同种子轮上计算每个动作相对根0的回报差方差，用“原始终局均分−配对差标准误”选择。根0不罚；不足2样本保守保留根0。这个惩罚是启发式，不宣称置信保证；原始Q与选择目标分开记录，不能混成价值学习标签。

配置`configs/v125_round3.json`，development104–107；两新变种加original/mixed_roots/mixed_compiled，20局、21核。训练预检只用train索引2。前两轮44局3173个真实动作均已从环境seed重新执行，所有前后状态与终局分完全一致。

## 比较与保留

默认4环境×1搜索seed，方法间严格配对。4局用于筛选，不宣称统计可靠或游戏已解；有潜力/互补/不确定的候选才补2–4环境或1搜索重复。每轮加入/轮换部分开发环境，最后用未参与本阶段选型的小开发子集检查；不同环境原始均分不能直接排序。

记录真实终局分、动物/地形、放动物/完成卡、实际solver时间、模拟步数和fallback。完整公共种子采样轮才可进入Q均值，末尾残缺轮仍收费但不计分；未评估候选均值为null，不能当零分负例。配对置信区间只是描述性统计，存在自适应选型偏差。

保留原基线、高分候选、快速方案和互补方法；榜单区分小样本暂领先、补样后参考及单局最高分。没有自动生产策略更新。

## 数据与版本

新增日志显式保存动作前公开观测、动作、动作后观测、原始分数增量；整局`result.json`保存自然终局分。候选记录动作、样本数、均值、样本方差（不足2样本则null）；这些是特定后续策略下的估计，不是最优Q。

源码与旧入口通过默认不变的可选参数隔离。`task_plan.md`等恢复记录位于`tasks/main.harmonies-autonomous-iteration/`；实验数据归`outputs/v125/`、日志归`logs/v125/`，不提交生成的逐局数据。每轮实验前记录源代码提交和冻结快照。

## 执行

在b35上，以项目绝对路径作为`--repo`与`--root`，新任务目录不可覆盖：

```bash
tmux new-session -d -s boardbench-v125-r1 \
  'bash -o pipefail -c "cd /datapool/data3/storage/ruihan/code/boardgame && PYTHONPATH=src .venv/bin/python -u -m boardbench.v125.campaign launch --repo /datapool/data3/storage/ruihan/code/boardgame --root /datapool/data3/storage/ruihan/code/boardgame/outputs/v125/round1_20261020_a --plan configs/v125_round1.json 2>&1 | tee logs/v125/round1_20261020_a.log"'
```

启动前必须验证当前资源、源码与训练集校准，日志目录须已存在。`--preflight-core-seconds`可声明实现/验证费用；不能跨多个任务重复计入同一费用。`task_seconds`在启动时转换为该任务独立的绝对截止。

每轮的`REPORT.md`和`summary.json`自动更新。研究分析与下一轮修改由代理持续执行，调度器本身不负责创造新策略；仅启动tmux不构成三轮完成。

## 恢复与数据导出（0.4.8起）

新版本任务如果被中断，只能在父任务watchdog已经收尾、无活跃worker后恢复，创建新目录，保持任务身份和父策略源码：

```bash
PYTHONPATH=src .venv/bin/python -m boardbench.v125.campaign resume \
  --repo /datapool/data3/storage/ruihan/code/boardgame \
  --parent /datapool/data3/storage/ruihan/code/boardgame/outputs/v125/PARENT \
  --root /datapool/data3/storage/ruihan/code/boardgame/outputs/v125/NEW_ATTEMPT \
  --task-seconds 1500
```

已完成自然终局直接复用、不会重新执行；未完成局从头重试但保留原失败及成本，不伪称恢复到了原局中间。`resume.json`记录血缘，统计结果通过`inherited_from`定位真实轨迹。父任务的核时保留在父账本，跨任务汇总避免重复。初始0.4.7首轮冻结驱动尚无恢复hook，工具会明确拒绝对它自动恢复，而不是静默重复；已完成首轮无需恢复。

```bash
PYTHONPATH=src .venv/bin/python -m boardbench.v125.research \
  --repo /datapool/data3/storage/ruihan/code/boardgame \
  --root outputs/v125/research_20261020
PYTHONPATH=src .venv/bin/python -m boardbench.v125.data \
  --run outputs/v125/COMPLETED_TASK --out outputs/v125/datasets/UNIQUE_NAME.jsonl.gz
```

导出仅接受已收尾任务及train/development，验证合法动作、相邻状态、分数增量和自然终局；失败不伪造训练目标。文件不覆盖，附SHA256/条数清单。`environment_reward`是游戏接口的终局奖励，`raw_score_delta`是额外记录的计分变化，二者不可混淆。候选Q是模拟后续玩家的绝对终局估计，非最优值。
