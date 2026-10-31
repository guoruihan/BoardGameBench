# V1.6.0 环境族与冻结对手交付

本轮包含 17 款固定原版双人配置、80 个可搬移 bot。15 款新环境完成策略训练，
Harmonies 复用已有双人 RL；Five Tribes 保留研究态及随机／启发式，不提供已验证的
学习对手。覆盖范围见 [配置表](Family_Overview_zh.md)，算法与评测设置见
[方法](Methods.md)，成绩见 [结果](Results.md)。

这是一组初始基准对手。通用神经策略是人工候选特征上的残差 actor-critic，
训练目标是胜负；固定启发式不是高预算搜索或人类专家。当前结果不标定游戏上限。

## 版本和文件

运行代码冻结在 `6bcb71173ab9257d99f66404986862012b46d2a2`，Python 包版本
`0.5.2`。最终研究版本的文档提交可以晚于此运行快照；不得将文档提交冒充训练代码。
开发、确认的原始 manifest 各自保留实际源码身份，前六款确认使用更早的冻结源码。

以下路径相对于项目根目录；权重、原始对局和交付包保留在共享存储，不提交 Git：

| 内容 | 路径 |
| --- | --- |
| 一体交付压缩包（约 13 MiB） | `outputs/v160/boardbench_v1.6.0_delivery.tar.gz` |
| 压缩包对应目录 | `outputs/v160/delivery_v160/` |
| 全量冻结对手池 | `outputs/v160/frozen_pool_v160/` |
| 可安装 wheel | `outputs/v160/package_052/boardbench-0.5.2-py3-none-any.whl` |
| 全量安装验收 | `outputs/v160/package_052/final_acceptance/acceptance.json` |
| 各款训练与评测 | `outputs/v160/bots/<game>/` |
| 开发集选模记录 | `outputs/v160/selections/<game>.json` |
| 全部确认资格核对 | `outputs/v160/reports/qualification_final.json` |

16 款保留池两两互战合计 157,696 场：157,654 场自然完成、17 场 Jaipur 规则未决、
25 场 New World 截断。每款主策略对启发式均安排 1,024 场；其中 Jaipur 有 2 场未决，
New World 有 5 场截断，其他主策略对启发式对阵全部完成。缺失结果的上下界与抽样
置信区间分开报告，原始平均分仅使用自然完成局。

外部保存的可信摘要如下。对手池摘要是 manifest 的规范化内容摘要，**不是**对
`manifest.json` 原始字节执行 `sha256sum` 的值，应使用下方 verify 命令。

```text
delivery archive SHA256:
4d590056b5ce5cff36113e3a164fbd2b9d03ab1770a5be86d0fc05e39e985dd0
wheel SHA256:
f2924bb4ef5ba466cb89a5d0921732d38f13dc4df9e4bd4e914a9525088bae07
pool manifest content SHA256:
59bd6210692fbf4998deca42ff105852172093ac94203e303bae1faaf57eef93
runtime package content SHA256:
1f803c9988898ab155360a35265a6c4bf8b1f60a9ef014807ae347b32a23ab3b
```

池内有 17 个随机、17 个启发式、11 个 CEM 线性、34 个通用神经以及 1 个
既有 Harmonies RL，共 35 个模型文件。每款的 `recommended` 在开发阶段冻结，
其他保留策略仍可选；Five Tribes 的 `recommended` 为空，状态为 `research`。
配置、规则版本、模型 SHA、开发来源和规则边界都记录在池的 manifest 中。

## 在当前项目运行

```bash
.venv-train/bin/python -m boardbench.families.bundle verify \
  outputs/v160/frozen_pool_v160 \
  --sha256 59bd6210692fbf4998deca42ff105852172093ac94203e303bae1faaf57eef93

.venv-train/bin/python -m boardbench.families match --game cascadia \
  --bot0 outputs/v160/frozen_pool_v160/games/cascadia/bots/ppo_prior_r2.json \
  --bot1 heuristic --seed 36001000 --trajectory \
  --output outputs/v160/example_cascadia.json
```

原子动作接口包括合法反应和额外选择，不能假设双方严格交替。
让外部 agent 对战时，用 `env.current_player` 判断当前行动者，仅传给策略该玩家的
`env.observe()` 与 `env.legal_actions()`；不要传入私有 snapshot。
对手对象每局创建一次并保留到结束，不能每步重新创建。

```python
from boardbench.families.bundle import load_bundle, load_bot
from boardbench.families.registry import game_class

root = "outputs/v160/frozen_pool_v160"
manifest = load_bundle(root, expected_sha256=
    "59bd6210692fbf4998deca42ff105852172093ac94203e303bae1faaf57eef93")
game = "cascadia"
name = manifest["games"][game]["recommended"][0]
bot = load_bot(root, game, name, seed=123)
env = game_class(game)()
env.reset(seed=456)
env.step(bot.decide(env.observe(), env.legal_actions()))
```

## 搬到其他机器

整体复制 wheel 和对手池，使用独立 Python 环境安装 wheel。已验证的依赖为
Python 3.10.4、NumPy 1.26.4、PyTorch 2.4.1；推理使用 CPU，不需要 GPU。
已有匹配依赖时可 `python -m pip install --no-deps /path/to/boardbench-0.5.2-py3-none-any.whl`。
然后从源码目录之外执行相同的 `bundle verify`，路径替换为搬移后的目录。
内部模型引用均为相对路径，原训练目录不需要一并复制。

已实际验证独立 wheel 安装、异地工作目录、带空格的搬移路径、17 环境的初始化／
合法一步／JSON 恢复，以及全部 80 个真实 bot 的加载与合法动作。文件或运行源码
遭修改时会拒绝校验。[完整验证范围](Validation.md)。这不是跨所有 Python、
PyTorch 版本或硬件的逐位一致性保证。

## 恢复和边界

已保存的训练 checkpoint 包括参数、优化器及 RNG；恢复原运行必须使用其配置和
对应源码快照，且原截止时间不会自动延长。新训练应另开运行目录并明确 warm-start。
冻结联赛恢复使用原 manifest、已复制模型和 `--resume`，不能替换种子或候选。
详细命令见 [运行手册](Runbook.md) 和 [评测协议](Evaluation.md)。

Jaipur 的罕见三重平手保留为规则未决；New World 的少数合法策略循环保留截断。
它们不产生虚构终局奖励。Splendor／Gizmos 使用已披露的规则版本 2 裁定。
Harmonies 普通 JSON 往返已验证；排序 JSON 的键可能改变候选动作顺序，因而可能
影响随机选取或并列动作的选择。Five Tribes 尚不适合作为无条件正式评测环境。

## 资源与退出核验

最终从 126 个启动任务的记录核算分配区间，所有任务都有有效的退出码 0。
其中 GPU 训练 41 项：37 项自然完成，4 个历史分支按计划提前保存并停止；退出码 0
不表示这 4 项跑满了原定训练步数。2026-10-31 15:57:44–15:57:45 UTC 对 8 个
已使用节点的只读检查确认，本项目没有剩余训练／评测进程或 tmux 会话。

| 资源口径 | 时长 |
| --- | ---: |
| 任务分配 CPU 核时之和 | 660.86 核时 |
| 任务分配 GPU 卡时之和 | 46.77 卡时 |
| 同一物理 GPU 上重叠区间合并后 | 42.84 卡时 |

这些是从启动到退出标记的**分配区间上界**，不等于 CPU/GPU 实际忙碌时间；
CPU 数量按 affinity 中的可调度 CPU ID 计，同卡多任务的任务卡时可以大于物理卡时。
最初五项 CEM 缺失的节点字段按已保存的
成功启动记录补回，只用于物理资源去重，不重复增加任务时长。工具调用、源码开发、
本地检查等零散开销没有逐进程计量，不能把此表当作整个项目的完整忙碌计算量。
所有远端启动遵守每台最多一半 CPU 的配额，最终核验未干预其他任务。
公开统计为 `outputs/v160/reports/final_public_resources_20261031T155332Z.json`，
退出证据摘要为同目录 `final_public_lifecycle_20261031T155332Z.json`；交付压缩包
内分别为 `resources.json`、`lifecycle.json`，不含节点标识或私有连接配置。
