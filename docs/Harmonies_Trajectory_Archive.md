# Harmonies 轨迹归档与训练读取

保存所有实际对局，不只保存最高分专家。原始日志不删除；额外压缩副本、
来源索引和严格训练读取器用于后续模仿学习、离线到在线 RL、价值学习和示范状态课程。
归档并不意味着所有记录都可以训练，也不代表新策略已开始训练。

## 保存范围

- `outputs/**/episodes/*/decisions.jsonl`：逐步实际动作、公开后继状态及已有的搜索候选统计。
  较新版本已保存动作前状态和原始分数增量。
- 从 0.4.16 起，共用对局记录器默认保存公开动作前状态，并显式记录环境奖励、
  `terminated`、`truncated`。策略行为概率未知时记录 `null`，不由候选分数推算。
- 每局的 `job.json`、`result.json`，每个来源运行的计划、划分、资源及源码标识；
  模型路径/哈希由原始配置保留。代码快照和模型文件仍由原运行目录持有。
- 历史 `semantic_trajectories.jsonl` 原样压缩为 `legacy_raw_only`；
  其中可能含私有末局快照，严格读取器禁止将该格式直接用作模型输入。
- 保留低分、失败、不完整记录；复制中的活动运行跳过并列入 `summary.json`。
  复制的源码/检查点树不重复扫描。

这不包含搜索内部每条模拟分支，也不恢复过去未保存的 PPO 采样轨迹。
内部 rollout 的逐分支保存应另设受控采样与存储预算，明确模拟种子和后续策略。
现有分支候选统计保留，但它们不是实际环境奖励，也不是最优 Q。

## 归档和恢复

项目根目录执行：

```bash
.venv/bin/python -m boardbench.artifacts.trajectories \
  --source outputs \
  --out outputs/trajectory_archive/harmonies_20261020_a \
  --workers 4

.venv/bin/python -m boardbench.artifacts.trajectories \
  --out outputs/trajectory_archive/harmonies_20261020_a --verify
```

再次执行同一归档命令会续接并去重，适合每轮数据采集结束后运行。
实际对局原始日志仍由现有 runner 自动保存；压缩索引需执行上述命令刷新。
同一目标有进程锁；退出后自动释放。失败/修复后的元数据修订保留为不同 entry，
相同原始字节共用一个 gzip blob，不能把修订条目重复计作独立环境。

目录结构：

```text
archive.json          归档格式及范围
blobs/<sha>.jsonl.gz   与原日志逐字节一致的压缩内容
campaigns/<id>.json   来源计划、split、源码标识和运行状态
entries/<id>.json     每份来源日志与结果的不可变索引
index.jsonl           汇总索引，可重新生成
summary.json          数量、字节、分类和所有跳过项
```

所有条目和来源元数据有内容哈希。`--verify` 检查归档字节和元数据，
**不等于逐局引擎重放**。`train_candidate` 只表示可申请训练读取，
最终资格由下面的完整重放检查决定。归档是项目内副本，不是异地备份。

## 从归档读取训练 transition

```python
import json
from pathlib import Path
from boardbench.artifacts.trajectories import read_train_episode

root = Path("outputs/trajectory_archive/harmonies_20261020_a")
validation_seeds = [...]  # 本次训练按环境划出的全部内部验证 seed
for line in (root / "index.jsonl").open():
    entry = json.loads(line)
    if not entry["train_candidate"] or entry["behavior"] not in {"search", "learned_search", "heuristic"}:
        continue
    if entry["job"]["environment_seed"] in validation_seeds:
        continue
    episode = read_train_episode(
        root, entry["id"], excluded_seeds=validation_seeds, expert_only=True,
    )
    # 整局验证完成才返回；按 episode/environment 分组取序列，不随机拆相邻行。
```

读取器核对 split 原始哈希、环境归属、完整终局、每步合法动作、
公开前后状态、奖励及最终得分。旧记录缺少动作前状态时，通过种子和实际动作
精确重放恢复，若和保存的后继状态不符则拒绝，不凭空补造。
引擎规则版本不兼容的历史数据需要使用其来源代码进行单独迁移。

输出包含 `observation/action/next_observation`、公开合法动作、
真实 `environment_reward`、`raw_score_delta`、分数增量 return-to-go、
终止标记，以及单独的 `expert_action/search`。seed 和源码标识只用于来源/分组，
不作为 observation 特征；不包含隐藏供给和 RNG 状态。

## 学习使用边界

1. **划分**：只有 manifest 验证的 train 可以训练；dev/confirmation/test 只作隔离归档。
   train 内部 validation 由调用方显式排除；所有变种/重复按环境 seed 一起划分。
2. **行为**：`dagger` 是学生实际轨迹加教师标签，不是教师控制的轨迹。
   只有显式 `expert_only=False` 才允许读取学生数据；专家建议不能替换实际动作后仍沿用原后继状态。
3. **回报**：默认提供真实环境回报和原始分数增量，不混入搜索估值。
   候选值依赖 rollout 后续策略，不是学生当前价值或最优 Q。
4. **概率**：历史未记录的行为概率保持未知；无法直接声称支持精确重要性采样、
   V-trace/Retrace 或基于行为概率的离线评估。适合先用不要求已知行为概率的方法。
5. **课程重置**：示范公开局面可以用来重置训练；随机未来应按当时公开信息重采样，
   不能把重放时为了核对证据而恢复的真实隐藏未来交给策略或规划器。
6. **质量**：保存全部不等于平均模仿全部。后续按真实回报、行为来源、完整性和覆盖度
   选择训练用途，不只看总分最高的少数轨迹。
