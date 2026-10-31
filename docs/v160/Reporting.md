# V1.6 本地只读报告

`boardbench.families.report` 读取现有 CEM、PPO 和冻结 league 产物，输出 JSON。
它不启动或中止任务、不调用 SSH、不加载模型、不选择最佳策略，也不修改原始结果。
默认报告没有机器明细、绝对模型路径、原始命令或服务配置。

在仓库根目录运行：

```bash
PYTHONPATH=src .venv-train/bin/python -m boardbench.families.report \
  --root outputs/v160/bots \
  --output outputs/v160/reports/status_20261031T180000Z.json
```

文件名由调用者指定，建议使用当前时间。输出必须位于 bots 实验树以外，已有文件不会
被覆盖；省略 `--output` 则打印 JSON。`--include-host-aliases` 仅供本地诊断，允许输出
形如 `a226`、`b35` 的机器别名，IP、域名和其他 host 内容不会被复制。
两种报告都不应自动提交 Git。

只想快速查看任务进度、分配区间与冻结 league 时，可以添加 `--skip-training-records`。
这会跳过历史训练轨迹文件，将 `written_training` 设为 null，并明确标记
`training_records_scanned=false`，不会猜测或沿用旧的训练对局数。默认仍读取完整记录。

## 任务状态和已写入对局

任务来自 `bots/<game>/<run>/launch.json`，并补充没有 launch 的 state/manifest
目录。评测的 `<run>_job` 启动目录关联同级 `<run>` 数据目录；不跟随 launch 中的旧
绝对目录，因此移动整份产物后仍可以汇总。

`status` 是文件证据：

| 值 | 含义 |
| --- | --- |
| `completed` | 有有效退出码 0；存在旧退出记录或数据问题时会降为异常 |
| `activity_unverified` | 有启动记录但没有有效退出证据；**既不声称仍在运行，也不因 state 显示完成就声称已退出** |
| `artifacts_complete` | 没有启动记录，只有 state 或冻结计划的完成产物；不表示确认了进程状态 |
| `abnormal` | 非零退出、blocked/stopped 状态、超过 deadline 仍无退出记录、评测出现 unresolved/truncated/error 等结果、评测退出但计划未全部完成，或文件/数据校验异常 |
| `unknown` | 缺少启动与完成证据 |

报告不将 timeout、主动停止或未完成评测解释成算法失败；查看 `phase`、`exit_code`
和 `issues` 才能区分原因。训练的 `completed_updates`/`completed_generations` 原样来自
state 的 `next_update`/`next_generation`，不据此推算完成对局。
`process_liveness` 始终是 unverified，`status_evidence` 说明判断来自哪类文件。
退出标记早于本次 launch，或 state 更新晚于退出标记，会明确报警，避免恢复后误用
旧的退出文件。没有退出标记时，即使第一次读到 state=complete，也不会报任务完成。

`written_training` 只读取已有的 `bootstrap.json`、`g*_population.json`、
`g*_gate.json`、`rollout_*.json`、`gate_*.json` 和 `confirmation.json`。
每个可读结果行计一次，分别统计 complete、truncated、error 等状态与阶段。
配置中的计划 games、population、updates，summary 中的汇总数字，以及重复写入的
metrics，都不增加此计数。损坏文件与不合法行单独报警，不伪装为零错误。
被覆盖或未落盘的历史对局无法恢复；计数表示当前文件中的证据，不是环境总调用次数。

大型 trajectory 文件不会整份载入内存：超过 256 KiB 时，按现有原子 writer 的
`indent=2` 格式分块读取，仅识别顶层每局边界和 `status`，忽略嵌套轨迹里的同名字段。
未知格式、未闭合文件或缺失状态头会报警。`file_read_modes` 明确区分完整 JSON 解析
与流式记录头扫描；后者**没有重新校验整个嵌套轨迹 JSON**。小文件使用完整 JSON
解析。首次统计仍需顺序读取所有结果字节，几十 GiB 目录可能需要几分钟，但不会因
单个 GiB 级轨迹文件而一次性构造完整对象树。
为缓解共享文件系统大量小文件的打开延迟，训练结果与评测尝试最多使用 4 个只读线程，且不超过
当前进程可用 CPU 数量的一半（至少 1 个）；不会启动环境对局或训练进程。

## 冻结评测

报告验证 manifest 的内容哈希，读取 attempts 中每个 match 的最大 attempt 编号。
重复、未知 match 或非法 attempt 记录会报警。旧的重试仍计入实际已写尝试数，但只将
最新尝试用于胜率，同一局不会因重试被重复放大。尚无尝试的计划局保留为 not_started。

每次报告从冻结 schedule 和当前尝试重建统计，不覆盖原 `summary.json`。
`saved_summary_count_check` 只说明原 summary 的 manifest、计划数、尝试数和状态计数
是否一致；它不是对模型二进制的重新验证。`bots` 提供冻结身份和模型哈希，不提供
模型路径。报告不读取 `best.json` 作为选择标准。

每组对阵保留：

- 计划数、完成数、未完成数、胜/平/负，以及胜利记 1、平局记 0.5 的 win points。
- `completed_win_rate`：只对完成局计算，可能受完成选择偏差影响。
- `all_scheduled_win_rate`：win points 除以完整计划分母，是未完成结果未知时的下界；
  同时给出 `all_scheduled_win_rate_bounds` 的最好/最坏范围，**未完成不等于输掉**。
- 95% 描述性 bootstrap 区间：以环境 seed 为抽样单位，保留同 seed 的换边关系。
  默认 2,000 次，可用 `--bootstrap-samples` 调整；不作多重比较校正。
- 分数仅来自已完成局的终局官方分数。缺失的 race 分数仍是 null，不合成替代分数。

训练结果与冻结评测不是相同抽样总体，因此不生成混合总胜率。模型是否保持冻结文件
哈希由评测器负责；报告只展示记录的哈希，不重新加载/散列所有权重文件。

## 分配资源时长上界

每任务的时间区间从 `launch.launched_unix` 至 `exit_code` 文件 mtime；尚无有效退出
标记时延伸至报告时刻。该区间乘以 `cpu_ids` 中不同核的数量或分配 GPU 数量，分别
称为 **allocated CPU core-hours / GPU card-hours**。它们不表示 CPU busy time、GPU
实际利用率或显存占用，更不能由启动前采样的利用率乘出来。
如果长时间读取期间才观察到新的退出文件，其 mtime 晚于本次报告参考时刻，则在参考
时刻截断分配区间，并标记 `report_time_before_observed_exit`。

`task_allocated_*` 是任务区间之和；`unique_allocated_*` 在内部按机器和物理资源 ID
合并重叠区间，避免同卡多任务或相同 CPU 核被重复累计。公开报告只输出总量与缺项数，
不输出用于合并的机器/设备 ID。缺少分配、非法时间或退出记录会明确报告；有效部分的
小计不能被解释成完整总量。

暂停、退出文件写入延迟和恢复间隔会使分配区间高估实际忙碌时长。多次恢复、被覆盖的
launch、改动过的分配和缺失的早期记录不能从当前文件重建，因此这些数字仅对**已记录
区间**给出上界，不保证覆盖任务整个生命周期。`invocations` 反映现有评测调用记录数，
不能据此假定每次使用相同资源。

报告读取期间任务可以继续写入。跨文件快照不保证原子性；可检测的部分 JSON、缺失字段和
相互不一致的记录会被标记，刷新报告可以获得后续状态。不要把这种瞬时异常自动用于
停止任务或选择模型。

## 验证

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest tests/families/test_report.py -q
```

测试覆盖实际结果计数、部分文件、重试与完整分母、配对区间、冻结哈希、无效终局奖励、
进度/退出/截止时间、GPU 0、同卡与 CPU 重叠区间、恢复边界、公开信息过滤，以及
报告前后实验文件字节和 mtime 不变。
