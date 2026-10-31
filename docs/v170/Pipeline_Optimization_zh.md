# V1.7.0 / 0.6.2：采样与保存 pipeline 优化

用户要求暂停试跑后，32 路原任务全部停止，32 份 checkpoint 均能读取，权重为有限值，
优化器状态存在。原始数据、模型和代码冻结目录保留；没有自动重启正式训练。

暂停操作曾同时命中 learner 和外层 timeout 包装器：16 路正常退出，14 路出现
BrokenProcessPool 并保存 checkpoint，2 路被外层终止但周期 checkpoint 可用。
最后未提交的一批可能需要重跑。0.6.2 新增请求文件式暂停，避免向采样子进程和包装器发送信号。

## 找到的开销与修改

旧路径通过 `json.dump` 把大量细小片段逐次送入文本 gzip writer。对真实数据拆分计时后，
主要开销是 Python 序列化/调用链，而不是单纯的磁盘写入带宽。优化保留全部数据：

1. 使用 C encoder 的 `json.dumps` 生成字节，再整体压缩/原子写入，避免大量小写调用。
2. 每个 CPU actor 在对局完成后压缩完整轨迹，只把 PPO 所需数组、对齐元数据和压缩字节
   传回主进程，减少大型 observation 字典在进程间重复传输。父进程直接写入 gzip members。
3. 采样队列从“每一波都等待最慢对局”改成有界滚动提交。返回数据仍按原 job 顺序排列，
   因而不引入按完成顺序变化的 PPO shuffle 或 opponent statistics。
4. 固定初始面板结果和公开指纹局面只保存一次，checkpoint 用校验和引用；动态模型、
   optimizer、RNG 和历史池仍正常保存。每批只计算一次行为模型文件 hash。
5. 出错时保留上一份已提交 checkpoint，避免把半次 PPO 更新覆盖成可恢复状态。

轨迹格式升级为 `boardbench_full_trajectory_jsonl_v2`：标准 gzip JSONL，第一行 batch
元数据，随后每行完整 episode。合法动作、公开观察、log_prob、value、玩家/策略/对手
身份、回报及截断标记均保留。`read_episode_batch` 同时读取原 `.json.gz` 和新 `.jsonl.gz`。

## 实测结果

保存测试使用五份已生成的真实轨迹，每种方法重复两次，比较保存步骤中位数；解压后的
JSON 字节完全一致。纯写入约 0.04–0.06 秒，旧路径总计约 1.0–5.6 秒。

| 游戏 | 旧保存 / 秒 | 缓冲保存 / 秒 | 提速 |
|---|---:|---:|---:|
| Harmonies | 1.58 | 0.30 | 5.33× |
| Patchwork | 3.12 | 0.78 | 4.02× |
| Cascadia | 4.75 | 0.83 | 5.74× |
| Century A New World | 1.02 | 0.20 | 5.11× |
| Dominion | 5.59 | 0.80 | 6.98× |

完整批次对照使用同一 GPU、同样 6 个采样 worker、相同 24 局 seed/策略，运行完整 PPO
更新；两次重复交换方法执行顺序。下表是**采样＋归档＋PPO 更新**的平均耗时，
不包含启动预热、定期验证和 checkpoint。它不是一小时完整训练的提速保证。

| 游戏 | 旧路径 / 秒 | 优化路径 / 秒 | 提速 |
|---|---:|---:|---:|
| Harmonies | 3.82 | 1.62 | 2.36× |
| Patchwork | 4.46 | 1.35 | 3.30× |
| Cascadia | 9.47 | 4.39 | 2.16× |

每次对照都校验完整 learner 轨迹、回报、终局结果一致，更新后的模型 tensor 逐项完全相同。
分 episode 压缩文件略大，三个批次平均增幅约 0.15%–1.22%；没有通过删除演示数据换取速度。
这只验证了三个环境的批次速度，尚未重测全部 16 环境的长跑或不同 CPU/GPU 并发数。

完整 family 回归通过 525 项；增加恢复预算测试后的 20 项针对性检查通过。
另在实际 GPU 上验证三轮优化版完整训练入口、静态 checkpoint 恢复，以及一次正常的
文件式暂停（退出码 0）。这些小规模验证输出与正式策略目录分开，不作策略晋升依据。

测量归档在本地 `outputs/v170/pipeline_optimization_20261101/`，包含冻结源码 hash、
输入清单、逐次计时和内容一致性证据；机器和进程明细不提交 Git。

## 暂停与后续恢复

0.6.2 启动的任务可以通过以下命令协作暂停：

```bash
.venv-train/bin/python -m boardbench.families.pool_control pause "$RUN_DIR"
```

它写入请求文件，停止新增对局，并等待已提交对局/更新收尾；不能控制旧的 0.6.1
进程。不要用模糊的命令行匹配同时给 timeout 包装器和子进程发送信号。

解除暂停后，可用 `pool_resume` 根据已验证的暂停记录准备**新输出目录**，保留模型、
optimizer、RNG、历史池和原剩余预算。准备动作会启动新的剩余预算时钟，但不启动训练，
因此只应在确认恢复时、紧接启动前执行。原失败日志和 checkpoint 不覆盖。

```bash
.venv-train/bin/python -m boardbench.families.pool_resume \
  --pause-record "$PAUSE_RECORD" --game "$GAME" --arm B \
  --output "$NEW_OUTPUT" --source-commit "$SOURCE_COMMIT"
```

后续资源评估应基于优化后的 pipeline。旧版低 GPU 利用率部分来自序列化等待，
不能直接据此减少 CPU 或增加 GPU 共享任务数；需要恢复后测持续吞吐和任务间竞争。
