# V1.2.4 执行批次 b：保留16物理核的无重复扩容

2026-10-20用户授权：b35保留16个CPU，其余可用于当前实验。本次保守解释为**16个物理核**，因此本项目使用最多112/128物理核，不把SMT线程重复算作核。每局仍单线程，三策略、候选、样本规则、环境/搜索种子和08:00 CST截止均不变；不增加实验局数、不使用GPU。

## 为什么独立执行批次

旧运行`outputs/v124/fullgame_continuations_20261020_a`使用冻结源码，不能改它的配置或代码热加worker。新的`fullgame_continuations_20261020_b`只接管原1056项中尚未启动的项，worker仍从旧运行的`source/src`导入策略，策略源码摘要保持：

`4a4f21b595416ee0b290698e2c05e8300a0e664bea87d747a9f0f3f48cae2808`

扩容实现独立在`scripts/expand_v124.py`，启动时另行冻结为b运行的`driver.py`并记录哈希/Git提交。只改变任务派发和CPU资源，不修改策略源码或重新做策略选择。

## 交接协议

1. 验证旧控制器、旧watchdog和worker的UID、命令、PID启动时间、CPU绑定、输出根及冻结源码；旧运行必须仍在运行。
2. 只有所有旧worker槽位都被真实存活子进程占满，且`已派发job文件集合 = 已结算集合 ∪ 活跃集合`时，暂停旧**控制器**派发；不暂停worker，不暂停旧watchdog。不满足则立刻恢复控制器并重试，超时失败也保持旧任务运行。
3. 冻结交接清单：旧已派发项归旧运行，新运行仅获得余集。每次启动新worker前再次检查旧目录没有对应已派发job，发现冲突即停止派发，禁止重复执行。
4. 初始保留旧64核，新增48核含1控制核及47worker，共110worker＋2控制核。旧worker逐个自然退出后，新worker才能借用其核；这些核仍由旧预留账单计费，不重复计费。
5. 所有旧worker结束后，向已核验旧控制器依次发送TERM/CONT，让它结算并退出，不再派发。旧watchdog退出、释放原CPU锁后，新watchdog接管锁直至整个新批次清理结束，最后也接管旧控制核，达到111worker＋1控制核。
6. 新旧watchdog均保持原绝对截止；旧控制器暂停期间，独立的新watchdog继续执行旧worker原始单局超时，不重置计时。新控制器即使SIGSTOP，watchdog仍能清理其专属进程树并收尾旧批次。

若在派发新任务之前交接失败，恢复旧控制器继续原队列；派发后不能盲目恢复旧队列，否则重复执行。后者保留清单/失败/结果，旧worker按原时限完成或超时，新控制器停止派发，由watchdog收尾。

## 报告和成本

- b的`REPORT.md`/`summary.json`合并原1056项，明确标记**混合并发负载**，不声称固定并发条件；a保持原始轨迹与结果，`expansion_link.json`指向新运行。
- 在交接时冻结“完全未启动任何方法/重复的环境”集合，单独生成`fresh_only/REPORT.md`。其前缀为8/16/该集合全部环境（不超过集合大小），仅按任务归属选择，不按分数筛选。仍是开发集，不是新测试集。
- CPU核时按旧64核预留、新增48核预留、旧退出后的112核预留分段记账；旧核被新worker借用时不重复收费。继承旧全部预检/校准费用，另记本次扩容验证费用。GPU为0。
- `origin.json`绑定旧运行、原计划/jobs摘要、冻结driver/策略摘要、16核保留约束、截止和新预检费用；`handoff.json`固定PID启动身份/任务归属；`borrowed_cores.json`和`resource_transfer.json`保存资源转移证据。

## 启动

先通过生命周期测试并发布变更，再在b35项目目录运行；不要对已经完成或已经交接的旧运行重复执行：

```bash
tmux new-session -d -s boardbench-v124-expanded \
  'bash -o pipefail -c "PYTHONPATH=outputs/v124/fullgame_continuations_20261020_a/source/src PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/expand_v124.py launch --parent outputs/v124/fullgame_continuations_20261020_a --root outputs/v124/fullgame_continuations_20261020_b --reserve-physical 16 --preflight-core-seconds 8640 2>&1 | taskset -c 64 tee logs/v124/fullgame_continuations_20261020_b.log"'
```

8640核秒是本次本地实现/验证的保守上界（00:57–01:15 CST，8核），具体见`outputs/v124/expansion_preflight_20261020.json`；不是worker实际CPU耗时。若准备超过该窗口，须在启动前更新费用及命令。

## 验证

最终driver的4项检查全部通过（63.19秒），包括真实旧worker自然结束、额外核启动、不重复任务归属、逐核借用/最终CPU锁转移，以及故意SIGSTOP新控制器后的独立截止清理。既有策略/报告10项相关检查另行通过。冻结策略源码哈希没有改变；正式扩容前先记录版本并实时复核资源。
