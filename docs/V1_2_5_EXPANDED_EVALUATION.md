# Harmonies V1.2.5：保留策略扩测

用户在三轮自主探索结束后批准扩大环境数量，与此前结果对照。本任务只评测冻结策略，不继续调参；不是正式confirmation/test。

启动快照（2026-10-20 05:24 CST）：已在b35启动480局，源码提交`4bc0d94`，100worker/101预留核。100个2秒训练校准全部自然终局；完整CPU回归299通过、21资源条件跳过，另在b35运行生命周期与门禁检查26通过。实际worker核绑定全部匹配、线程数均1、100局均有首步进展。预计06:40–06:55完成，硬截止07:24:04。**这是启动时快照，最新状态和结果以运行目录REPORT/status为准，不代表已跑完。**

## 冻结协议

- 32个development环境，索引64–95；与V1.2.4原900秒评测完全相同，与本阶段选型用的96–107不重叠。它们仍是曾用过的开发环境，不冒称历史未见测试。
- 每环境3个搜索seed：12420261020、12420261021、12420261022，与旧评测完全相同；不是3次训练。
- 5个保留策略：original、habitat、mixed_roots、mixed_compiled、mixed_confident。共480局，全部每局900秒累计solver墙钟。
- 不重跑明显未改善的宽候选/后期切换等变种；保留compiled来区分等价加速与不确定性选择的作用。
- 配置：`configs/v125_expanded_eval.json`；预注册8/16/32环境完整面板，不能仅按先完成的容易局宣布结果。

主要比较是**本轮同批重跑对照**。历史original104.667、space107.531、habitat110.323另列；旧实验扩容前后负载不同，历史差值不冒称同批受控因果效果。原有v125策略及规则文件与第三轮快照逐文件hash一致，改动只有调度资源配置和报告。

## 历史来源与统计

旧批次`outputs/v124/fullgame_continuations_20261020_b`的resolved只含扩容后任务，前段来自`origin.parent`。快照工具合并两个已结束批次、拒绝重复身份/缺失任务，核验旧冻结源码和每个job参数/种子/真实终局，保存288条900秒参考：

```bash
PYTHONPATH=src .venv/bin/python -m boardbench.v125.historical \
  --run outputs/v124/fullgame_continuations_20261020_b \
  --out outputs/v125/references/v124_900s_20261020.json
```

文件不可覆盖。快照SHA256为`74bee2b43dcb2f3c2bafb7dba414577b0be750292b22ab9e4cda39284130f9ce`，启动时连同内容嵌入新的冻结plan，报告不依赖历史文件之后是否变化。

先在每个环境平均3次搜索，再对32个环境求均分、标准差、配对差值和描述性95%区间，不把96局当96个独立环境。总分、动物/地形、放动物数、完成卡、搜索seed均分波动、实际耗时、失败和保守成本继续保留。固定规模，中途不按分数停止或更换策略；不作多重比较校正后的显著性承诺。

## 资源与执行

b35已有最多112/128物理核权限，必须保留16核。显式`resource_profile=b35_reserve16_v1`只允许在`b35.cluster.helixon.com`启用，上限`min(112,可用物理核−16)`；其它主机/未知profile拒绝。默认和旧后端继续限制一半物理核，未全局放宽。

本任务101核=100单线程worker+1控制核，保留27物理核；不把SMT当额外物理核。约120决策核时，预计75–90分钟；任务独立硬截止7200秒，单局watchdog1020秒，最坏202预留核时另加预检成本。GPU0。共享CPU锁、固定核绑定、源码冻结、任务身份检查、独立watchdog与恢复机制不变。

启动前用`configs/v125_expanded_calibration.json`在4个train环境×5个校准搜索seed×5策略上启动100个2秒任务，检验真实并发/合法终局/清理，不用短时成绩筛掉900秒方法。

正式运行命名tmux为`boardbench-v125-expanded`，日志`logs/v125/expanded_20261020_a.log`。从项目根启动：

```bash
tmux new-session -d -s boardbench-v125-expanded \
  'bash -o pipefail -c "cd /datapool/data3/storage/ruihan/code/boardgame && PYTHONPATH=src .venv/bin/python -u -m boardbench.v125.campaign launch --repo /datapool/data3/storage/ruihan/code/boardgame --root /datapool/data3/storage/ruihan/code/boardgame/outputs/v125/expanded_20261020_a --plan configs/v125_expanded_eval.json 2>&1 | tee logs/v125/expanded_20261020_a.log"'
```

启动前须资源预检并声明实现/验证费用；实际启动记录以allocation.json与账本为准，不重复使用已存在的root。上述命令示意不包含实际预检费用参数。

报告自动刷新：`outputs/v125/expanded_20261020_a/REPORT.md`、`summary.json`、`curve.csv`。历史比较单独列于报告末尾和`summary.historical_comparison`。所有策略/所有重复都齐全才进入预注册面板；预计第一个8环境完整面板约30分钟出现，最终约75–90分钟。

恢复复用[V1.2.5说明](V1_2_5_AUTONOMOUS_ITERATION.md)中的`campaign resume`入口，必须等旧watchdog收尾，使用新输出目录。已完成局复用、未完成局从头重试，历史快照留在冻结plan，父费用不清零。报告自动生成不等于代理离线后会自行修改策略；本任务无需下一轮决策即可跑完固定矩阵。
