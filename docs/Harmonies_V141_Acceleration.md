# V1.4.1 双人采样加速

2026-10-22；包版本0.4.25。仅执行优化，不更换学习方法、环境或奖励。
V140双人策略参数键、244动作/4865特征、FP64、PPO超参数和对手配比保持不变。
单人训练/推理实现不动；旧运行仍有自己的冻结源码。

## 实现与语义边界

- 原双人forward的主干7次调用：己方两个市场视图各3次、对手1次。
  现在三个视图各编码一次，合为一个3B批次；所有头共享带梯度的结果。
- 两个市场视图的actor、critic、adapter、gate和advice合批；没有跨参数更新缓存，
  没有detach学习表示，也不改用FP32或删除对手公开信息。
- 按模型对象分组，不按角色名分组。learner/current共用一个forward，冻结/历史对手仍独立。
  整步特征/mask一次上传，所有组action/logp/value统一取回；日志仍保留原角色。
- PPO只使用学习方轨迹，对手轨迹保留但不混入损失。采样、评测都不新增搜索。
- 同版本精确恢复继续要求下一批轨迹/权重逐位相等。跨版本合批会改变随机数消费顺序，
  因此不承诺与旧版本随机训练路径逐位一致；验证的是函数/梯度和行为概率一致。

## 检查与实测

本地跨单人/双人的相关回归81项通过/17项明确CUDA跳过，包含完整状态迁移后真实更新。
真实A100小门控使用已训练V140权重：score/margin的logits最大差3.55e-15，
梯度最大差7.29e-17；8场成对贪心完整轨迹相同。
同卡存在原训练时，小批采样6.014秒→2.766秒，约2.17倍；这不是最终128场独占测速。
证据：outputs/v141/small_gate_20261022_a/result.json；额外4核，仍在半机CPU上限内。

替换训练必须先通过4卡实际PPO更新/精确恢复门控，再通过128场、13worker、
相同权重与种子的旧/新采样对比。预热两条路径，3组交替顺序配对，要求完整贪心
轨迹一致且中位采样耗时至少加速1.2倍；结果写入新run的speed_gate/result.json。
采样加速不是得分提升，亦不是同资源单人/双人的严格比较。

## 接续、预算与恢复

父run：outputs/v140/duo_a226_20261022_a。先向逐一核实过身份的四个trainer发送SIGTERM，
让它们完成当前批次并保存latest/final及interrupted.json；不杀环境worker或覆盖父run。
等父控制器退出释放锁，使用新目录和新tmux运行：

```bash
PYTHONPATH=src PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-train/bin/python -u -m boardbench.v140.campaign \
  --repo /datapool/data3/storage/ruihan/code/boardgame \
  --out outputs/v141/duo_a226_20261022_a \
  --config configs/v140_a226_duo.json \
  --continue-from outputs/v140/duo_a226_20261022_a
```

仍是a226 GPU0–3，各13worker+1learner，控制器合计57物理核；不使用其他卡。
迁移核对父latest/best摘要，只更新执行来源配置；模型、Adam、冻结/历史对手、RNG、
学习计数、累计时间与best选择全部保留。父轨迹不复制/删除，新轨迹使用原批次编号续写。
原截止2026-10-22 07:27:07北京时间不延长；训练预留确认评测时间仍1800秒。
新cost.json引用父campaign，必须连同父cost.json看总消耗；小门控与原GPU占用重叠，
不能再将其GPU时间作为不同物理卡重复累加。

若替换门控失败，新训练不会启动；旧冻结源码/检查点可按原config的--resume恢复，
但仍须遵守原截止时间且先核对资源。新版本中断后同样用该run的source/src和config
运行boardbench.v141.training --resume；不得跳过来源校验或重新初始化优化器。

本节不表示最终训练已完成；运行进度看status.json，完成与确认成绩看done.json/confirmation.json。
