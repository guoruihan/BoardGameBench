# Harmonies V1.2.9：原生编码采样与原检查点续训

2026-10-20，包版本0.4.13。用户授权将已验证的C++加速接入实际采样并继续训练，不从头重跑。

## 改动范围

- 把V128独立原型的C++编码器移入`boardbench.v129`包，显式编译并固定C++源码、动态库SHA256和布局版本；冻结源码包含C++文件，worker启动重新校验。编译器/动态库缺失或哈希不符直接失败，不悄悄退回慢路径。
- `native_v129`采样器每个worker一次编码本批仍存活的环境，复用原fast环境的状态转移、计分、合法动作、动作顺序、奖励和自然终局。公开信息仍先打包为161个int16，再生成相同1346维float32特征；不是完整C++游戏重写。
- 仅内部采样环境省略训练不消费的`reset_info()`静态规则说明深拷贝；对外Harmonies/FastHarmonies接口不变。step的观测/奖励/info契约、公开与私有信息边界不变。
- 模型、FP64数值策略、128局完整同步批次、PPO超参数、环境顺序均不变；没有异步旧策略数据、GPU环境重写或混合精度变更。
- 显式允许已安全停止的`interrupted`父任务迁移，必须开启`allow_interrupted_parent`、父编排已终止且有完整final检查点。校验父文件及源码哈希；`carry_stage_budget`逐任务扣除旧阶段已消耗时间，不借换源码重置4小时上限。

## 旧任务安全停止

原任务`outputs/v128/ppo_packed_a218_20261020_a`按用户本次授权停止。发送SIGTERM前核对外层PID791的完整命令行与run root；外层要求各worker在完整批次边界保存。结束后原外层消失，八个任务均有`interrupted`summary和校验通过的final.pt，GPU全部释放。停止receipt为原root下`native_switch_stop_20261020.json`。

|任务|保留的累计PPO动作|旧阶段已用秒|
|---|---:|---:|
|anchor_a|6,354,676|779.40|
|anchor_b|6,471,229|778.76|
|bc_a|6,339,689|779.81|
|bc_b|6,367,977|779.88|
|random_a|6,190,858|779.34|
|random_b|6,295,192|779.65|
|soft_a|6,319,132|779.58|
|soft_b|6,413,158|779.71|

新目标仍为累计1000万动作，不是再从0跑1000万。每任务剩余时间约13,620秒。旧模型、优化器、CPU/CUDA/NumPy随机状态、批次/局数/动作计数完整保留；不重置critic，不重做初始价值预热。新阶段曲线的initial是上述迁移策略，历史最佳检查点和早期曲线仍在旧root内，不覆盖。

旧V128账本最终为19.901684 CPU核时、0.601053物理GPU小时，含它自己的预检；不重算进新阶段。V127和此前独立原型费用继续各自保留。

## 验证设计

- CPU验证覆盖串行/多worker逐步数组一致、重复reset、STOP边界、所有卡牌/计数/地形编码、动态库来源拒绝、完整PPO下一更新/恢复一致，以及中断导入开关和剩余时间约束。
- 冻结GPU基准使用刚停下的soft_b成熟检查点，所有任务相同父模型/环境/RNG。两卡8任务，每任务7worker，65核总分配。
- 对照顺序fast/native/native/fast；每个任务预热后做保存->下一次更新->重新加载->同一次更新的逐位恢复检查，再计时12个PPO批次。跨全部任务/条件核对最终模型SHA256。
- 用实际动作吞吐与阶段时间决定切换，不以GPU利用率或原型编码加速比代替。基准root为`outputs/v129/native_benchmark_a218_20261020_a`，冻结源码哈希`450acde01b8afce0c47c16323d26c79a17e8a5571a0d686336ebc166d003e731`。

已完成的成熟模型两卡八任务对照，每组12批、857,920个合计动作：

|顺序|后端|合计动作/秒|平均批秒|环境/等待秒|
|---|---|---:|---:|---:|
|1|fast_v128|48,797|1.442|0.731|
|2|native_v129|61,344|1.141|0.426|
|3|native_v129重复|61,740|1.140|0.429|
|4|fast_v128重复|48,975|1.431|0.717|

两次均值比较，端到端PPO批次吞吐提高25.9%，环境阶段时间降低41.0%。所有32个进程的训练后模型SHA256一致：`ef23be26a5688dfb86eeec6c99e2a694cdcf2c8d042cb47aa2efc0148ffc00c1`，并且每个进程都通过真实CUDA保存/重载后的下一更新逐位比较。该吞吐窗口不包含启动/预热/恢复测试和退出；这些额外开销全计入资源账本。

基准合计8,774.261 CPU核秒，加138.637控制器秒，以及276.288物理GPU秒。新配置预检加上保守3,600 CPU核秒覆盖本地构建/回归/审查/打包验证，合计12,512.898 CPU核秒/276.288 GPU秒，不重复累计旧任务成本。

## 构建、运行和回退

原生库构建命令：

```bash
.venv-train/bin/python -m boardbench.v129.encoding --out outputs/v129/native_build_<new_id>
```

库单独保存于项目outputs，`artifact.json`固定路径、二进制/源码哈希、编译命令和版本。现有构建为`outputs/v129/native_build_20261020_a`；每个运行配置携带完整artifact，不能直接覆盖库文件。若迁移到不同CPU/操作系统须重新构建、验证并登记新artifact。

正式启动使用`boardbench.v127.launch`已有编排器及新配置；原生后端仅为显式选项，旧配置仍能运行原后端。长任务保持命名tmux和tee日志，前两张GPU每卡四任务，65核亲和性及共享节点CPU锁；不会使用其余六卡。

在a218项目根目录确认资源空闲后：

```bash
mkdir -p logs/v129
tmux new-session -d -s boardbench-v129-ppo "bash -o pipefail -c 'cd /datapool/data3/storage/ruihan/code/boardgame && PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src .venv-train/bin/python -u -m boardbench.v127.launch launch --repo /datapool/data3/storage/ruihan/code/boardgame --root outputs/v129/ppo_native_a218_20261020_a --config configs/v129_a218_native_ppo.json 2>&1 | tee logs/v129/ppo_native_a218_20261020_a.log'"
```

新root的`jobs.json`逐任务记录扣减后的剩余预算，`trials/*/config.json`固定各自父检查点及原生库。`curves.svg`/`REPORT.md`给出开发曲线；`state.json`给出物理GPU区间合并后的成本；`training.jsonl`可核对实际后端、更新与阶段计时。

恢复新的运行：同root执行`resume`，不用传新config；入口自动加载该root冻结源码与固定原生库。原库损坏/丢失须恢复哈希匹配的文件，不能绕过校验。

若新版本验证失败，可在确认新任务退出和资源空闲后，使用旧root的冻结源码恢复V128；其完整检查点未被修改。当前只在验证通过后启动新版本，不同时恢复新旧两套。

正式test/confirmation保持未运行。固定32开发局面用于曲线；另64开发局面历史已使用，只能叫开发对照，不能改称未见测试。

打包验证：运行环境自带setuptools58.1低于项目要求>=68，不能用该版本关闭隔离构建（产生的UNKNOWN wheel无效，未使用）。标准隔离构建成功产出`outputs/v129/wheel_check_isolated_20261020/boardbench-0.4.13-py3-none-any.whl`，检查包含`boardbench/v129/native/encode.cpp`和全部原生模块；运行环境未被升级。训练使用冻结源码，不依赖该wheel。
