# 兼容性决定

## V1 决定

- 保留 V0.1 恢复隔离、reset 失败记录和有限数值 RiskRL 的修复；不依交接包旧基线倒退。
- Harmonies 固定单人 A 面/base32/120 枚资源，末袋无法补齐九枚时结束；这条是项目裁定。
- 三款引擎、数值适配、搜索与网页共用规则；启发式和候选裁剪只存在 Solver 层。
- 采用同一 masked PPO 方法，固定 MLP + 任务专用数值编码/动作空间；gamma=lambda=1，
  显式 score-delta shaping，仅训练使用。没有神经网络输出之外的“纠正策略”伪装成 RL。
- 验证候选保留旧进度，测试在选择冻结后运行；单种子 pilot 如实报告，难度三级未宣称完成。
- 静态 HTML/CSS/SVG/JS + 轻量 Python 服务，不引入浏览器内模型推理或外部美术依赖。
- 模型采用 CPU 常驻推理以减少小网络设备往返，GPU 用于三款游戏的独立训练。
- 完整训练断点续训不是当前 CLI 能力；公开承诺的是权重/策略恢复与精确游戏存档。

## V0 历史决定

依据：实施规范 v0.3 与用户确认的范围。本文件记录实际行为。

1. **动作错误**：引擎抛出 `InvalidAction`，必须保持状态与 RNG 不变。
   Runner 先计一次真实动作请求，记录 `action_error`，允许重新决定；不构造假转移。
   自然终止后再次 step 也报此错误，必须显式 reset。

2. **随机流**：使用私有 `random.Random`。根种子通过 SHA-256 标签派生环境局种子和
   Solver 种子；不使用 Python hash 或全局 RNG。搜索 `fork(sim_seed)` 复制局面计数，
   创建新的模拟 RNG。具体 Python 版本记录在产物中；不承诺跨不同 Python 实现的逐位相同。

3. **时间与收尾**：所有 Solver 钩子前后检查墙钟；普通 Python 代码采用协作式超时。
   动作返回后若已经超时，记录 `action_not_executed`，不执行动作。
   已发生的转移仍发送一次；`end_episode` 仍调用一次，关闭新的 consult/job。
   已自然终止的真实得分不因收尾超时或异常丢失，异常单独报告。

4. **额度与事件**：非法动作占动作请求额度，模型失败占已发送次数，拒绝单独记录。
   模型包装层统一计量，mock 用量未知记 null。Solver 自报数据只进 `solver_report`。
   参考适配器统计模拟步数；第三方自报模拟量不能覆盖此账本。
   决策总耗时来自 `decision_end`，其中可能包含模型和 job 时间。

5. **恢复契约**：采用周期局末 checkpoint；manifest 绑定实际源码和产物哈希。
   源码快照包含 `pyproject.toml` 和整个 `src/boardbench/`；Solver 额外代码／提示应写在
   workspace，随其一并快照。跨任务扩展也应放入该源码包，以便完整快照。
   先将源码快照复制到独立安装目录，再通过 `pip install -e` 明确指定版本；
   避免安装器生成的 egg-info 改动原 checkpoint。不执行 pickle 或自动多版本加载。
   新 run 开始新游戏，旧局内记忆清空；参数、任务经验与 Solver RNG 保留。
   `.partial` 保存目录不可恢复，已有目标目录不覆盖。
   V0.1 起，恢复同时复制 solver 与 workspace，load 只获得私有 solver 路径。
   适应 run 的副本保留在新 run/restored_solver；不进入 workspace 的重复快照。

6. **评测隔离**：每个种子重新构造 Solver，start_task→load→start_episode；恢复相同的
   checkpoint RNG，而不额外按评测局号修改它。同一环境种子应产生同样行为。
   每局有独立预算（含初始化和 load），工作副本使用后删除，原 checkpoint 只读。
   工作副本覆盖 solver 文件和 workspace；即使 load 写缓存，后续局也不可见。
   副本保持到该局所有 Solver 回调结束，支持 load 后持续读取的磁盘模型。
   配置不继承适应 run 的剩余额度，也不隐式继承适应 run 的全局时间上限。
   V0 没有额外整个评测批次的总时间预算。

7. **失败与汇总**：自然终止由 `info.score` 提供真实得分，失败牌属于正常完成的游戏。
   未完成局 score=null。任务定义负责汇总缺失局的计分方式，Runner 不硬编码风险采集字段。
   初始化失败／超时可有评测尝试记录，但不计为已经开始游戏；未开始局仍进入请求局分母。
   V0.1 起，适应与评测的 reset 报错都记为一次未开始的失败尝试（reset_error），
   不对 reset 未成功的局调用 end_episode。适应停止，评测可继续其他独立种子。

8. **历史成本**：run 记录本次真实成本；checkpoint 同时提供本次保存前成本及从父保存点
   累计的适应投入。当前保存操作自身耗时在 run 事件和总耗时中，不倒填修改旧 checkpoint。
   独立评测分别报告历史适应资源和新发生的在线资源。

9. **UI**：标准库 HTTPServer、本地单会话、固定 loopback 监听。UI 不参与 headless run；
   回放不导入 Solver、不实例化引擎或重新采样。UI 不包含完整前端平台或权限体系。

10. **RL 数值编码（V0.1）**：保持 8 槽形状，未知 score 编为 0，复用 terminated_int
    作为有效性标志，终止且零分与未终止可区分。原始观测和 info 不变；不新增 NumPy
    或 RL 依赖。动作接收 Integral 标量并转 int，拒绝 bool、float、数组和越界值。
