# V0 review 处理说明

审查依据：[BoardBench_V0_Review.md](BoardBench_V0_Review.md)。
修补版本：V0.1，Python 包 `boardbench 0.1.1`。

三项意见均采纳；R1、R2 是已复现的正确性问题，R3 作为小范围接口修正一并完成。
原始 review 文档保持原样。

| 意见 | 判断与修改 | 验证 |
| --- | --- | --- |
| R1：恢复共享 Solver 文件 | 合理。每次恢复复制 solver 与 workspace，load 获得私有路径。评测每局独立副本；恢复适应的副本位于新 run。副本在 decide／end_episode 时仍存在。 | 加载时写 compiled_cache.bin；两局加载前均无缓存；恢复适应路径位于新 run；磁盘模型在后续回调中可读；原目录摘要不变且 checkpoint 可再次验证。 |
| R2：reset 失败账本缺失 | 合理。reset 纳入局级错误记录；started=false、score=null、reason=reset_error。失败尝试进入 attempted／failed 计数，未开始游戏不调用局回调。 | 适应记一次失败并停止；评测逐种子记录两次失败；无伪造转移、无 start_episode／end_episode。 |
| R3：RL 观测含 None | 合理，修正成本小，现在处理。保留 8 个槽，score 未知用 0、既有 terminated 标志标记有效性；支持 Integral 标量动作。 | array('f') 转换及有限性检查；未终局／终局字段核对；IntEnum 回归测试；已有 NumPy 1.26.4 环境下验证 float32[8]、int32／int64 与非法类型拒绝。 |

新增 `tests/test_review_regressions.py`，对审查版本先运行得到 **6 failed**，
修复后同一批测试 **6 passed**，完整测试 **40 passed**。
前后原始日志随新证据提供于 `runs/acceptance_v0_1/review_regressions_before.log`、
`review_regressions_after.log`。NumPy 检查使用现有基础环境，不增加项目运行依赖。

恢复副本的实现不包含每局源码重装，也不依赖权限封锁。模型自行保存仍必须导出完整
持久状态。reset 失败与成功开始后异常的统计口径分别保留，真实终局得分处理不变。

源码哈希校验保持启用。旧 `runs/acceptance_v0/`、旧 checkpoint 和旧 ZIP 保留，
新代码生成自己的 checkpoint 与证据；旧产物继续通过其自带的源码快照恢复。
最终验收状态、产物路径与已知边界以 [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) 为准。
