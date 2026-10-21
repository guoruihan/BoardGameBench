# V1.3.8 单人基线交付

2026-10-22；包版本0.4.23。冻结选择V137 control_a：
开发均分113.9766高于control_b的113.3477，未使用新验证集选模型。
保留所有旧策略，不宣称最优、人类高手水平或搜索上限。

## 交付与验证

- 推理产物：outputs/v138/solo_baseline/weights.pt和manifest.json。
- 权重SHA256：9799f3e8bbff77b01e4b2eec18debf4cd8f9c73358ff7e014dcc9e0b5cd1fc8b。
- 导出只包含FP64模型与显式架构/规划器元数据，不读取祖先checkpoint、训练数据或优化器。
- harmonies_baseline接入标准Solver及父进程3秒限时协议；持久规划器与原采样器逐步一致。
- 新512局验证，种子91000000–91000511，完整局均分**112.1797**。
  此面板与V137确认集不同，不用于选模型，不冒充原有封存正式测试集。
- 前32局独立单线程CPU、每局在线3秒：均分**113.8438**，32/32自然结束、0失败；
  在线平均0.454秒，单独统计启动平均1.939秒。这个较小子集的均分不能直接和512局比较。
- 原始结果：outputs/v138/solo_validation.json，包含逐局成绩、超时、失败、启动和在线耗时。
  总验证墙钟44.37秒；测试/开发成本另计，没有启动新单人训练。

## 使用

源码在Git，权重不随Git上传。需取得上面的产物，或通过交付包同时取得源码与权重。
Python3.10+，安装项目train依赖；本机实测PyTorch2.4.1、NumPy1.26。

~~~python
from boardbench.environments.harmonies import Harmonies
from boardbench.v138.baseline import BaselineSolver

env = Harmonies()
obs, _ = env.reset(123)
policy = BaselineSolver()
policy.load_weights("weights.pt")
policy.start_episode(obs)
while not obs["terminated"]:
    obs = env.step(policy.decide(obs)).observation
print(obs["score_breakdown"])
~~~

正式限时调用boardbench.v12.timed.run_timed，solver id为harmonies_baseline，
并提供weights路径及manifest中的sha256。不要把直接循环的墙钟当成限时协议结果。

本次不改变旧网页默认策略指针；可独立使用标准Solver，避免悄悄替换历史实验。
下一项已获用户授权：继续V1的双人Harmonies环境与RL基线，而非V2自主编码agent平台。
