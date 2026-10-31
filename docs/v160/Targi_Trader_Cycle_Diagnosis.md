# Targi Trader 循环诊断与可选 bot 防护

2026-10-31。检查对象是 `outputs/v160/bots/targi/cem_r1/g000_population.json`
中的 generation 0、候选 7。这里讨论策略是否会结束交易，不据此宣称 bot 棋力提高。
规则覆盖和组件依据见 [Targi 配置说明](games/targi.md)。

## 结论与复现证据

两条 120 秒截断来自**合法交易形成的策略循环**，不是阶段推进错误。
原版 Trader 可以重复交换；取得 `trader_discount` 后，普通货物之间可以 1:1
交换。环境必须保留这些合法动作，也一直提供 `done`。候选的线性评分持续偏好
交换，没有选择结束本次交易。

| 原 JSON 行号（从 0 开始） | 环境 seed | 候选席位 | 原决策数 | 原耗时 |
| --- | --- | --- | --- | --- |
| 84 | 10100000 | 0 | 2590 | 120.045 秒 |
| 90 | 10100000 | 0 | 2610 | 120.074 秒 |

两行对应 baseline 和当时 best 两个对手标签；这时两个对手的权重相同，因此
是同一对局的重复运行，不能算成两个独立的失败环境。

第 11 轮出现以下循环。资源顺序是 `[枣, 盐, 胡椒, 黄金]`，决策序号从 0 开始：

| 决策 | 动作前资源 | 动作 | 动作后资源 |
| --- | --- | --- | --- |
| 222 | `[2, 0, 1, 2]` | 一份枣换一份盐 | `[1, 1, 1, 2]` |
| 223 | `[1, 1, 1, 2]` | 一份盐换一份枣 | `[2, 0, 1, 2]` |
| 224 | `[2, 0, 1, 2]` | 回到决策 222 的局面 | — |

去掉只追加记录的 `history` 和 `decision_index` 后，决策 222 与 224 的
**全部内部状态及环境 RNG 完全相同**；两次独立复跑都复现 period=2。
第 222 步中，候选给“枣换盐”的分数为 `1.38822764`，给 `done` 的分数为
`-1.50424041`。这支持评分选择导致循环的解释。

诊断使用的 Targi 引擎与原训练版本 `c949b657b95bcabcd58d24d22a85805b8e3e0cba`
相同，文件 SHA256 为
`ea2e9d763127c9cddda5648e4afd004aa7a25db267d150e7119794c1fadcb1f0`。
本次更改没有修改该引擎、组件数据或合法动作生成器。

## 可选策略防护

新模块 [targi_policy.py](../../src/boardbench/families/targi_policy.py) 提供
`TargiTraderAcyclicPolicy`，显式标识为 `targi_trader_acyclic_v1`：

1. 只在 `trader` 阶段工作，读取自己公开的资源和实际公开动作历史。
2. 倒序读取本次 Trader 激活末尾连续的交易记录，重建已访问资源组合；遇到
   其他动作、其他玩家或其他轮次就停止，避免跨激活错误约束策略。
3. 策略候选中去掉会回到本次已访问资源组合的交易，保留其他交易和 `done`。
4. 将余下候选交给原策略评分。环境仍允许所有原动作，也不增加游戏终局条件。

Trader 中交易只改变资源向量，返回原资源组合没有产生其他游戏收益。一条
完整交易序列可以去掉其中的循环，并保持最终资源组合；但给贪心策略增加这个
过滤器仍会改变其后续选择，不保证保持或提高得分，需要作为新策略重新评估。
防护不读取对手私牌、牌堆顺序或内部 snapshot，没有额外持久状态，JSON
恢复后的公开历史足以重建其行为。

这是独立、显式启用的 wrapper；本提交没有改共享 `make_policy` 或正在运行的
CEM。接入方应在新策略 artifact 中记录
`"decision_guard": "targi_trader_acyclic_v1"`，并显式实例化或在加载器中支持
该字段。**单独添加 metadata 字段不会自动启用 wrapper**；旧 artifact 不应
在无版本记录的情况下改变语义。例如：

```python
from boardbench.families.games.targi import Targi
from boardbench.families.play import FeaturePolicy, policy_seed, run_match
from boardbench.families.targi_policy import TargiTraderAcyclicPolicy, GUARD_ID

seed = 10100000
candidate_spec = {"kind": "linear", "weights": candidate_weights,
                  "decision_guard": GUARD_ID}
specs = [candidate_spec, {"kind": "heuristic"}]
policies = [
    TargiTraderAcyclicPolicy(FeaturePolicy(Targi, specs[0], policy_seed(seed, 0))),
    FeaturePolicy(Targi, specs[1], policy_seed(seed, 1)),
]
result = run_match("targi", specs, seed=seed, policies=policies, validate=True)
```

应分别报告是否完成、胜负和官方分数；不要把截断记录当作自然终局。防护也不是
通用的规则未决处理方法，不能替代每款游戏对真实规则边界的说明。

## 检查结果

- [回归测试](../../tests/families/test_targi_policy.py) 使用原候选 7 的 24 维权重，
  覆盖两步和三步循环、激活边界、普通 2:1 与黄金 3:1 交易、输入不变性、公开
  历史恢复、有限交易退出，以及原合法循环仍被环境允许。
- 原有 80 项规则测试加上新增 8 项，共 **88 项通过**。
- 原候选 7 的 12 条记录全部重新运行，仍使用原 seed、先手、席位和策略 RNG，
  只给该候选套防护，对手保持原样。**12/12 自然完成**。
- 两条原超时记录均为 259 步、约 1.49 秒自然结束，官方比分 `18:26`，
  候选输给对手。防护解决的是无法结束的问题，没有使这个候选成为强策略。
- 其余 10 条记录的自然终局结果和决策数与原记录完全一致。
- 这 12 次验证合计 17.22 秒，逐步执行 conservation validation；墙钟耗时
  依赖机器负载，不应作为跨机器速度基准，也没有涉及 GPU 或远程任务。

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest -q \
  tests/families/test_targi.py tests/families/test_targi_policy.py
```

原复现轨迹、循环快照和 guarded 逐局结果保留在开发 worktree 的忽略目录
`tmp/targi_cycle_diagnostic/`，没有提交生成数据。即使该本地目录不可用，仓库内
回归测试仍可独立复现合法循环和防护行为。
