# Century: A New World：公开局面重复与可选 bot 防护

2026-10-31。配置和组件范围见 [游戏说明](games/century_a_new_world.md)。
本项改动仅调整显式启用的 bot 如何选动作，不修改规则、合法动作、官方计分、
终局条件、原始特征或 PPO/CEM 训练代码。旧策略 artifact 保留原行为。

## 原始诊断

`league_dev_r1` 有 3/384 条截断，`league_dev_r2` 有 12/2688 条截断；
本轮只使用开发数据，没有读取 confirmation。原评测上限为每局 120 秒／10000
次决策。所有 15 条记录都已从原 seed、先手、冻结权重和策略 RNG 重新复现；
不是正常长局，而是双方不再推进购卡的策略循环。

去掉仅增加的 `history`、`decision_index` 和双方 `turns` 后，循环首尾的全部
引擎状态及环境 RNG 相同。循环内线性策略的最优动作均唯一，神经策略使用
确定性 argmax，因此策略 RNG 的额外消耗不产生逃离循环的机会。逐步执行
资源、组件和阶段 invariant 检查没有失败。当前引擎、组件数据、神经推理代码
的文件散列与两个原始 dev manifest 相同。

最短 dev2 例子是 seed `33000035`，先手 `ppo_r1_last`、后手 `cem_r1`。
第 217→221 次决策形成四步循环：双方各有 10 个棕资源，轮流在 `B1_r2c1`
取得 1 个绿资源，然后立即把绿资源丢掉。占位被对方替换后，工人也回到原
配置。四张公开订单都要求至少 1 黄和 1 绿，因而没有购卡进展。

其余 dev2 循环周期为 5–366 步。主候选 `ppo_mc_r2` 与 `ppo_r1` 也会循环：
seed `33000043`、席位依次为 `ppo_r1 / ppo_mc_r2`，第 257→623 步重复。
所以不能仅把它归因于较弱对照策略。

特征表示也提供了一个解释：弃资源动作的 24 维特征只有“资源等级的负值”和
“颜色”两维非零，没有公开订单缺口或具体资源数量。生产评分先奖励收益和
订单进展，弃资源策略随后又倾向保留高等级资源，可能把刚得到的所需颜色
丢掉。这里没有直接修改特征，因为已有权重依赖原特征语义。

## 可选策略机制

[century_new_world_policy.py](../../src/boardbench/families/century_new_world_policy.py)
提供 `CenturyNewWorldRepeatPolicy(policy)`。通过
`"decision_guard": "century_new_world_public_repeat_v1"` 显式启用，当前保留版本
仅在 `phase="turn"` 工作；`discard` 和 `explore` 原样转发。
`make_policy` 与 portable bundle 只接受该 game/id 组合及原有 Targi 组合，
不接受任意字符串。`POLICY_FIELDS` 原来已包含 `decision_guard`，没有扩大
其余字段白名单。

指纹保留公开市场及牌堆顶／数量、奖励堆顶／数量、覆盖地点、占位、双方资源、
工人、已获组件、当前玩家、观察者、阶段和 `final_round`。只去掉顶层历史、
决策序号及玩家回合计数；不读取隐藏牌序、埋在奖励堆中的组件或环境 snapshot。

首次访问把完整合法动作序列原样交给底层策略，保留相同策略 RNG 行为。
再次访问时，找出该观察下尝试次数最少的合法动作，保持原顺序，将这个子集
交给底层策略决定。动作的次数只在成功返回子集内动作后增加。若所有动作
次数相同，则再次提供完整集合；不添加新动作、强制结束或伪造赢家。

候选子集变化可能影响神经策略的归一化和上下文评分，因此不能宣称结果等价于
“在原完整动作排名中跳过一个动作”。这是一项需要独立评估的策略变种；
也不保证消除所有循环或提高分数。

每个玩家每局应新建一个 wrapper。guard 的计数属于 bot 运行时记忆：仅恢复
环境 JSON snapshot 不会恢复这份记忆。若重用 wrapper，需要清空 guard 计数
并另行重置底层策略；从开局使用同样的公开观察和模型可重放计数过程。

## 验证与交付边界

只给准备参评的 `ppo_mc_r2` 与 `ppo_r1` 加 guard。旧 dev2 的 12 条截断中，
6 条包含这两个候选，另 6 条不包含它们，策略完全没变；后者保留已有循环证据，
没有浪费时间重复运行到 120 秒。

受影响的六局比较如下。两版本均最多 120 秒／10000 步；turn-only 先用
2000 步做诊断，对达到该较小步数上限的两局再以原上限独立重跑。表中截断
采用后一次原预算结果；仅 turn 的四次正常终局时长包含逐步 invariant 检查。
`MC` 表示 `ppo_mc_r2` best，`R1` 表示 `ppo_r1` best；`-last` 为对应 last
checkpoint。只给没有 `-last` 后缀的这两个候选加防护。

| seed | 席位 0 / 席位 1 | 仅 turn：状态／步数／秒 | turn+discard：状态／步数／秒 |
| --- | --- | --- | --- |
| 33000022 | R1 / R1-last | 完成／398／2.09 | 完成／951／9.71 |
| 33000034 | MC / R1-last | 完成／212／0.76 | 完成／780／6.67 |
| 33000034 | MC-last / R1 | 完成／588／4.06 | 完成／773／6.54 |
| 33000043 | MC-last / MC | 完成／930／9.36 | 完成／618／4.37 |
| 33000043 | R1 / MC | 截断／3427／120.02 | 截断／3421／120.04 |
| 33000043 | R1 / MC-last | 截断／3419／120.00 | 截断／3431／120.06 |

两个版本都使 **4/6** 个原截断案例自然结束。仅 turn 四局的官方终局分数依次
为 `51:59`、`58:53`、`40:51`、`50:53`；turn+discard 依次为 `50:59`、
`58:53`、`40:51`、`50:60`。未终局的分数不当作终局成绩。

加入 discard 没有提高完成覆盖，且前三局更慢，因此本提交保留更小的仅 turn
方案。turn+discard 只是本地实验，不在公开 factory／bundle 中注册。
两份代码的原预算两局测试都没有解决剩余截断，不能声称 guard 保证终局。
这批局面专门从失败记录中选取，也不能据此估计一般棋力或全开发集完成率。

包括初步 2000 步检查、两条原预算复跑、六条替代变种测试，累计游戏 worker
墙钟约 606.5 秒；这不是实测 CPU 秒。最多两个本地 CPU 进程并行，无 GPU
训练。67 项相关测试通过，其中专属测试 16 项。

[专属测试](../../tests/families/test_century_new_world_policy.py) 覆盖首次动作及
RNG 一致、最少尝试选择、相同次数回退、公开字段敏感性、隐藏牌序不变性、
非目标阶段转发、JSON 观察、episode 重置、错误动作、明确 id 与 game 的
校验，以及 heuristic／neural 两类 guarded bot 的冻结评测和 portable bundle
导出／加载。原 Targi guard 的行为和未知 id 拒绝规则保持。

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest -q \
  tests/families/test_century_new_world_policy.py \
  tests/families/test_targi_policy.py tests/families/test_century_a_new_world.py \
  tests/families/test_bundle.py tests/families/test_play.py
```

完整局面快照、逐局比较和新 dev 输入保存在独立 worktree 的忽略目录
`tmp/new_world_cycle_diagnostic/`，没有覆盖原实验。输入保留 random、heuristic、
CEM、原 `ppo_mc_r2`、原 `ppo_r1`，并增加两份显式 guarded spec；模型内容和
SHA256 与原策略相同，guard 字段构成新的策略身份。这里尚未执行最终选择。
