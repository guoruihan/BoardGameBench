# 可搬移的 frozen bot pool

`boardbench.families.bundle` 把开发集已确定的一批 bot 复制成独立目录。
同一个游戏可以保留推荐策略、对照策略和研究候选，不会只留下一个 `best`。
导出器不读取评测成绩、不排名、不训练，也不根据 confirmation 改推荐项。
这是现有 `evaluate.freeze_bots`、`code_identity` 和 `play.make_policy` 的薄封装。

## 明确提供开发集选择

先准备原有格式的 `bots.json`。可以引用训练产物，也可以直接填写 frozen
dev manifest 中的策略对象；后者适合训练器已经继续更新 `best.json` 的情况。
引用模型时应指向 dev 目录中已冻结的模型，避免把后来训练的权重混进来。
每个导出的 bot 必须与指定 dev manifest 中**同名候选**的完整身份一致。
对 dev 之后变化的 `best.json`，导出会报错，不会静默采用新权重。

导出计划示例：

```json
{
  "games": {
    "machi_koro": {
      "bots": "machi_koro/bots.json",
      "dev_manifest": "machi_koro/league_dev/manifest.json",
      "status": "selected",
      "recommended": ["linear"],
      "research": ["neural_candidate"]
    },
    "five_tribes": {
      "bots": "five_tribes/bots.json",
      "dev_manifest": "five_tribes/league_dev/manifest.json",
      "status": "research",
      "recommended": [],
      "research": ["linear"]
    }
  }
}
```

这里只示范格式，不宣称这些名字已经胜出或 Five Tribes 已通过完整规则认证。
每个游戏至少提供两个 bot，与现有冻结评测接口一致。
计划里的路径相对于计划文件，`bots.json` 内的引用仍相对于定义它的 JSON。
`recommended`、`research` 必须显式填写已存在且不重复的名字，两者不能重叠。
没进入这两个名单的 bot 标记为 `retained`。游戏的 `selected/research` 状态由
调用者明确指定；游戏配置标识本身不代表规则没有未决边界。

```bash
PYTHONPATH=src .venv-train/bin/python -m boardbench.families.bundle export \
  --plan outputs/v160/pool_plan.json --output outputs/v160/frozen_pool_01
```

输出目录必须不存在。先在同一文件系统的临时目录完成复制和校验，成功后再
整体发布；失败会清理本次临时目录，不覆盖旧包。导出打印 manifest 的内容
SHA256，建议在外部交付记录中保留它。权重和导出目录不进入 Git。

## 包内内容与来源

```text
manifest.json
games/<game>/bots.json
games/<game>/bots/<name>.json
games/<game>/models/<sha256>.pt
games/<game>/dev_provenance.json
```

单 bot JSON 仅保留推理必需字段：策略类型、游戏/config/rules 身份、24维线性
权重或相对模型路径与 SHA256，以及神经策略的 `stochastic` 参数。
显式推理字段 `decision_guard="targi_trader_acyclic_v1"` 只允许用于 Targi，
原样保留；未知 guard 或用于其他游戏均拒绝，不会静默删掉改变策略行为。训练产物中
额外的路径、主机和服务配置不复制。模型复制真实字节，不创建符号链接，也不
跟随原训练目录的后续更新；同一游戏内相同模型按 SHA 去重。

manifest 包含每个文件的 SHA256 和字节数、每个 bot 的状态、原 dev 候选身份
与去除运行元数据后的推理身份、官方固定配置 ID、规则版本和源码内容身份。
`dev_provenance.json` 是脱敏后的来源摘要，保留原 dev manifest 文件 SHA256、
manifest 内容 SHA256、种子和候选身份；不复制含绝对机器路径的原始 manifest。
原始文件仍应保留在实验档案中，以便按 SHA 对照。

权重文件不改写，其内部训练元数据也不会自动清除，因此只能打包已经获准
交付的模型。当前 family 神经权重的元数据仅记录源码版本与训练步数。
本版不附源码归档：历史整包源码包含集群启动脚本，不能直接放进可分发包。
包记录源码内容 SHA、Git 版本、Python 和 NumPy/PyTorch 版本，源码单独从
对应的已发布版本获取。若导出时 `tracked_dirty=true`，Git SHA 不能替代实际
源码快照；正式交付应从已提交的版本执行。

## 验证和调用

```bash
PYTHONPATH=src .venv-train/bin/python -m boardbench.families.bundle verify \
  /path/to/moved_pool --sha256 <exported-manifest-content-sha256>
```

目录可以整体移动。读取不依赖当前工作目录，也不会回头读取原来的 `best.json`。
默认核对整个当前源码/data 内容身份，并始终核对游戏配置、规则版本、每个
JSON 和权重文件。多出文件、缺文件、符号链接、路径越界及内容变化都会拒绝。
必要时 `--allow-code-change` 只跳过完整源码 hash，配置、规则和包内校验仍保留；
使用它表示调用者接受源码变化，不构成同版本可复现声明。

```python
from boardbench.families.bundle import load_bundle, load_bot
from boardbench.families.registry import game_class

root = "/path/to/moved_pool"
metadata = load_bundle(root, expected_sha256="<pinned-content-sha256>")
bot = load_bot(root, "machi_koro", "linear", seed=123)
env = game_class("machi_koro")()
env.reset(seed=456)
action = bot.decide(env.observe(), env.legal_actions())
env.step(action)
```

`games/<game>/bots.json` 也可直接作为现有 `families.evaluate --bots` 的输入。
这样确认集评测仍会独立冻结同一批候选，不改变本包的推荐项。
`load_bot` 先校验整个包再实例化现有策略；把实例保留到整局结束，不要每一步
重新加载。运行端仍需安装匹配版本的 BoardGameBench 及对应依赖。

SHA 校验用于发现内容变化，不是数字签名。外部保存的 manifest hash 可以拒绝
别人同时替换文件和重算 manifest；没有外部可信 hash 时不提供身份认证。

## 验证范围

`tests/families/test_bundle.py` 使用真实小型神经权重，验证多游戏状态、目录搬移
与原文件删除后推理、任意 CWD、原评测器兼容性、字段脱敏、模型/JSON/manifest
篡改与符号链接拒绝、确认集来源拒绝、动态 `best.json` 不可替换 dev 身份、源码
和规则不匹配拒绝、导出失败不留下半成品。
