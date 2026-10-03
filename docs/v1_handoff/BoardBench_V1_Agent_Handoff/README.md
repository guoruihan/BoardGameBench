# BoardBench V1：本地实现交付包

版本：2026-10-03 / v1.0。目标：在现有 BoardBench V0 项目上实现三个完整的单人游戏环境、网页互动、人工设计的 RL／搜索方法和经过评测的策略产物。

**给本地 coding agent 的入口是 `IMPLEMENT_PROMPT.md`。** 把本目录放在已有项目内，或把它作为项目的旁边目录，并告知项目路径。不要用本包覆盖整个现有仓库。

## 本轮范围已固定

| 任务 | 首次交付配置 |
|---|---|
| Micro Tiles | 自定义 3×3，三色，二选一，九次放置；相邻 +1、同色整行／列 +3 |
| Take It Easy! | 标准单人、19格、固定朝向、27种牌无放回供给 |
| Harmonies | 官方单人供给、A面23格、完整32张基础动物牌、关闭自然之灵 |

Harmonies 单人是三组各三枚资源，回合末未选六枚弃置；与双人五组、只补选走一组的规则不同。末袋不足九枚时，按本项目实施裁定在完成当前回合后结束，不再开下一回合。

每个任务完成：正确引擎 → RL／搜索适配和人类可玩网页 → 人工设计 baseline 与真实训练 → 策略演示、提示、接管、同供给比较。每个任务最多3–5类方法。

本地 coding agent 是开发助手。让参赛 agent 自己发明、训练和调用模块的研究评测属于 V2，不是本轮要加入的产品功能。V1 不需要 LLM API key，不增加真实远程模型网关。

## 阅读顺序

1. `IMPLEMENT_PROMPT.md`：任务指令、执行顺序、汇报要求。
2. `docs/01_V1_SCOPE_AND_ARCHITECTURE.md`：共同约定与模块边界。
3. `docs/02_V0_INTEGRATION.md`：依据实际 V0 源码整理的增量改造。
4. `docs/03_MICRO_AND_TAKE_IT_EASY.md`、`docs/04_HARMONIES_SOLO.md`：具体规则与任务接口。
5. `docs/05_BASELINES_UI_ACCEPTANCE.md`：训练、网页、验收与成果要求。
6. `data/harmonies/animal_cards.base32.v1.json`：完整基础动物牌数据。

`reference/` 包含来源、计分交叉核对和末袋构造；它不是需要照搬的游戏实现。冲突时，以本包具体任务规范和最新用户指示为准；有可证实的规则错误应记录并修正，不能无声改规则迎合 baseline。

## 可直接使用的内容

- `data/harmonies/animal_cards.base32.v1.json`：32张牌的坐标图案、落点、高度、数量和累计分值。
- `data/scoring_examples.json`：独立给定的 Micro／Take It Easy! 计分样例。
- `reference/bag_exhaustion_witness.json`：13回合后仍有10空格的单人合法构造。
- `scripts/validate_handoff.py`：仅用 Python 标准库检查包内数据和文件完整性；不会运行或声称验证尚未实现的引擎。
- `PACKAGE_MANIFEST.json`：交付文件哈希与数量。

解压后可运行：

```bash
python scripts/validate_handoff.py
```

本包是实现规范和数据包，未包含新游戏引擎、已训练策略或游戏网页。它以用户提供的 `boardbench-v0-review.zip` 为接口参照；若本地项目已有更新，先检查差异，保留有效修改。
