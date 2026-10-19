"""Reuse complete environment pairing, with explicit small-sample caveats."""
from pathlib import Path

from boardbench.artifacts.store import write_json
from boardbench.v124.report import publish as base_publish
from boardbench.v122.report import paired


def publish(root, plan, jobs, results, status):
    summary = base_publish(root, plan, jobs, results, status)
    panel = summary['balanced_panel']
    for a in panel:
        for ref in plan.get('references', ['original', 'habitat']):
            if ref in panel and ref != a:
                summary['paired'][f'{a}-vs-{ref}'] = paired(
                    panel[a]['by_environment'], panel[ref]['by_environment'])
    summary['research_round'] = plan.get('research_round')
    write_json(Path(root) / 'summary.json', summary)
    report = Path(root) / 'REPORT.md'
    content = report.read_text().replace('Harmonies V1.2.4 完整局后续策略对照', 'Harmonies V1.2.5 小批量策略迭代')
    content = content.replace('timed组共享原启发式top6、按模拟剩余步数分配时间、相同缓存容量和公共未来采样。',
        '所有方法共享累计墙钟调度和公开未来种子；根候选/后续策略/缓存差异见冻结配置和源码。')
    content += '\n## 本轮配对参考与逐环境分数\n\n'
    content += '小样本开发筛选，不是独立测试或最优水平；候选经过自适应选型。\n\n'
    for arm, cell in panel.items():
        content += f"- {arm}: {cell['by_environment']}\n"
    for name, comparison in summary['paired'].items():
        content += f"- {name}: n={comparison['n']}, gap={comparison['mean_gap']}, descriptive95={comparison['normal_95_ci']}\n"
    report.write_text(content)
    return summary
