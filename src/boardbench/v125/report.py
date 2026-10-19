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
                comparison = paired(panel[a]['by_environment'], panel[ref]['by_environment'])
                left, right = panel[a]['decision_seconds'], panel[ref]['decision_seconds']
                ratio = left / max(1e-12, right) if left is not None and right is not None else None
                comparison.update(actual_time_ratio=ratio,
                    actual_mean_times_within_10_percent=ratio is not None and .9 <= ratio <= 1.1)
                summary['paired'][f'{a}-vs-{ref}'] = comparison
    summary['research_round'] = plan.get('research_round')
    if 'historical_reference' in plan:
        from .historical import comparisons
        summary['historical_comparison'] = comparisons(plan,summary)
    if 'resource_charge' in summary:
        summary['resource_charge']['reported_worker_cpu_seconds'] = sum(
            r.get('cpu_seconds', 0) for r in results if not r.get('reused_completed_attempt'))
        summary['resource_charge']['resume_parent'] = plan.get('resume_from')
    write_json(Path(root) / 'summary.json', summary)
    report = Path(root) / 'REPORT.md'
    content = report.read_text().replace('Harmonies V1.2.4 完整局后续策略对照', 'Harmonies V1.2.5 小批量策略迭代')
    if 'historical_reference' in plan:
        content=content.replace('Harmonies V1.2.5 小批量策略迭代','Harmonies V1.2.5 保留策略扩展评测')
    content = content.replace('timed组共享原启发式top6、按模拟剩余步数分配时间、相同缓存容量和公共未来采样。',
        '所有方法共享累计墙钟调度和公开未来种子；根候选/后续策略/缓存差异见冻结配置和源码。')
    content += '\n## 本轮配对参考与逐环境分数\n\n'
    content += ('开发集扩测，策略已冻结；不是独立正式测试或最优水平。\n\n' if 'historical_reference' in plan
                else '小样本开发筛选，不是独立测试或最优水平；候选经过自适应选型。\n\n')
    for arm, cell in panel.items():
        content += f"- {arm}: {cell['by_environment']}\n"
    for name, comparison in summary['paired'].items():
        content += f"- {name}: n={comparison['n']}, gap={comparison['mean_gap']}, descriptive95={comparison['normal_95_ci']}\n"
    if 'historical_comparison' in summary:
        history=summary['historical_comparison']
        content+='\n## 历史900秒对照（与本轮同批对照分开）\n\n'
        content+='环境及搜索seed完全相同；每环境先平均搜索重复，再作配对。旧实验CPU负载混合，历史差值不是同批受控因果比较。\n\n'
        content+='|新方法-vs-历史方法|完整环境数|新均分|历史均分|差值|描述性95%区间|实际耗时比|\n|---|---:|---:|---:|---:|---|---:|\n'
        def fmt(value):
            return '—' if value is None else f'{value:.3f}'
        for name,c in history['comparisons'].items():
            ci=c['normal_95_ci'];interval=f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '—'
            content+=f"|{name}|{c['n']}|{fmt(c['current_mean'])}|{fmt(c['historical_mean'])}|{fmt(c['mean_gap'])}|{interval}|{fmt(c['actual_time_ratio'])}|\n"
    report.write_text(content)
    return summary
