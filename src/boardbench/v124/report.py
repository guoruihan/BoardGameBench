"""Matched full-game score / actual-cost panels; no completed-only promotion."""
from collections import defaultdict
import csv
import io
import math
from pathlib import Path
from statistics import mean
import time

from boardbench.artifacts.store import read_json, write_json
from boardbench.v122.report import environment_means, paired, summarize as base_summary


def summarize(plan, jobs, results):
    by_id = {j['id']: j for j in jobs}
    for r in results:
        if r['id'] not in by_id or r['params'] != by_id[r['id']]['params']:
            raise ValueError('result parameters differ from registered attempt')
        if r['status'] == 'completed':
            if not math.isfinite(r['decision_seconds']) or r['decision_seconds'] < 0:
                raise ValueError('invalid measured decision time')
            if r['components']['total'] != r['score']:
                raise ValueError('component / true score mismatch')
    summary = base_summary(plan, jobs, results)
    grouped = defaultdict(list)
    for row in results:
        grouped[row['arm']].append(row)
    specs = {a['id']: a for a in plan['arms']}
    env_means = {a: environment_means(grouped[a], 1 if spec['method'] == 'heuristic'
                                    else len(plan['search_seeds'])) for a, spec in specs.items()}
    groups = defaultdict(list)
    for a, spec in specs.items():
        p = spec['params']
        label = f"timed_{p['seconds']:g}s" if p['mode'] == 'timed' else p['mode']
        groups[label].append(a)
    tables, comparisons = {}, {}
    seeds = plan['environment_seeds']
    for group, arms in groups.items():
        panel = max((n for n in plan['panels'] if all(set(seeds[:n]) <= set(env_means[a])
                                                    for a in arms)), default=0)
        selected = set(seeds[:panel])
        cells = {}
        for a in arms:
            rows = [r for r in grouped[a] if r['environment_seed'] in selected]
            cells[a] = {
                'mean': mean(env_means[a][s] for s in selected) if selected else None,
                'decision_seconds': mean(r['decision_seconds'] for r in rows) if rows else None,
                'animals': mean(r['components']['animals'] for r in rows) if rows else None,
                'simulation_steps': mean(r['simulation_steps'] for r in rows) if rows else None,
                'fallback_decisions': mean(r.get('fallback_decisions', 0) for r in rows) if rows else None,
                'budget_utilization': mean(r['solver_budget_utilization'] for r in rows)
                    if rows and specs[a]['params']['mode'] == 'timed' else None,
                'max_budget_overrun_seconds': max((r.get('solver_budget_overrun_seconds', 0) for r in rows), default=0),
                'terminal_metrics': {k: mean(r['terminal_metrics'][k] for r in rows)
                    for k in ('terrain', 'animals_placed', 'completed_cards', 'turns', 'tokens')} if rows else {},
            }
        reference = next((a for a in arms if specs[a]['params']['continuation'] == 'original'), None)
        for a in arms:
            if a == reference or reference is None:
                continue
            comparison = paired({s: env_means[a][s] for s in selected},
                                {s: env_means[reference][s] for s in selected})
            ratio = (cells[a]['decision_seconds'] / max(1e-12, cells[reference]['decision_seconds'])) if panel else None
            comparison.update(actual_time_ratio=ratio,
                actual_mean_times_within_10_percent=ratio is not None and .9 <= ratio <= 1.1,
                shared_time_cap=specs[a]['params'].get('seconds'),
                caveat='Equal caps are not equal actual time; check ratio and score-cost curve.')
            comparisons[f'{a}-vs-{reference}'] = comparison
        tables[group] = {'balanced_panel_size': panel, 'cells': cells}
    # Keep the legacy reference independently visible, never fold its fixed
    # sample count into the wall-cap axis or call it time-matched automatically.
    fixed = [a for a in specs if specs[a]['params']['mode'] == 'fixed']
    for ref in fixed:
        for a, spec in specs.items():
            if spec['params']['mode'] != 'timed':
                continue
            n = max((n for n in plan['panels'] if set(seeds[:n]) <= set(env_means[a]) & set(env_means[ref])), default=0)
            chosen = seeds[:n]
            comparison = paired({s: env_means[a][s] for s in chosen}, {s: env_means[ref][s] for s in chosen})
            comparison['time_matched_claim'] = False
            comparisons[f'{a}-vs-{ref}'] = comparison
    summary.update(protocol=plan['protocol'], split=plan['split'], groups=tables, paired=comparisons,
        interpretation='Full natural-terminal development games, NOT 3-second deployment or held-out test. '
        'Repeat means are paired by environment. Intervals descriptive, no multiplicity correction. '
        'Equal caps do not establish equal measured times. Failures remain visible; only complete prefixes qualify.')
    return summary


def publish(root, plan, jobs, results, status):
    root = Path(root)
    summary = summarize(plan, jobs, results)
    summary['status'] = status
    if (root / 'allocation.json').exists():
        a = read_json(root / 'allocation.json')
        stop = status.get('watchdog', {}).get('finished_unix', time.time())
        reservation = len(a['cpu_affinity']) * max(0., stop - a['started_unix'])
        prior = a.get('prior_charged_cpu_core_seconds', 0.)
        summary['resource_charge'] = {'reserved_cpu_core_seconds': reservation,
            'preflight_charged_cpu_core_seconds': prior, 'total_charged_cpu_core_hours': (reservation + prior) / 3600,
            'reported_worker_cpu_seconds': sum(r.get('cpu_seconds', 0.) for r in results), 'gpu_hours': 0}
    write_json(root / 'summary.json', summary)
    def fmt(x):
        return '—' if x is None else f'{x:.3f}'
    lines = ['# Harmonies V1.2.4 完整局后续策略对照', '',
        f"状态：{status['phase']}；已结算 {len(results)}/{len(jobs)}；划分：{plan['split']}。", '',
        '全部使用公开信息、真实自然终局分；不是局部Q，也不是3秒部署成绩。',
        'timed组共享原启发式top6、按模拟剩余步数分配时间、相同缓存容量和公共未来采样。',
        '时间按solver调用墙钟累计；环境执行和回放I/O另计。末尾不完整采样轮整轮丢弃，成本仍保留。',
        '同一秒数是相同预算上限，不自动代表实际耗时相同；只有均耗时比在0.9–1.1时标记近似匹配。', '']
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['group', 'arm', 'environments', 'score', 'animals', 'decision_seconds', 'budget_utilization', 'simulation_steps'])
    for name, group in summary['groups'].items():
        lines += [f"## {name}：完整配对环境 {group['balanced_panel_size']}", '',
            '|方法|总分|动物分|决策秒/局|预算使用率|放动物数|完成卡|',
            '|---|---:|---:|---:|---:|---:|---:|']
        for arm, cell in group['cells'].items():
            metrics = cell['terminal_metrics']
            lines.append('|' + '|'.join([arm] + [fmt(x) for x in (cell['mean'], cell['animals'],
                cell['decision_seconds'], cell['budget_utilization'], metrics.get('animals_placed'), metrics.get('completed_cards'))]) + '|')
            writer.writerow([name, arm, group['balanced_panel_size']] + [cell[k] for k in
                ('mean', 'animals', 'decision_seconds', 'budget_utilization', 'simulation_steps')])
        lines.append('')
    lines += ['## 配对差值', '', '|对比|环境数|均差|描述性95%区间|实际耗时比|', '|---|---:|---:|---|---:|']
    for name, c in summary['paired'].items():
        ci = c['normal_95_ci']
        lines.append(f"|{name}|{c['n']}|{fmt(c['mean_gap'])}|" +
                     (f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '—') + f"|{fmt(c.get('actual_time_ratio'))}|")
    lines += ['', '## 覆盖与失败', '', '|方法|完成/已结算/计划|失败/截断|', '|---|---:|---:|']
    for arm, cell in summary['cells'].items():
        lines.append(f"|{arm}|{cell['completed']}/{cell['attempted']}/{cell['planned']}|{cell['failed_or_censored']}|")
    lines += ['', '每环境先平均固定搜索重复，再计算环境级配对区间。未校正多重比较；禁止按先完成的容易局宣布提升。',
              '未完成自然终局的尝试不伪造终局分；summary.json同时保留不完整按0的保守均值。',
              'greedy是不搜索的机制对照；fixed使用配置中的固定样本数，不自动属于同耗时对照。',
              '逐动作及分项在episodes/*/decisions.jsonl；失败和成本也保留；不训练、不更新部署策略。', '']
    if 'resource_charge' in summary:
        lines += [f"含预检累计预留：{summary['resource_charge']['total_charged_cpu_core_hours']:.3f} CPU核时，GPU0。", '']
    for name, content in [('REPORT.md', '\n'.join(lines)), ('curve.csv', buffer.getvalue())]:
        temp = root / (name + '.tmp')
        temp.write_text(content)
        temp.replace(root / name)
    return summary
