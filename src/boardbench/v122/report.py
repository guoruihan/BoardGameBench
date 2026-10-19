"""All-attempt coverage plus paired, environment-level development summaries."""
from collections import defaultdict
import csv
import io
import math
from pathlib import Path
import statistics
import time

from boardbench.artifacts.store import read_json, write_json


def mean(values):
    return statistics.mean(values) if values else None


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else None


def environment_means(rows, expected_repetitions):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['environment_seed']].append(row)
    return {seed: mean([r['score'] for r in group]) for seed, group in grouped.items()
            if len(group) == expected_repetitions
            and len({r['search_seed'] for r in group}) == expected_repetitions
            and all(r['status'] == 'completed' and r['natural_terminal'] for r in group)}


def paired(left, right):
    seeds = sorted(set(left) & set(right))
    differences = [left[s] - right[s] for s in seeds]
    average, spread = mean(differences), sd(differences)
    half = 1.96 * spread / math.sqrt(len(differences)) if spread is not None else None
    return {'environment_seeds': seeds, 'n': len(seeds), 'mean_gap': average,
            'normal_95_ci': [average - half, average + half] if half is not None else None,
            'unit': 'environment mean after averaging fixed search repetitions; descriptive normal interval'}


def summarize(plan, jobs, results):
    if len({r['id'] for r in results}) != len(results):
        raise ValueError('duplicate attempt result')
    job_map = {j['id']: j for j in jobs}
    planned_ids = set(job_map)
    if any(r['id'] not in planned_ids for r in results):
        raise ValueError('result not in frozen plan')
    for row in results:
        if any(row[k] != job_map[row['id']][k] for k in
               ('arm', 'environment_seed', 'search_seed', 'method', 'multiplier')):
            raise ValueError('result identity differs from frozen job')
        if row['status'] == 'completed' and (not row['natural_terminal'] or
                not isinstance(row['score'], (int, float)) or not math.isfinite(row['score'])):
            raise ValueError('completed job requires a finite natural-terminal score')
    arms = {a['id']: a for a in plan['arms']}
    grouped, planned = defaultdict(list), defaultdict(list)
    for job in jobs:
        planned[job['arm']].append(job)
    for row in results:
        grouped[row['arm']].append(row)
    complete_means, cells = {}, {}
    for arm, spec in arms.items():
        rows = grouped[arm]
        repeats = 1 if spec['method'] == 'heuristic' else len(plan['search_seeds'])
        envs = environment_means(rows, repeats)
        complete_means[arm] = envs
        finished = [r for r in rows if r['status'] == 'completed' and r['natural_terminal']]
        finished_ids = {r['id'] for r in finished}
        seed_means = {str(s): mean([r['score'] for r in finished
                                   if r['search_seed'] == s and r['environment_seed'] in envs])
                      for s in sorted({r['search_seed'] for r in rows})}
        simulations = sum(r.get('simulation_steps', 0) for r in rows)
        rollouts = sum(r.get('rollouts', 0) for r in rows)
        cells[arm] = {
            **spec, 'planned': len(planned[arm]), 'attempted': len(rows), 'completed': len(finished),
            'failed_or_censored': len(rows) - len(finished), 'complete_environments': len(envs),
            'completed_only_mean': mean([r['score'] for r in finished]),
            'attempted_zero_for_incomplete_mean': mean([r['score'] if r['id'] in finished_ids else 0 for r in rows]),
            'complete_environment_mean': mean(list(envs.values())),
            'environment_sd': sd(list(envs.values())), 'search_seed_means': seed_means,
            'search_seed_mean_sd': sd([v for v in seed_means.values() if v is not None]),
            'mean_decision_seconds_completed': mean([r['decision_seconds'] for r in finished]),
            'mean_simulation_steps_completed': mean([r['simulation_steps'] for r in finished]),
            'mean_animal_score_completed': mean([r['components'].get('animals', 0) for r in finished]),
            'mean_terrain_score_completed': mean([r['score'] - r['components'].get('animals', 0) for r in finished]),
            'simulation_steps_recorded': simulations,
            'cpu_seconds_recorded': sum(r.get('cpu_seconds', 0) for r in rows),
            'attempt_wall_seconds': sum(r.get('supervised_seconds', r.get('elapsed_seconds', 0)) for r in rows),
            'terminal_rollout_fraction': sum(r.get('terminal_rollouts', 0) for r in finished) /
                max(1, sum(r.get('rollouts', 0) for r in finished)),
            'horizon_completed_fraction': sum(r.get('horizon_completed_rollouts', 0) for r in finished) /
                max(1, sum(r.get('rollouts', 0) for r in finished)),
            'reported_rollouts': rollouts,
            'root_candidate_coverage': sum(r.get('candidate_total', 0) for r in finished) /
                max(1, sum(r.get('legal_total', 0) for r in finished)),
            'pruned_decisions': sum(r.get('pruned_decisions', 0) for r in finished),
        }
    # Never silently compare arms on different easy-to-finish subsets. The main
    # plot uses the largest PREDECLARED complete prefix shared by primary arms.
    primary = [a['id'] for a in plan['arms'] if a.get('primary', True)]
    seeds = plan['environment_seeds']
    complete_panels = [n for n in plan['panels']
                       if all(set(seeds[:n]) <= set(complete_means[a]) for a in primary)]
    panel = max(complete_panels, default=0)
    selected = seeds[:panel]
    panel_cells = {}
    for arm in primary:
        values = [complete_means[arm][s] for s in selected]
        spread, average = sd(values), mean(values)
        half = 1.96 * spread / math.sqrt(panel) if spread is not None else None
        panel_rows = [r for r in grouped[arm] if r['environment_seed'] in selected]
        panel_cells[arm] = {'n': panel, 'mean': average, 'environment_sd': spread,
                           'normal_95_ci': [average - half, average + half] if half is not None else None,
                           'animal_mean': mean([r['components'].get('animals', 0) for r in panel_rows]),
                           'decision_seconds': mean([r['decision_seconds'] for r in panel_rows]),
                           'simulation_steps': mean([r['simulation_steps'] for r in panel_rows]),
                           'by_environment': {str(s): complete_means[arm][s] for s in selected}}
    comparisons = {}
    for arm in primary:
        for reference in ('heuristic', 'short_1x'):
            if arm != reference and reference in primary:
                comparisons[f'{arm}-vs-{reference}'] = paired(
                    {s: complete_means[arm][s] for s in selected},
                    {s: complete_means[reference][s] for s in selected})
    for method in ('short', 'terminal'):
        names = [a for a in primary if arms[a]['method'] == method]
        names.sort(key=lambda a: arms[a]['multiplier'])
        for previous, current in zip(names, names[1:]):
            comparisons[f'{current}-vs-{previous}'] = paired(
                {s: complete_means[current][s] for s in selected},
                {s: complete_means[previous][s] for s in selected})
    return {'updated_unix': time.time(), 'cells': cells, 'complete_panels': complete_panels,
            'balanced_panel_size': panel, 'balanced_panel': panel_cells, 'paired': comparisons,
            'planned_jobs': len(jobs), 'resolved_jobs': len(results),
            'interpretation': 'Development only. Partial cells are coverage diagnostics, not matched evidence. '
                              'Sampling repetitions are NOT independent environment or training seeds. '
                              'A flat curve is not an optimality certificate.'}


def curve_svg(summary):
    panel = summary['balanced_panel']
    if not summary['balanced_panel_size']:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="150"><text x="20" y="70">Waiting for a complete predeclared balanced panel.</text></svg>\n'
    values = [v['mean'] for v in panel.values()]
    low, high = min(values) - 8, max(values) + 8
    def y(value):
        return 340 - (value - low) / max(1, high - low) * 280
    def x(scale):
        return 95 + math.log(scale, 4) * 175
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="425" viewBox="0 0 760 425">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="70" y="26" font-size="17">Harmonies: matched development {summary["balanced_panel_size"]} games</text>',
                '<path d="M70 45V350H680" fill="none" stroke="#333"/>']
    for scale in (1, 4, 16, 64):
        elements.append(f'<text x="{x(scale)-8:.1f}" y="373">{scale}x</text>')
    elements.append('<text x="220" y="403">Samples per candidate (4 x multiplier); not equal seconds</text>')
    for i in range(6):
        value = low + (high - low) * i / 5
        elements.append(f'<text x="20" y="{y(value)+4:.1f}">{value:.1f}</text>')
    for method, color in (('short', '#2166ac'), ('terminal', '#b2182b')):
        points = [(summary['cells'][a]['multiplier'], v['mean']) for a, v in panel.items()
                  if summary['cells'][a]['method'] == method]
        points.sort()
        coordinates = ' '.join(f'{x(m):.1f},{y(v):.1f}' for m, v in points)
        elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        for scale, value in points:
            elements.append(f'<circle cx="{x(scale):.1f}" cy="{y(value):.1f}" r="4" fill="{color}"/>')
        elements.append(f'<text x="{460 if method == "short" else 550}" y="48" fill="{color}">{method}</text>')
    if 'heuristic' in panel:
        value = panel['heuristic']['mean']
        elements.append(f'<path d="M70 {y(value):.1f}H680" stroke="#777" stroke-dasharray="5 4"/>')
        elements.append(f'<text x="580" y="{y(value)-6:.1f}" fill="#777">heuristic</text>')
    elements.append('</svg>\n')
    return ''.join(elements)


def publish(root, plan, jobs, results, status):
    root = Path(root)
    summary = summarize(plan, jobs, results)
    summary['status'] = status
    if (root / 'allocation.json').exists():
        allocation = read_json(root / 'allocation.json')
        stop = status.get('watchdog', {}).get('finished_unix', time.time())
        current = len(allocation['cpu_affinity']) * max(0., stop - allocation['started_unix'])
        prior = allocation.get('prior_charged_cpu_core_seconds', 0.)
        summary['resource_charge'] = {'campaign_reserved_core_seconds': current,
                                     'preflight_charged_core_seconds': prior,
                                     'cumulative_charged_core_hours': (current + prior) / 3600,
                                     'gpu_hours': 0.}
    write_json(root / 'summary.json', summary)
    lines = ['# Harmonies V1.2.2 搜索预算诊断', '',
             f'状态：{status["phase"]}；已结算 {len(results)}/{len(jobs)} 项。', '',
             '仅开发集；不更新部署策略。倍率增加采样次数与短搜索步数上限，不代表等倍实耗或两方法等耗时。',
             '完整对局参考与 3 秒部署协议不可混同；曲线变平不代表接近最优。', '',
             f'## 主要证据：预先固定、所有方法均完整完成的开发局前缀 n={summary["balanced_panel_size"]}', '',
             '|方法|总分均值|局间 SD|动物分|决策秒/局|模拟步/局|',
             '|---|---:|---:|---:|---:|---:|']
    def number(value):
        return '—' if value is None else f'{value:.3f}'
    for arm, cell in summary['balanced_panel'].items():
        lines.append('|'+ '|'.join([arm] + [number(cell[k]) for k in
                     ('mean', 'environment_sd', 'animal_mean', 'decision_seconds', 'simulation_steps')]) + '|')
    lines += ['', '每局先平均固定搜索随机种子，再计算局间 SD 和配对区间；不能把三次搜索当三局独立环境。',
              '区间是描述性的正态近似，不是多重比较校正的显著性结论。`summary.json` 包含配对差值及搜索种子间波动。', '',
              '![budget-score curve](curve.svg)', '', '## 覆盖与成本（不完整单元格不得作为匹配成绩比较）', '',
              '|方法|完成/已结算/计划|完整局均值（可能有选择偏差）|失败或截断|已记录 CPU 秒|',
              '|---|---:|---:|---:|---:|']
    for arm, cell in summary['cells'].items():
        lines.append(f'|{arm}|{cell["completed"]}/{cell["attempted"]}/{cell["planned"]}|'
                     f'{number(cell["completed_only_mean"])}|{cell["failed_or_censored"]}|{number(cell["cpu_seconds_recorded"])}|')
    lines += ['', '未开始不算已尝试；超时/异常保留记录，终局分为 null，另存最后已提交局面分。',
              'JSON 另列已尝试且不完整按 0 的保守均值。进程中断时 CPU/模拟步计数为最近心跳下界；',
              '资源占用上界以 allocation/watchdog_result 的整段 CPU 核时计，含初始化、失败、评测、输出。GPU 使用为 0。', '',
              '## 接下来如何判断', '',
              '- 采样增加且配对分数提高：先研究压缩或加速该老师。',
              '- 实耗增加而分数平：先检查候选剪枝、动物布局评价和后续启发式，而非宣称已最优。',
              '- 搜索种子波动大：增加估计稳定性诊断；终局 rollout 不保证胜过短搜索。',
              '- 未完成相同面板：先补齐覆盖，不从先完成的容易局下结论。', '']
    if 'resource_charge' in summary:
        lines += [f'含预检的累计资源记账：约 {summary["resource_charge"]["cumulative_charged_core_hours"]:.3f} CPU 核时；GPU 0。', '']
    for filename, content in [('REPORT.md', '\n'.join(lines)), ('curve.svg', curve_svg(summary))]:
        temporary = root / (filename + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(root / filename)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['arm', 'balanced_games', 'score', 'environment_sd', 'animals', 'decision_seconds', 'simulation_steps'])
    for arm, cell in summary['balanced_panel'].items():
        writer.writerow([arm] + [cell[k] for k in ('n', 'mean', 'environment_sd', 'animal_mean', 'decision_seconds', 'simulation_steps')])
    temporary = root / 'curve.csv.tmp'
    temporary.write_text(buffer.getvalue())
    temporary.replace(root / 'curve.csv')
    return summary
