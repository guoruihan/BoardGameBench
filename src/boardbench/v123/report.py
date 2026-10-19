"""Independent confirmation contrasts, clustered by original environment."""
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
from statistics import mean, stdev
import time

from boardbench.artifacts.store import write_json
from .policies import POLICIES


def interval(values):
    values = list(values)
    center = mean(values) if values else None
    error = 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {'n_environments': len(values), 'mean': center,
            'normal_95_ci': [center - error, center + error] if error is not None else None}


def effects(row):
    matrix = {name: {c['rank']: c for c in row['confirmation'][name]} for name in POLICIES}
    choices = row['choices']
    result = {}
    def value(policy, rank):
        return matrix[policy][rank]['mean']
    for name in POLICIES:
        pick = choices[name]
        result[f'{name}:all_minus_top6'] = value(name, pick['all']) - value(name, pick['top'])
        if pick['outside'] is not None:
            result[f'{name}:outside_minus_top6'] = value(name, pick['outside']) - value(name, pick['top'])
    fixed = choices['original']['top']
    result['fixed_root:space_minus_original'] = value('space', fixed) - value('original', fixed)
    result['fixed_root:habitat_minus_space'] = value('habitat', fixed) - value('space', fixed)
    for name in ('space', 'habitat'):
        old, new = choices['original']['all'], choices[name]['all']
        new_gap = value(name, new) - value(name, old)
        old_gap = value('original', new) - value('original', old)
        result[f'{name}:root_switch_under_new'] = new_gap
        result[f'{name}:root_switch_under_original'] = old_gap
        result[f'{name}:preference_interaction'] = new_gap - old_gap
    return result


def component_effects(row):
    fixed = row['choices']['original']['top']
    cells = {name: next(c for c in row['confirmation'][name] if c['rank'] == fixed) for name in POLICIES}
    if not all('components' in c for c in cells.values()):
        return {}
    return {f'{new}_minus_{old}:{metric}': cells[new]['components'][metric] - cells[old]['components'][metric]
            for new, old in (('space', 'original'), ('habitat', 'space'))
            for metric in ('terrain', 'animals', 'turns', 'tokens', 'animals_placed', 'completed_cards', 'empty_cells')}


def summarize(plan, bank, jobs, results):
    job_by_id = {j['id']: j for j in jobs}
    if len(job_by_id) != len(jobs) or len({r['id'] for r in results}) != len(results):
        raise ValueError('duplicate diagnostic job/result')
    for row in results:
        if row['id'] not in job_by_id or row['environment_index'] != job_by_id[row['id']]['environment_index']:
            raise ValueError('unregistered result')
        if row['status'] == 'completed':
            roots = set(r['rank'] for r in row['confirmation']['original'])
            for name in POLICIES:
                if {r['rank'] for r in row['confirmation'][name]} != roots:
                    raise ValueError('continuations evaluated on different confirmation roots')
                if len(row['discovery'][name]) != row['legal_count']:
                    raise ValueError('missing discovery action')
                for cell in row['confirmation'][name]:
                    if len(cell['scores']) != plan['confirmation_samples']:
                        raise ValueError('incomplete confirmation samples')
    completed = {r['id']: r for r in results if r['status'] == 'completed'}
    panel = 0
    for size in plan['panels']:
        if all(j['id'] in completed for j in jobs if j['environment_index'] < size):
            panel = size
    selected = [row for row in completed.values() if row['environment_index'] < panel]
    grouped = defaultdict(lambda: defaultdict(list))
    components = defaultdict(lambda: defaultdict(list))
    for row in selected:
        for name, value in effects(row).items():
            grouped[name][row['environment_index']].append(value)
        for name, value in component_effects(row).items():
            components[name][row['environment_index']].append(value)
    estimates = {name: interval(mean(values) for values in groups.values()) for name, groups in grouped.items()}
    source_metrics = defaultdict(lambda: defaultdict(list))
    for source in bank.get('sources', []):
        producer = source['job'].rsplit('_e', 1)[0]
        for metric, value in source['terminal_metrics'].items():
            source_metrics[producer][metric].append(value)
    return {'protocol': 'local_teacher_diagnostic_v123', 'split': bank['split'],
            'planned_states': len(jobs), 'resolved': len(results), 'completed': len(completed),
            'failed_or_censored': len(results) - len(completed), 'unresolved': len(jobs) - len(results),
            'statuses': dict(Counter(r['status'] for r in results)),
            'balanced_environment_panel': panel, 'balanced_states': len(selected),
            'effects': estimates,
            'fixed_root_components': {name: interval(mean(v) for v in groups.values()) for name, groups in components.items()},
            'state_bank_coverage': {'phase_counts': dict(Counter(s['phase'] for s in bank['states'])),
                                    'states_with_pruned_actions': sum(len(s['observation']['legal_actions']) > plan['width'] for s in bank['states']),
                                    'target_kind_misses': sum(not s.get('target_kind_available', True) for s in bank['states'])},
            'reported_simulation_steps': sum(r.get('simulation_steps', 0) for r in results),
            'reported_terminal_rollouts': sum(r.get('terminal_rollouts', 0) for r in results),
            'reported_cpu_seconds_lower_bound': sum(r.get('cpu_seconds', 0.) for r in results),
            'source_trajectory_means': {p: {k: mean(v) for k, v in metrics.items()}
                                        for p, metrics in source_metrics.items()},
            'source_environments': bank['environments'],
            'caveats': ['Development local Q estimates, not full-game or equal-time performance.',
                        'Confirmation is independent of selection, not a held-out environment test.',
                        'State replicates and rollout seeds are not independent source games.',
                        'Intervals are descriptive normal approximations, no multiple-testing correction.',
                        'Outside-top6 effects use only eligible states; n may differ.',
                        'Incomplete counters are lower bounds; reservation charges are separate.']}


def publish(root, plan, bank, jobs, results, status):
    root = Path(root)
    summary = summarize(plan, bank, jobs, results)
    summary.update(phase=status['phase'], updated_unix=time.time())
    write_json(root / 'summary.json', summary)
    temporary = root / 'states.csv.tmp'
    with temporary.open('w') as stream:
        columns = ['id', 'environment_index', 'producer', 'status', 'phase', 'policy', 'legal_count',
                   'top_rank', 'outside_rank', 'all_rank', 'outside_type', 'outside_minus_top6',
                   'all_minus_top6', 'elapsed_seconds']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in results:
            common = {k: row.get(k) for k in ('id', 'environment_index', 'producer', 'status', 'phase',
                                             'legal_count', 'elapsed_seconds')}
            if row['status'] != 'completed':
                writer.writerow(common)
                continue
            contrasts = effects(row)
            for policy in POLICIES:
                selection = row['choices'][policy]
                outside = selection['outside']
                writer.writerow({**common, 'policy': policy, 'top_rank': selection['top'],
                                 'outside_rank': outside, 'all_rank': selection['all'],
                                 'outside_type': row['discovery'][policy][outside]['action']['type'] if outside is not None else '',
                                 'outside_minus_top6': contrasts.get(f'{policy}:outside_minus_top6'),
                                 'all_minus_top6': contrasts[f'{policy}:all_minus_top6']})
    temporary.replace(root / 'states.csv')
    lines = ['# V1.2.3 局部教师诊断', '',
             f"状态：{status['phase']}；完成 {summary['completed']}/{len(jobs)} 个局面；"
             f"失败/截断 {summary['failed_or_censored']}；尚未结算 {summary['unresolved']}。", '',
             f"主结果仅使用完整前缀：{summary['balanced_environment_panel']} 个源环境、"
             f"{summary['balanced_states']} 个局面。每个源环境先平均，再计算描述性区间。", '',
             f"数据划分：{bank['split']}。这是公开局面的局部 Q 诊断，不是完整游戏、同耗时成绩或最优性证明。",
             ('训练集校准只验证流程与成本，不用于方法优劣结论。' if bank['split'] == 'train_calibration'
              else '开发环境用于诊断；独立复核随机数不是新的确认/测试环境。'), '',
             '## 独立复核', '', '| 对比（分） | 环境数 | 均差 | 描述性 95% 区间 |',
             '|---|---:|---:|---|']
    for name, estimate in summary['effects'].items():
        ci = estimate['normal_95_ci']
        lines.append(f"| {name} | {estimate['n_environments']} | {estimate['mean']:.3f} | "
                     + (f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '不足以估计') + ' |')
    lines += ['', 'all_minus_top6：发现阶段从全动作/前6各选一个，再独立复核；'
              'outside_minus_top6：发现阶段最好的被剪动作与前6赢家比较。未赢也保留负数。', '',
              'fixed_root：固定原策略发现阶段选中的前6赢家，只换后续策略；'
              'preference_interaction：新旧根动作偏好的差中之差，不把所有动作一起涨分误判为排序改善。', '',
              '区间未作多重比较校正，不能根据单个局面的显著性宣称发现全局更优动作。', '',
              '## 固定根动作的后续收益分解', '',
              '仍使用同一完整环境前缀；下表为均差，环境级区间保存在summary.json。', '',
              '| 后续对比 | 地形 | 动物 | 回合 | 资源数 | 动物数 | 完成卡 | 空格 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for contrast in ('space_minus_original', 'habitat_minus_space'):
        cells = [summary['fixed_root_components'].get(f'{contrast}:{k}') for k in
                 ('terrain', 'animals', 'turns', 'tokens', 'animals_placed', 'completed_cards', 'empty_cells')]
        if all(cells):
            lines.append('| ' + contrast + ' | ' + ' | '.join(f'{c["mean"]:.3f}' for c in cells) + ' |')
    lines += ['',
              '## 源轨迹分解（仅状态库选定的源环境）', '',
              '| 来源 | 总分 | 地形 | 动物 | 回合 | 已放资源 | 动物数 | 完成卡 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for producer, metrics in summary['source_trajectory_means'].items():
        lines.append('| ' + producer + ' | ' + ' | '.join(f'{metrics[k]:.2f}' for k in
                     ('score', 'terrain', 'animals', 'turns', 'tokens', 'animals_placed', 'completed_cards')) + ' |')
    lines += ['', '## 证据与成本', '',
              '- `states.csv`：所有已结算局面（包括失败），剪枝赢家动作类型与独立差值。',
              '- `episodes/*/selection.json`：复核前冻结的动作选择、发现值及分离的随机数流。',
              '- `episodes/*/rollouts.jsonl`：每个样本的真实终局分、分项、资源/动物数与终局标记。',
              '- `resolved/*.json`：原始矩阵、所有失败/截断状态及时间/CPU/模拟步计数。',
              '- `allocation.json`、`queue_result.json`、`watchdog_result.json`：来源与全部预留/前置成本。',
              '- 未完成计数为下界；队列和校准也计费，不只统计成功局面。生产策略保持不变。', '']
    temp = root / 'REPORT.md.tmp'
    temp.write_text('\n'.join(lines))
    temp.replace(root / 'REPORT.md')
    return summary
