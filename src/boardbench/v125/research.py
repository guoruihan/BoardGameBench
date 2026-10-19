"""Consolidate explicitly registered research tasks, decisions and resource costs.

This is a recovery/report tool, not a preprogrammed replacement for research
decisions. The agent updates the manifest after inspecting each complete round.
"""
import argparse
from pathlib import Path
import time

from boardbench.artifacts.store import read_json, write_json

def publish(directory, repo):
    directory, repo = Path(directory), Path(repo)
    manifest = read_json(directory / 'experiment.json')
    rows, charges, receipts_seen = [], [], set()
    lines = ['# Harmonies V1.2.5 自主迭代进度', '',
             '所有均分按对应任务的相同环境比较；跨不同环境的原始均分不能直接排名。', '',
             '4环境筛选属于探索；不是正式测试、3秒部署或最优水平证明。', '']
    for item in manifest['tasks']:
        path = repo / item['run']
        if not (path / 'status.json').exists():
            rows.append({**item, 'phase': 'not_started'})
            continue
        status = read_json(path / 'status.json')
        row = {**item,'phase':status['phase'],'resolved':status.get('resolved',0)}
        if (path / 'summary.json').exists():
            summary = read_json(path / 'summary.json')
            row['paired_panel'] = summary['balanced_panel']
            row['paired'] = summary['paired']
            row['failures'] = sum(c['failed_or_censored'] for c in summary['cells'].values())
            lines += [f"## {item['id']} — {status['phase']}", '', item.get('hypothesis',''), '',
                      '|策略|环境数|均分|动物分|秒/局|', '|---|---:|---:|---:|---:|']
            for name, cell in summary['balanced_panel'].items():
                fmt=lambda x:'—' if x is None else f'{x:.3f}'
                lines.append(f"|{name}|{cell['n']}|{fmt(cell['mean'])}|{fmt(cell['animal_mean'])}|{fmt(cell['decision_seconds'])}|")
            if item.get('conclusion'):
                lines += ['', item['conclusion']]
            lines += ['', f"完整报告：{path / 'REPORT.md'}", '']
        if (path / 'allocation.json').exists():
            alloc = read_json(path / 'allocation.json')
            if (path / 'watchdog_result.json').exists():
                watch = read_json(path / 'watchdog_result.json')
                reserved = watch.get('allocated_cpu_core_seconds',0.)
            else:
                reserved = len(alloc['cpu_affinity']) * max(0,time.time()-alloc['started_unix'])
            preflight = 0.
            for receipt in alloc.get('prior_costs',[]):
                key=(receipt['path'],receipt['sha256'])
                if key not in receipts_seen:
                    receipts_seen.add(key);preflight+=receipt['charged_cpu_core_seconds']
            charges.append({'task':item['id'],'reserved_cpu_core_seconds':reserved,
                            'unique_preflight_cpu_core_seconds':preflight})
        rows.append(row)
    total=sum(c['reserved_cpu_core_seconds']+c['unique_preflight_cpu_core_seconds'] for c in charges)/3600
    lines += ['## 资源与下一步', '', f'累计保守CPU核时：{total:.3f}；GPU：0。失败、预检和复测不隐去。', '',
              manifest['next_action'], '', '正常研究迭代无需逐轮审批；资源权限、规则和信息边界不自动扩大。', '']
    state={'updated_unix':time.time(),'tasks':rows,'charges':charges,'charged_cpu_core_hours':total,
           'gpu_hours':0,'next_action':manifest['next_action'],'retained':manifest.get('retained',[])}
    write_json(directory/'state.json',state)
    (directory/'REPORT.md').write_text('\n'.join(lines))
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--repo',required=True)
    a=p.parse_args()
    state=publish(a.root,a.repo)
    print({'tasks':[(t['id'],t['phase']) for t in state['tasks']],
           'charged_cpu_core_hours':state['charged_cpu_core_hours'],'next_action':state['next_action']})


if __name__=='__main__':
    main()
