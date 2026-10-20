"""Small read-only-to-input synthesis of V131 diagnostics, PPO and total cost."""
import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v131.counterfactual import paired_stats


def write_curves(root):
    curves=read_json(root/'ppo/curves.json');config=read_json(root/'ppo/config.json')
    offset=config['training']['target_actions']-config['actions_per_round']
    points=[p for values in curves.values() for p in values]
    if not points:return
    low=min(p['score'] for p in points)-2;high=max(p['score'] for p in points)+2
    colors=['#2563eb','#60a5fa','#dc2626','#f87171','#059669','#34d399']
    x=lambda actions:65+675*(actions-offset)/config['actions_per_round']
    y=lambda score:375-300*(score-low)/(high-low)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="440" viewBox="0 0 960 440">',
         '<rect width="960" height="440" fill="white"/>',
         '<text x="65" y="28" font-family="sans-serif" font-size="17">V1.2.11 model-only greedy score (32 development games)</text>']
    for value in np.linspace(low,high,6):
        svg.append(f'<path d="M65 {y(value):.2f} H740" stroke="#ddd"/><text x="20" y="{y(value)+4:.2f}" font-size="12">{value:.1f}</text>')
    for i,(name,values) in enumerate(curves.items()):
        coordinates=' '.join(f"{x(p['actions']):.2f},{y(p['score']):.2f}" for p in values)
        svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')
        svg.append(f'<text x="770" y="{90+28*i}" font-family="sans-serif" fill="{colors[i]}">{name}</text>')
        for p in values:svg.append(f'<circle cx="{x(p["actions"]):.2f}" cy="{y(p["score"]):.2f}" r="3" fill="{colors[i]}"/>')
    for fraction in (0,.25,.5,.75,1):
        svg.append(f'<text x="{65+675*fraction-10:.2f}" y="398" font-size="12">{fraction*config["actions_per_round"]/1e6:g}M</text>')
    svg+=['<text x="220" y="425" font-family="sans-serif">Additional PPO actions from the common frozen parent</text>','</svg>']
    (root/'learning_curves.svg').write_text('\n'.join(svg))


def make_report(root):
    root=Path(root).resolve();ppo=root/'ppo'
    diagnostic=read_json(root/'diagnostic/summary.json')
    labels=read_json(root/'labels/summary.json')
    state=read_json(ppo/'state.json')
    phase=state['phase']
    if phase=='running' and time.time()-state['updated_unix']>60:phase='stale_heartbeat_check_writer_process'
    total=dict(state['cost'])
    cpu=sum(read_json(root/name/'watchdog_result.json')['allocated_cpu_core_seconds']
            for name in ('diagnostic','labels'))/3600
    total['cpu_core_hours']+=cpu
    total['basis']='PPO reservation + CPU diagnostic/labels + conservative preflight once; physical GPU interval union'
    extra=None
    if (root/'development_extra32/cost.json').exists():
        cost=read_json(root/'development_extra32/cost.json')
        for key in ('cpu_core_hours','gpu_hours'):total[key]+=cost[key]
        total['basis']+=' + additional development evaluation'
    if (root/'development_extra32/summary.json').exists():extra=read_json(root/'development_extra32/summary.json')
    grouped=defaultdict(list);checkpoints=[]
    for trial in sorted((ppo/'trials').glob('*')):
        if not (trial/'summary.json').exists() or not (trial/'final_comparison.json').exists():continue
        summary=read_json(trial/'summary.json');c=read_json(trial/'final_comparison.json')
        if summary['status']!='completed' or not {'initial','latest','best'}<=set(c):continue
        meta=read_json(trial/'final.json')
        if digest(trial/'final.pt')!=meta['sha256']:raise ValueError('final checkpoint checksum mismatch')
        checkpoints.append({'trial':trial.name,'path':str(trial/'final.pt'),'sha256':meta['sha256'],
                            'score':c['latest']['mean_score'],'animal_mean':c['latest']['animal_mean']})
        grouped[trial.name.rsplit('_',1)[0]].append(c['latest'])
    arms={}
    for arm,replicates in grouped.items():
        if len(replicates)!=2:continue
        arms[arm]={'training_seeds':2,'score':float(np.mean([r['mean_score'] for r in replicates])),
                   'animal_score':float(np.mean([r['animal_mean'] for r in replicates]))}
        if len(grouped.get('control',[]))==2:
            # Average training seeds within each environment BEFORE uncertainty.
            deltas=defaultdict(list)
            for left,right in zip(replicates,grouped['control']):
                a={g['environment_seed']:g['score'] for g in left['games']}
                b={g['environment_seed']:g['score'] for g in right['games']}
                if a.keys()!=b.keys():raise ValueError('unpaired development environments')
                for seed in a:deltas[seed].append(a[seed]-b[seed])
            arms[arm]['minus_control']=paired_stats([np.mean(v) for v in deltas.values()])
    complete=len(checkpoints)==6 and all(j['status']=='completed' for j in state['jobs'].values())
    if phase=='finished' and not complete:phase='incomplete'
    output={'phase':'completed' if complete else phase,'updated_unix':time.time(),
            'diagnostic':diagnostic,'labels':labels,'arms':arms,'additional_development':extra,'cost':total,'checkpoints':checkpoints,
            'selection_warning':'All scores are reused development data. Two training seeds, not a formal generalization claim.'}
    write_json(root/'summary.json',output)
    lines=['# Harmonies V1.2.11 实验结果','',f"状态：{output['phase']}；完整检查点 {len(checkpoints)}/6。",
        '','## 独立配对诊断','',
        f"64状态/16环境：habitat后续优势 {diagnostic['habitat']['mean']:+.3f}，学生后续优势 {diagnostic['student']['mean']:+.3f}。",
        f"768状态/48环境复核：habitat {labels['habitat']['mean']:+.3f}，学生 {labels['student']['mean']:+.3f}；只说明训练分歧状态的反事实收益，不是整局策略分数。",
        '','## 纯模型固定终点，同64开发环境，两个训练种子平均','',
        '|方法|总分|动物分|相对PPO|描述性配对95%区间|','|---|---:|---:|---:|---|']
    for name,arm in arms.items():
        delta=arm.get('minus_control',{})
        interval=f"[{delta['lower']:+.2f}, {delta['upper']:+.2f}]" if delta else 'pending'
        lines.append(f"|{name}|{arm['score']:.3f}|{arm['animal_score']:.3f}|{delta.get('mean','pending')}|{interval}|")
    if extra:
        lines+=['','## 冻结模型后的32个额外开发环境','',
                f"父模型：{extra['parent']:.3f}。所有六个固定终点都评估，未按此面板挑checkpoint；不是正式测试。",'',
                '|方法|总分|动物分|相对PPO|描述性配对95%区间|','|---|---:|---:|---:|---|']
        for name,r in extra['arms'].items():
            d=r['minus_control']
            lines.append(f"|{name}|{r['mean_score']:.3f}|{r['animal_mean']:.3f}|{d['mean']:+.3f}|[{d['lower']:+.2f}, {d['upper']:+.2f}]|")
    lines+=['','曲线和逐局原始数据：ppo/curves.svg、ppo/curves.json、ppo/trials/*/evaluations/*.json。',
            '所有分支的final和curve-best保留；retained_best仅按32dev曲线选择。正式test/confirmation未参与。',
            '','## 累计资源','',f"CPU {total['cpu_core_hours']:.3f} 核时；物理GPU {total['gpu_hours']:.4f} 卡时。",
            '包括诊断、标签、训练/评估/保存/清理及保守预检；同卡并发不重复算物理卡时。',
            '','## 解释边界','',
            '教师后续不匹配已经在独立未来样本中得到支持；筛选后的指导是否提高完整策略，必须看上表，不能用配对局部优势替代。',
            'habitat只有130条fit、student只有41条fit；两组数据量不同，不能把差异全归因于一个机制。',
            '开发环境和父模型已经反复筛选，描述性区间未校正模型选择；两个训练种子仍不足以确认普遍收益。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_curves(root)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    a=p.parse_args();result=make_report(a.root)
    print({'phase':result['phase'],'arms':result['arms'],'cost':result['cost']})
