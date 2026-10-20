"""Six independent GPU jobs, frozen source and the existing bounded PPO runner."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from boardbench.artifacts.store import digest, read_json, source_files, source_id, write_json
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.launch import execute, validate_allocation, report as base_report
from boardbench.v130.pipeline import checkpoint_spec, paired


def build_jobs(config,repo,paired_dataset):
    repo=Path(repo);paired_dataset=Path(paired_dataset)
    dataset=repo/config['dataset'];manifest=repo/config['split_manifest'];parent=repo/config['parent']
    if digest(dataset)!=config['dataset_sha256'] or digest(manifest)!=config['split_sha256']:
        raise ValueError('pinned original dataset/split mismatch')
    paired_meta=read_json(paired_dataset.with_suffix('.json'))
    if digest(paired_dataset)!=paired_meta['sha256']:raise ValueError('paired dataset hash mismatch')
    export=read_json(paired_dataset.parent/'student.json')
    if export['parent_sha256']!=digest(parent) or export['sha256']!=paired_meta['checkpoint_sha256']:
        raise ValueError('guidance evaluated with a different parent policy')
    sets=read_json(manifest)['sets'];spec=checkpoint_spec(parent)
    count=read_json(parent.with_suffix('.json'))['ppo_actions']
    excluded=sets['train'][2176:2192]+sets['train'][2208:2224]+paired_meta['validation_seeds']
    jobs=[]
    for arm in ('control','habitat','student'):
        for replicate in range(2):
            trial={**config['training'],'research_version':'1.2.7','guidance_version':'1.2.11',
                'hidden':256,'seed':13101+replicate,'reward_scale':150.,
                'dataset':str(dataset),'dataset_sha256':config['dataset_sha256'],
                'split_manifest':str(manifest),'split_sha256':config['split_sha256'],
                'curve_seeds':sets['development'][:32],'final_seeds':sets['development'][32:96],
                'excluded_train_seeds':excluded,'bc_weight':0. if arm=='control' else config['guidance_weight'],
                'paired_dataset':{'path':str(paired_dataset),'sha256':paired_meta['sha256']},
                'guidance_z':config['guidance_z'],'guidance_minimum_gain':config['guidance_minimum_gain'],
                'guidance_gain_scale':config['guidance_gain_scale'],'minimum_guidance_rows':32,
                'initial_checkpoint':None,'initial_sha256':None,'arm':arm,'replicate':replicate,
                'continuation':spec,'target_actions':count+config['actions_per_round'],
                'milestones':[count+n for n in config['milestone_offsets']]}
            jobs.append((arm+'_'+'ab'[replicate],trial))
    return jobs


def report(root,state):
    root=Path(root)
    base_report(root,state)
    text=(root/'REPORT.md').read_text().replace('Harmonies V1.2.7 在线 PPO','Harmonies V1.2.11 配对优势指导 PPO')
    text=text.replace('四组各两个种子：BC初始化、软标签初始化、BC初始化+衰减模仿约束、随机初始化。初始冻结策略作为逐环境对照。',
        '三个分支各两个种子：纯PPO、habitat后续收益过滤、当前模型后续收益过滤；同一成熟父模型，教师不参与评测。')
    rows=[];retained=[]
    for name in state['jobs']:
        trial=root/'trials'/name
        if not (trial/'summary.json').exists() or not (trial/'final_comparison.json').exists():continue
        summary=read_json(trial/'summary.json');c=read_json(trial/'final_comparison.json')
        if not {'initial','latest','best'}<=set(c):continue
        row={'trial':name,'latest':c['latest']['mean_score'],'animal_mean':c['latest']['animal_mean'],
             'initial':c['initial']['mean_score'],'gain_from_parent':paired(c['latest'],c['initial']),
             'best64':c['best']['mean_score']}
        control=root/'trials'/('control_'+name[-1])/'final_comparison.json'
        if control.exists() and 'latest' in read_json(control):
            row['latest_minus_control']=paired(c['latest'],read_json(control)['latest'])
        rows.append(row)
        if all((trial/p).exists() for p in ('best.pt','best.json','final.pt','final.json')):
            retained.append({'trial':name,'curve_score':summary['best_curve_score'],'development64':c['best']['mean_score'],
                             **checkpoint_spec(trial/'best.pt')})
    write_json(root/'paired_results.json',rows)
    if retained:
        champion=max(retained,key=lambda r:r['curve_score'])
        write_json(root/'retained_best.json',{**champion,'selection':'32dev curve only, not final64 selection'})
    text+='\n## 固定终点的64开发环境对照\n\n|任务|总分|动物分|相对同种子PPO|\n|---|---:|---:|---:|\n'
    for r in rows:
        delta=r.get('latest_minus_control',{}).get('mean')
        text+=f"|{r['trial']}|{r['latest']:.2f}|{r['animal_mean']:.2f}|{delta if delta is not None else 'pending'}|\n"
    text+='\n父模型和开发集已参与历史筛选；此处是配对开发结果，不是未见正式测试。保留所有分支，报告不改变模型。\n'
    (root/'REPORT.md').write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('launch','resume','run','report'))
    p.add_argument('--repo',required=True);p.add_argument('--root',required=True)
    p.add_argument('--config');p.add_argument('--paired-dataset')
    a=p.parse_args();root,repo=Path(a.root).resolve(),Path(a.repo).resolve()
    if a.mode=='report':report(root,read_json(root/'state.json'));return
    if a.mode in ('launch','resume'):
        if os.uname().nodename!='a218.cluster.helixon.com' or not root.is_relative_to(repo/'outputs/v131'):
            raise ValueError('a218 and V131 output required')
        if a.mode=='launch':
            if root.exists():raise ValueError('refusing run overwrite')
            config=read_json(a.config);jobs=build_jobs(config,repo,Path(a.paired_dataset).resolve())
            cpus=physical_cpus()[:len(jobs)*(config['training']['workers']+1)+1]
            validate_allocation(cpus,config['gpu_uuids'],config['training']['workers']+1,config['jobs_per_gpu'],len(jobs))
            root.mkdir(parents=True);(root/'logs').mkdir()
            cfg={**config,'training':{**config['training'],'target_actions':jobs[0][1]['target_actions']}}
            write_json(root/'config.json',cfg);write_json(root/'jobs.json',jobs)
            for relative in source_files(repo):
                target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,target)
            write_json(root/'launch.json',{'cpus':cpus,'host':os.uname().nodename,'started_unix':time.time(),
                'source_id':source_id(root/'source'),
                'git_head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()})
        os.environ.update(PYTHONPATH=str(root/'source/src'),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',
                          OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        os.execv(sys.executable,[sys.executable,'-u','-m','boardbench.v131.launch','run','--repo',str(repo),'--root',str(root)])
    execute(root,repo,training_module='boardbench.v131.training',reporter=report)


if __name__=='__main__':main()
