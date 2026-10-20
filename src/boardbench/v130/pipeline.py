"""Three bounded, restartable teacher-correction/PPO rounds, using existing runners."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, source_files, source_id, write_json
from boardbench.v122.campaign import physical_cpus
from boardbench.v126.data import build as build_data
from boardbench.v127.launch import execute, validate_allocation, report as base_report
from boardbench.v130.training import export_checkpoint

ARMS = {'control': (0.,0.), 'static': (.05,0.), 'correct_low': (.05,.75), 'correct_high': (.2,.75)}
SSH = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','GlobalKnownHostsFile=/dev/null',
       '-o','UserKnownHostsFile=/tmp/codex_known_hosts_cluster','-o','StrictHostKeyChecking=accept-new',
       '-o','CheckHostIP=no','ruihan@192.168.2.58']


def checkpoint_spec(path):
    path = Path(path); meta = read_json(path.with_suffix('.json'))
    if digest(path) != meta['sha256']:
        raise ValueError('checkpoint checksum mismatch')
    return {'checkpoint':str(path), 'sha256':meta['sha256'], 'source_id':meta['source_id']}


def label_indices(round_number):
    # 64 games/round: first two roll-in modes x32env, then four policies x16env.
    return list(range(3000,3032)) if round_number == 1 else list(range(3032+(round_number-2)*16,3048+(round_number-2)*16))


def validation_indices(round_number):
    ids = label_indices(round_number)
    return ids[-4:] if round_number == 1 else ids[-2:]


def build_jobs(config, repo, round_number, previous, corrections):
    repo = Path(repo); manifest = repo/config['split_manifest']; dataset = repo/config['dataset']
    if digest(manifest) != config['split_sha256'] or digest(dataset) != config['dataset_sha256']:
        raise ValueError('pinned dataset/split mismatch')
    sets = read_json(manifest)['sets']
    # Reserve every future correction validation environment in ALL arms.
    excluded = sets['train'][2176:2192]+sets['train'][2208:2224]
    excluded += [sets['train'][i] for r in range(1,config['rounds']+1) for i in validation_indices(r)]
    jobs = []
    for kind, (weight, fraction) in ARMS.items():
        for replicate in range(2):
            name = kind+'_'+'ab'[replicate]
            parent = repo/config['parent'] if previous is None else Path(previous)/'trials'/name/'final.pt'
            spec = checkpoint_spec(parent)
            count = read_json(parent.with_suffix('.json'))['ppo_actions']
            trial = {**config['training'], 'research_version':'1.2.7', 'guidance_version':'1.2.10',
                     'hidden':256,'seed':13001+replicate,'reward_scale':150.,
                     'dataset':str(dataset),'dataset_sha256':config['dataset_sha256'],
                     'split_manifest':str(manifest),'split_sha256':config['split_sha256'],
                     'curve_seeds':sets['development'][:32],'final_seeds':sets['development'][32:96],
                     'excluded_train_seeds':excluded,'bc_weight':weight,'bc_decay_actions':500000,
                     'correction_fraction':fraction,'correction_datasets':corrections if fraction else [],
                     'initial_checkpoint':None,'initial_sha256':None,'arm':kind,'replicate':replicate,
                     'continuation':spec,'round':round_number,'target_actions':count+config['actions_per_round'],
                     'milestones':[count+n for n in config['milestone_offsets']]}
            jobs.append((name,trial))
    return jobs


def report_round(root, state):
    base_report(root, state)
    p = Path(root)/'REPORT.md'
    text = p.read_text().replace('Harmonies V1.2.7 在线 PPO','Harmonies V1.2.10 持续教师指导 PPO')
    text = text.replace('四组各两个种子：BC初始化、软标签初始化、BC初始化+衰减模仿约束、随机初始化。初始冻结策略作为逐环境对照。',
        '同一成熟父模型：PPO对照、原示范持续BC、新局面补标BC(.05)、新局面补标BC(.2)，各两个种子。教师不参与评测决策。')
    p.write_text(text)


def paired(left, right):
    a = {g['environment_seed']:g['score'] for g in left['games']}
    b = {g['environment_seed']:g['score'] for g in right['games']}
    if set(a) != set(b):
        raise ValueError('paired evaluation seeds differ')
    diff = np.array([a[s]-b[s] for s in sorted(a)])
    half = 1.96*diff.std(ddof=1)/np.sqrt(len(diff)) if len(diff)>1 else 0.
    return {'n':len(diff),'mean':float(diff.mean()),'descriptive95':[float(diff.mean()-half),float(diff.mean()+half)]}


def report(root, state):
    root = Path(root); rows = []; candidates = []
    cpu = state.get('controller_seconds',0.)+state.get('preflight_cpu_core_seconds',0.)
    gpu = state.get('preflight_gpu_seconds',0.)
    for attempt in sorted(root.glob('round_*/teacher_attempt_*')):
        if not attempt.is_dir(): continue
        if (attempt/'watchdog_result.json').exists():
            cpu += read_json(attempt/'watchdog_result.json')['allocated_cpu_core_seconds']
        elif (attempt/'allocation.json').exists():
            allocation=read_json(attempt/'allocation.json')
            cpu += len(allocation['cpu_affinity'])*max(0.,time.time()-allocation['started_unix'])
    for p in sorted(root.glob('round_*/ppo/state.json')):
        costs = read_json(p).get('cost',{})
        cpu += costs.get('cpu_core_hours',0)*3600; gpu += costs.get('gpu_hours',0)*3600
    for round_dir in sorted(root.glob('round_*')):
        trials = round_dir/'ppo/trials'
        for trial in sorted(trials.glob('*')):
            comparison = trial/'final_comparison.json'
            if not comparison.exists():
                continue
            c = read_json(comparison)
            if not {'latest','initial','best'} <= set(c):
                continue
            row = {'round':round_dir.name,'trial':trial.name,'latest':c['latest']['mean_score'],
                   'best':c['best']['mean_score'],'animals':c['best']['animal_mean'],
                   'gain_from_stage_parent':paired(c['latest'],c['initial'])}
            ctrl = trials/('control_'+trial.name[-1])/'final_comparison.json'
            if ctrl.exists() and 'latest' in read_json(ctrl):
                row['latest_minus_control'] = paired(c['latest'],read_json(ctrl)['latest'])
            rows.append(row)
            summary = read_json(trial/'summary.json')
            candidates.append({'curve_score':summary['best_curve_score'],'path':str(trial/'best.pt'),
                               'development64':c['best']['mean_score'],'trial':trial.name,'round':round_dir.name})
    state['cost'] = {'cpu_core_hours':cpu/3600,'gpu_hours':gpu/3600,
                     'basis':'all attempts + preflight; CPU reservations, physical GPU interval union; prior versions excluded'}
    write_json(root/'state.json',state)
    write_json(root/'results.json',rows)
    if candidates:
        champion = max(candidates,key=lambda c:c['curve_score'])
        champion.update(checkpoint_spec(champion['path']))
        champion['selection'] = '32-development curve only, NOT final64/test selection'
        write_json(root/'retained_best.json',champion)
    lines = ['# Harmonies V1.2.10 持续教师指导 PPO','',f"状态：{state['phase']}；当前阶段：{state.get('stage','—')}",
             '搜索仅用于训练补标；所有评测为纯模型自然终局。开发集重复使用，不是正式测试。',
             '总计三轮，每轮64补标局、8个独立PPO任务；同一最佳父模型出发，保留PPO与静态示范对照。',
             '', '|轮次|任务|latest/64dev|curve-best/64dev|best动物分|latest-同种子PPO对照|',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        delta = r.get('latest_minus_control',{}).get('mean')
        shown = '—' if delta is None else f'{delta:+.2f}'
        lines.append(f"|{r['round']}|{r['trial']}|{r['latest']:.2f}|{r['best']:.2f}|{r['animals']:.2f}|{shown}|")
    lines += ['',f"含预检累计资源：{state['cost']}",
              'latest用于固定训练终点对照；best只在32开发曲线上选，不按64环境结果挑checkpoint。',
              '各轮ppo/curves.svg和curves.json提供训练曲线；结果不足时保留父策略，不宣称已达到搜索。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def teacher_plan(config, root, repo, round_number, previous):
    root, repo = Path(root), Path(repo)
    names = ['correct_low_a','correct_low_b'] if round_number == 1 else ['correct_low_a','correct_low_b','correct_high_a','correct_high_b']
    arms = []
    for i,name in enumerate(names):
        parent = repo/config['parent'] if previous is None else Path(previous)/'trials'/name/'final.pt'
        exported = root/f'{name}.npz'; meta = export_checkpoint(parent,exported)
        arms.append({'id':name,'method':'terminal','multiplier':1,'timeout_seconds':config['teacher_job_seconds'],
            'params':{'mode':'timed','continuation':'dagger','seconds':config['teacher_game_seconds'],
                      'checkpoint':str(exported),'checkpoint_sha256':meta['sha256'],'stochastic':name.endswith('b')}})
    return {'split':'train_calibration','split_manifest':config['split_manifest'],
            'environment_indices':label_indices(round_number),'panels':[len(label_indices(round_number))],
            'search_seeds':[13020261020+round_number],'arms':arms,'cpu_cores':config['teacher_cores'],
            'resource_profile':'b35_reserve16_v1','task_seconds':config['teacher_task_seconds']}


def start_teacher(plan, directory, root, repo):
    plan_path = directory.with_suffix('.input.json'); write_json(plan_path,plan)
    session = 'bb130-'+Path(root).name+'-'+directory.parent.name+'-'+directory.name
    log = Path(root)/'logs'/(directory.parent.name+'_'+directory.name+'.log')
    env = ['env','PYTHONNOUSERSITE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
           'NUMEXPR_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONPATH='+str(Path(root)/'source/src')]
    command = shlex.join(env+[str(Path(repo)/'.venv-train/bin/python'),'-u','-m','boardbench.v130.campaign','launch',
                             '--repo',str(repo),'--root',str(directory),'--plan',str(plan_path)])
    script = command+' 2>&1 | tee '+shlex.quote(str(log))
    remote = shlex.join(['tmux','new-session','-d','-s',session,shlex.join(['bash','-o','pipefail','-c',script])])
    subprocess.run(SSH+[remote],check=True,timeout=20)
    return session


def run(root, repo):
    root, repo = Path(root), Path(repo); config = read_json(root/'config.json'); launch = read_json(root/'launch.json')
    if os.uname().nodename != 'a218.cluster.helixon.com' or source_id() != launch['source_id']:
        raise ValueError('a218 frozen-source execution required')
    os.sched_setaffinity(0,{launch['pipeline_cpu']})
    lock = (root/'pipeline.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = read_json(root/'state.json') if (root/'state.json').exists() else {
        'preflight_cpu_core_seconds':config.get('preflight_cpu_core_seconds',0.),
        'preflight_gpu_seconds':config.get('preflight_gpu_seconds',0.)}
    if state.get('phase') == 'completed':
        lock.close(); return
    started = time.monotonic(); prior_seconds = state.get('controller_seconds',0.); stopping = [False]
    prior = {sig:signal.signal(sig,lambda *_:stopping.__setitem__(0,True)) for sig in (signal.SIGINT,signal.SIGTERM)}
    def publish(stage, phase='running'):
        state.update(phase=phase,stage=stage,pid=os.getpid(),updated_unix=time.time(),
                     controller_seconds=prior_seconds+time.monotonic()-started)
        report(root,state)
    corrections = []; previous = None
    try:
        for number in range(1,config['rounds']+1):
            if stopping[0]: break
            stage = root/f'round_{number:02d}'; stage.mkdir(exist_ok=True)
            publish(f'round{number}: teacher')
            ready = stage/'teacher_ready.json'
            if not ready.exists():
                plan = teacher_plan(config,stage,repo,number,previous)
                attempts = sorted(stage.glob('teacher_attempt_*'))
                attempts = [p for p in attempts if p.is_dir()]
                consumed = sum(read_json(p/'watchdog_result.json')['elapsed_seconds'] for p in attempts if (p/'watchdog_result.json').exists())
                if attempts and not (attempts[-1]/'watchdog_result.json').exists():
                    teacher = attempts[-1]  # Resume observing existing bounded remote task, never duplicate it.
                elif attempts and read_json(attempts[-1]/'summary.json')['completed'] == 64:
                    teacher = attempts[-1]
                else:
                    if consumed >= config['teacher_task_seconds']-30:
                        raise RuntimeError('teacher task budget exhausted; saved progress, no automatic budget reset')
                    if attempts:
                        plan['resume_from'] = str(attempts[-1])
                    plan['task_seconds'] = config['teacher_task_seconds']-consumed
                    teacher = stage/f'teacher_attempt_{len(attempts)+1:02d}'
                    session = start_teacher(plan,teacher,root,repo)
                    write_json(stage/'teacher_launch.json',{'root':str(teacher),'session':session,'unix':time.time()})
                deadline = time.monotonic()+config['teacher_task_seconds']+60
                while not (teacher/'watchdog_result.json').exists():
                    if stopping[0]: break
                    if time.monotonic()>deadline:
                        raise RuntimeError('remote watchdog receipt missing; inspect named tmux before retrying')
                    publish(f'round{number}: teacher running'); time.sleep(10)
                if stopping[0]: break
                outcome = read_json(teacher/'summary.json')
                if outcome['completed'] != 64 or read_json(teacher/'status.json')['phase'] != 'completed':
                    raise RuntimeError('incomplete teacher attempt; preserve and resume within remaining task budget')
                write_json(ready,{'root':str(teacher),'plan_sha256':digest(teacher/'plan.json')})
            teacher = Path(read_json(ready)['root'])
            if digest(teacher/'plan.json') != read_json(ready)['plan_sha256']:
                raise ValueError('teacher ready plan checksum mismatch')
            dataset = stage/'corrections.npz'
            if not dataset.exists():
                sets = read_json(repo/config['split_manifest'])['sets']
                val = [sets['train'][i] for i in validation_indices(number)]
                publish(f'round{number}: replay and dataset')
                build_data([teacher],dataset,val,minimum_teacher_rounds=2)
            meta = read_json(dataset.with_suffix('.json'))
            if digest(dataset) != meta['sha256']:
                raise ValueError('correction dataset checksum mismatch')
            corrections.append({'path':str(dataset),'sha256':meta['sha256']})
            train_root = stage/'ppo'
            if not train_root.exists():
                jobs = build_jobs(config,repo,number,previous,deepcopy(corrections))
                train_root.mkdir(); (train_root/'logs').mkdir()
                (train_root/'source').symlink_to(root/'source',target_is_directory=True)
                cfg = {**config,'training':{**config['training'],'target_actions':max(c['target_actions'] for _,c in jobs)},
                       'preflight_cpu_core_seconds':0.,'preflight_gpu_seconds':0.}
                write_json(train_root/'config.json',cfg); write_json(train_root/'jobs.json',jobs)
                write_json(train_root/'launch.json',{**launch,'cpus':launch['training_cpus']})
            finished = ((train_root/'state.json').exists() and
                        read_json(train_root/'state.json').get('phase')=='finished' and
                        len(read_json(train_root/'state.json')['jobs'])==8 and
                        all(j['status']=='completed' for j in read_json(train_root/'state.json')['jobs'].values()))
            if finished:
                previous=train_root
                publish(f'round{number}: reused complete stage')
                continue
            publish(f'round{number}: PPO')
            # Run separate frozen controller so its CPU affinity/signals cannot alter the pipeline.
            env = {**os.environ,'PYTHONPATH':str(root/'source/src')}
            with (root/'logs'/f'round_{number:02d}_ppo.log').open('a') as stream:
                child = subprocess.Popen([sys.executable,'-u','-m','boardbench.v130.pipeline','train',
                    '--repo',str(repo),'--root',str(train_root)],env=env,stdout=stream,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if stopping[0]: child.terminate()
                    publish(f'round{number}: PPO running'); time.sleep(10)
                if child.returncode:
                    raise RuntimeError('PPO controller failed; see stage logs')
            if stopping[0]: break
            s = read_json(train_root/'state.json')
            if any(j['status'] != 'completed' for j in s['jobs'].values()) or len(s['jobs']) != 8:
                raise RuntimeError('PPO task failed or reached time limit; no automatic stage advancement')
            previous = train_root
            publish(f'round{number}: completed')
        publish('saved at task boundary' if stopping[0] else 'three rounds completed', 'interrupted' if stopping[0] else 'completed')
    except BaseException as exc:
        state['error'] = repr(exc); publish(state.get('stage','initializing'),'failed'); raise
    finally:
        lock.close()
        for sig,handler in prior.items(): signal.signal(sig,handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('launch','resume','run','train','report'))
    p.add_argument('--repo',required=True); p.add_argument('--root',required=True); p.add_argument('--config')
    a = p.parse_args(); repo,root = Path(a.repo).resolve(),Path(a.root).resolve()
    if a.mode == 'report': report(root,read_json(root/'state.json')); return
    if a.mode == 'train': execute(root,repo,training_module='boardbench.v130.training',reporter=report_round); return
    if a.mode in ('launch','resume'):
        if os.uname().nodename != 'a218.cluster.helixon.com' or not root.is_relative_to(repo/'outputs/v130'):
            raise ValueError('a218 and project v130 output required')
        if a.mode == 'launch':
            if root.exists(): raise ValueError('refusing pipeline overwrite')
            config = read_json(a.config); cpus = physical_cpus()
            count = 8*(config['training']['workers']+1)+1
            validate_allocation(cpus[:count],config['gpu_uuids'],config['training']['workers']+1,config['jobs_per_gpu'])
            if config['rounds'] != 3 or count+1>min(128,len(cpus)) or config['teacher_cores'] != 65:
                raise ValueError('explicit three-round resource contract required')
            checkpoint_spec(repo/config['parent'])
            root.mkdir(parents=True); (root/'logs').mkdir()
            write_json(root/'config.json',config)
            for relative in source_files(repo):
                target=root/'source'/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(repo/relative,target)
            write_json(root/'launch.json',{'source_id':source_id(root/'source'),'training_cpus':cpus[:count],
                'pipeline_cpu':cpus[count],'started_unix':time.time(),'host':os.uname().nodename,
                'git_head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()})
        os.environ.update(PYTHONPATH=str(root/'source/src'),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',
                          OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        os.execv(sys.executable,[sys.executable,'-u','-m','boardbench.v130.pipeline','run','--repo',str(repo),'--root',str(root)])
    run(root,repo)


if __name__ == '__main__': main()
