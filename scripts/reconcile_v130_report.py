"""One-time auditable reconciliation, only after both remote V130 processes exit.

No model, frozen source, trajectory or old cost receipt is changed. Old state
and report are preserved; the historical cost remains explicitly incomplete.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import time

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v130.pipeline import report


def reconcile(root):
    root=Path(root).resolve()
    if root.name!='teacher_guided_20261020_a' or root.parent.name!='v130':
        raise ValueError('explicit reviewed historical run only')
    probes=[]
    for host in ('192.168.2.58','192.168.2.11'):
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',
            '-o','GlobalKnownHostsFile=/dev/null','-o','UserKnownHostsFile=/tmp/codex_known_hosts_cluster',
            '-o','StrictHostKeyChecking=accept-new','-o','CheckHostIP=no','ruihan@'+host,
            "pgrep -af 'python.* -m boardbench[.]v130'"],capture_output=True,text=True,timeout=15)
        if result.returncode!=1 or result.stdout.strip():
            raise RuntimeError('historical process may still exist or SSH probe failed; refuse reconciliation')
        probes.append({'host':host,'no_v130_process':True,'checked_unix':time.time()})
    completed=[]
    for number in (1,2):
        for trial in sorted((root/f'round_{number:02d}/ppo/trials').glob('*')):
            summary=read_json(trial/'summary.json');meta=read_json(trial/'final.json')
            c=read_json(trial/'final_comparison.json')
            if summary['status']!='completed' or digest(trial/'final.pt')!=meta['sha256'] or not {'initial','latest','best'}<=set(c):
                raise ValueError('incomplete historical trial')
            completed.append(str(trial.relative_to(root)))
    if len(completed)!=16 or (root/'round_03').exists():
        raise ValueError('unexpected historical run layout')
    backup=root/'pre_v131_reconciliation'
    if backup.exists():raise ValueError('reconciliation already recorded; do not overwrite history')
    backup.mkdir()
    for name in ('state.json','REPORT.md','results.json','retained_best.json'):
        if (root/name).exists():shutil.copy2(root/name,backup/name)
    state=read_json(root/'state.json')
    state.update(phase='failed',stage='report publication race after round2; round3 superseded by V131',
        reconciliation={'checked_unix':time.time(),'probes':probes,'completed_trials':completed,
            'third_round_started':False,'old_cost_is_partial':True,'backup':str(backup),
            'replacement':'outputs/v131/continuation_guidance_20261020_a'})
    report(root,state)
    with (root/'REPORT.md').open('a') as stream:
        stream.write('\n## V131核查更正\n\n两轮共16个训练任务及final检查点完整；原外层报表竞态退出，第三轮未启动，现由V131新方法实验替代。原运行状态/报告已备份到pre_v131_reconciliation。历史费用含未收尾的第二轮控制器账本，是不完整记录，不能当最终精确成本。未改模型、冻结源码或旧轨迹。\n')
    write_json(root/'reconciliation.json',state['reconciliation'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    reconcile(p.parse_args().root)
