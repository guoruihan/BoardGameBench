"""CPU-only finite job queue; resume skips successful immutable job records."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .play import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--cpus',required=True)
    a=p.parse_args();cfg=json.loads(a.manifest.read_text());cpus=[int(c) for c in a.cpus.split(',')]
    if not cpus or not set(cpus)<=os.sched_getaffinity(0):raise ValueError('Invalid worker affinity')
    groups=[cfg['jobs'][i::len(cpus)] for i in range(len(cpus))]
    def worker(pair):
        core,jobs=pair
        for job in jobs:
            directory=Path(cfg['output'])/'jobs'/job['id'];directory.mkdir(parents=True,exist_ok=True)
            done=directory/'done.json'
            if done.exists() and json.loads(done.read_text())['returncode']==0:continue
            began=time.time();cmd=['taskset','-c',str(core)]+job['argv']
            with (directory/'run.log').open('a') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write_json(directory/'active.json',dict(pid=child.pid,core=core,began_unix=began,argv=job['argv']))
                try:code=child.wait(timeout=job['timeout'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:code=child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
                    code=124
            write_json(done,dict(returncode=code,core=core,began_unix=began,ended_unix=time.time(),wall_seconds=time.time()-began))
            print(json.dumps(dict(job=job['id'],returncode=code)),flush=True)
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:list(pool.map(worker,zip(cpus,groups)))
    write_json(Path(cfg['output'])/(cfg['name']+'.complete.json'),dict(ended_unix=time.time(),jobs=len(cfg['jobs'])))

if __name__=='__main__':main()
