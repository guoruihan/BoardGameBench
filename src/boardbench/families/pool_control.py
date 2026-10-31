"""Request a cooperative pause; never signal process groups or timeout wrappers."""
import argparse
import time
from pathlib import Path

from .play import write_json
from .pool_report import report


def pause(directory):
    result=[]
    for row in report(directory):
        path=Path(row['output'])
        if row['exit_code'] is not None:continue
        write_json(path/'PAUSE',dict(requested_unix=time.time(),reason='explicit_operator_pause'))
        result.append(dict(game=row['game'],arm=row['arm'],output=str(path)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['pause']);p.add_argument('directory',type=Path)
    args=p.parse_args()
    print(f'Cooperative pause requested for {len(pause(args.directory))} tasks; wait for stopped checkpoints.')


if __name__=='__main__':main()
