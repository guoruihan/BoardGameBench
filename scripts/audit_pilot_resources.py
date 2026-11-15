#!/usr/bin/env python3
"""Read trusted pilot ledgers, including stopped configuration trials and probes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def audit(root):
    root = Path(root)
    accounts, resources, gt = [], [], []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        database = directory/'gateway/gateway.sqlite'
        if not database.is_file():
            database = directory/'gateway.sqlite'  # standalone transport gates
        if database.is_file():
            with sqlite3.connect(str(database)) as connection:
                connection.row_factory = sqlite3.Row
                for account in connection.execute('SELECT id,data FROM accounts'):
                    data=json.loads(account['data'])
                    known=Counter(input_tokens=0,output_tokens=0,cached_input_tokens=0,reasoning_output_tokens=0)
                    reserved=Counter(input_tokens=0,output_tokens=0)
                    states=Counter()
                    for request in connection.execute('SELECT * FROM requests WHERE account=?',(account['id'],)):
                        states[request['state']]+=1
                        detail=json.loads(request['data'])
                        if request['state']=='settled':
                            usage=detail['usage']
                            for key in ('input_tokens','output_tokens'): known[key]+=usage[key]
                            known['cached_input_tokens']+=(usage.get('input_tokens_details') or {}).get('cached_tokens',0) or 0
                            known['reasoning_output_tokens']+=(usage.get('output_tokens_details') or {}).get('reasoning_tokens',0) or 0
                        else:
                            reserved['input_tokens']+=request['input_reserved']
                            reserved['output_tokens']+=request['output_reserved']
                    accounts.append(dict(run=directory.name,account=account['id'],generation_attempts=sum(states.values()),
                        request_states=dict(states),known_usage=dict(known),unsettled_reservations=dict(reserved),
                        ledger_usage=data['usage'],first_binding_limit=data['first_binding_limit'],frozen=data['frozen']))
        candidates=[directory/'ledger.json',directory/'acceptance/ledger.json']
        candidates+=list((directory/'experiment/phases').glob('*/ledger.json'))
        for path in candidates:
            if path.is_file():
                value=json.loads(path.read_text())
                resources.append(dict(run=directory.name,phase=path.parent.name,cpu_seconds=value.get('cpu_seconds',0),
                    active_wall_seconds=value.get('active_wall_seconds',0),running=value.get('running',False),
                    gateway_settlement_wall_seconds=value.get('gateway_settlement_wall_seconds',0),
                    peak_rss_bytes=max((x.get('peak_rss_bytes',0) for x in value.get('runs',[])),default=0),
                    max_sampling_gap_seconds=max((x.get('max_sampling_gap_seconds',0) for x in value.get('runs',[])),default=0),
                    max_cpu_overshoot_seconds=max((x.get('cpu_overshoot_seconds',0) for x in value.get('runs',[])),default=0),
                    max_wall_overshoot_seconds=max((x.get('wall_overshoot_seconds',0) for x in value.get('runs',[])),default=0),
                    max_rss_overshoot_bytes=max((x.get('rss_overshoot_bytes',0) for x in value.get('runs',[])),default=0),
                    max_termination_latency_seconds=max((x.get('termination_latency_seconds',0) for x in value.get('runs',[])),default=0),
                    cleanup_verified=not value.get('running',False) and all(x.get('cleanup_verified',False) for x in value.get('runs',[])),
                    controller_stops=[x['reason'] for x in value.get('runs',[])]))
        database=directory/'gt/ledger.sqlite'
        if database.is_file():
            with sqlite3.connect(str(database)) as connection:
                for (raw,) in connection.execute('SELECT data FROM runs'):
                    value=json.loads(raw)
                    gt.append(dict(run=directory.name,phase=value['phase'],games_started=value['games'],
                        status=value['status'],natural_completed=sum(x.get('natural',False) for x in value['outcomes']),
                        service_wall_seconds=value['service_wall_seconds'],opponent_wall_seconds=value['opponent_seconds']))
    # The initial routing probe predates the gateway; account for it once.
    initial=root/'private/gateway_preflight_authenticated.json'
    if initial.exists():
        for record in json.loads(initial.read_text()):
            if record['endpoint']=='responses' and record['status']==200:
                usage=record['response']['usage']
                accounts.append(dict(run='initial_route_probe',account='probe',generation_attempts=1,
                    request_states={'settled':1},known_usage={'input_tokens':usage['input_tokens'],'output_tokens':usage['output_tokens'],
                    'cached_input_tokens':(usage.get('input_tokens_details')or{}).get('cached_tokens',0),
                    'reasoning_output_tokens':(usage.get('output_tokens_details')or{}).get('reasoning_tokens',0)},
                    unsettled_reservations={'input_tokens':0,'output_tokens':0},ledger_usage=usage,
                    first_binding_limit=None,frozen=False))
    totals=Counter(generation_attempts=0,input_tokens=0,output_tokens=0,cached_input_tokens=0,reasoning_output_tokens=0,
                   reserved_input_tokens=0,reserved_output_tokens=0,cpu_seconds=0,active_wall_seconds=0)
    for account in accounts:
        totals['generation_attempts']+=account['generation_attempts']
        totals.update(account['known_usage'])
        for key,value in account['unsettled_reservations'].items(): totals['reserved_'+key]+=value
    for resource in resources:
        for key in ('cpu_seconds','active_wall_seconds'): totals[key]+=resource[key]
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(),totals=dict(totals),accounts=accounts,
        resources=resources,gt=gt,limitations=[
            'Cache and reasoning fields are subsets of input/output totals, never additional tokens.',
            'Known usage excludes unsettled reservations; pending/unknown requests are retained separately.',
            'Unsettled reservations are estimated quota holds, not verified billed usage or a guaranteed billing upper bound.',
            'Controller wall time is summed across parallel phases, not elapsed makespan.',
            'Live running controllers have not settled their current CPU/wall usage; live totals are incomplete.',
            'Sampled agent compute and recorded acceptance gates only; implementation-assistant usage, host pytest/setup CPU, and trusted service CPU are not included.',
            'GT/opponent wall is reported separately; no invoice or monetary-cost claim.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    result=audit(args.root)
    (args.root/'resource_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['totals'],indent=2))


if __name__=='__main__':main()
