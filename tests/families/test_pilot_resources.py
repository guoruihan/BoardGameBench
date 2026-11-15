import json
from pathlib import Path
import runpy
import sqlite3
import pytest


@pytest.mark.parametrize('layout', ['main_c1/gateway', 'transport_gate'])
def test_usage_subsets_unknown_reservations_and_stopped_runs_are_retained(tmp_path, layout):
    run_name=layout.split('/')[0]
    directory=tmp_path/layout;directory.mkdir(parents=True)
    with sqlite3.connect(directory/'gateway.sqlite') as db:
        db.execute('CREATE TABLE accounts(id,data)')
        db.execute('CREATE TABLE requests(id,account,state,input_reserved,output_reserved,data)')
        db.execute('INSERT INTO accounts VALUES(?,?)',('development',json.dumps(dict(
            usage={'calls':2,'input_tokens':140,'output_tokens':60},first_binding_limit='unknown_usage',frozen=True))))
        usage={'input_tokens':100,'output_tokens':10,'input_tokens_details':{'cached_tokens':80},'output_tokens_details':{'reasoning_tokens':7}}
        db.execute('INSERT INTO requests VALUES(?,?,?,?,?,?)',('one','development','settled',120,50,json.dumps({'usage':usage})))
        db.execute('INSERT INTO requests VALUES(?,?,?,?,?,?)',('two','development','unknown',40,50,'{}'))
    phase=tmp_path/run_name/'experiment/phases/development';phase.mkdir(parents=True)
    (phase/'ledger.json').write_text(json.dumps(dict(cpu_seconds=12,active_wall_seconds=20,running=False,
        runs=[dict(cpu_seconds=12,active_wall_seconds=20,peak_rss_bytes=1024,
                   max_sampling_gap_seconds=.5,cpu_overshoot_seconds=.1,
                   cleanup_verified=True,reason='cpu_limit')])))
    audit=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/audit_pilot_resources.py'))['audit']
    result=audit(tmp_path)
    assert result['totals']['generation_attempts']==2
    assert result['totals']['input_tokens']==100 and result['totals']['output_tokens']==10
    assert result['totals']['cached_input_tokens']==80 and result['totals']['reasoning_output_tokens']==7
    assert result['totals']['reserved_input_tokens']==40 and result['totals']['reserved_output_tokens']==50
    assert result['accounts'][0]['run']==run_name and result['accounts'][0]['frozen']
    # Aggregate ledger totals already include their runs; don't count them twice.
    assert result['totals']['cpu_seconds']==12 and result['totals']['active_wall_seconds']==20
    assert result['resources'][0]['cleanup_verified']
    assert result['resources'][0]['max_cpu_overshoot_seconds']==.1
    assert result['resources'][0]['max_sampling_gap_seconds']==.5
