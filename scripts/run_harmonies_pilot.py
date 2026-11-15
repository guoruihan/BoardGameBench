#!/usr/bin/env python3
"""Run the calibrated isolated native-agent pilot. Private routing stays external."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import time

from boardbench.families import pilot_gt, pilot_gateway
from boardbench.families.pilot_run import (
    NativePilotAdapter, PilotConfig, PilotRunner, PhaseContext, Budget, Checkpoints,
    blank_submission, prepare_public_material, export_known_catalog, gateway_callbacks,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--private-upstream', type=Path, required=True)
    parser.add_argument('--rules', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--acceptance', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    if time.time() >= args.deadline:
        raise RuntimeError('deadline already reached')
    from boardbench.families.evaluate import archive_source
    from boardbench.families.pilot_supervisor import runtime_inventory
    provenance = archive_source(root)
    (root/"source_identity.json").write_text(json.dumps(provenance, indent=2))
    identity = pilot_gt.initialize(root/'gt', args.pool)
    gt = pilot_gt.PilotService(root/'gt')
    gateway = pilot_gateway.Gateway(root/'gateway', json.loads(args.private_upstream.read_text()))
    servers = []
    with tempfile.TemporaryDirectory(prefix='pilot-sockets-') as sockets:
        gt_socket, model_socket = Path(sockets)/'gt.sock', Path(sockets)/'model.sock'
        servers = [pilot_gt.make_server(gt, unix_socket=gt_socket),
                   pilot_gateway.make_server(gateway, unix_socket=model_socket)]
        for server in servers:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        sdk = root/'public'
        prepare_public_material(sdk, Path(pilot_gt.__file__).with_name('pilot_client.py'), [args.rules])
        export_known_catalog(sdk/'rules')
        account, status = gateway_callbacks(gateway)
        adapter = NativePilotAdapter(gt_service=gt, gt_socket=gt_socket, gateway_socket=model_socket,
                                     sdk_dir=sdk, gateway_account=account, gateway_status=status,
                                     development_phase='smoke' if args.acceptance else 'development')
        (root/"runtime_inventory.json").write_text(json.dumps(runtime_inventory(adapter.runtime), indent=2))
        try:
            if args.acceptance:
                base = root/'acceptance'
                work, home = base/'work', base/'home'
                work.mkdir(parents=True); home.mkdir()
                blank_submission(work/'submission')
                budget = Budget(10, 12, 300000, 32000, 8192, 600, 2, 240, 4)
                prompt = (
                    'This is an isolated infrastructure acceptance test, not strategy development. '
                    'Read /bench/SDK.md and inspect the SDK source if needed. Use the Python SDK '
                    'to start one game. Write a Python program that takes at least five legal real '
                    'actions in one execution, always getting the current observation before each action. '
                    'Use any simple legal choices; playing strength is irrelevant. Run it. '
                    'Then write brief infrastructure-only notes in submission/START_HERE.md, keep the '
                    'existing manifest valid, call pilot_control.checkpoint() and pilot_control.finish(). '
                    'Do not open extra games or run more than ten chosen actions. '
                    'The Python executable is python3 and SDK PYTHONPATH is preconfigured.'
                )
                (work/'task_config.json').write_text(json.dumps({'phase':'acceptance','budget':asdict(budget)}))
                context = PhaseContext('acceptance',work,home,base/'ledger.json',budget,
                                       min(args.deadline,time.time()+600),prompt,Checkpoints(base/'checkpoints'))
                result = adapter.run_development(context,None)
                record = adapter.sessions['acceptance']
                report = gt.report(record['gt']['run_id'])
                output = {'controller':asdict(result),'gt':report,'model':gateway.status('acceptance'),
                          'checkpoint':str(context.checkpoints.freeze() or '')}
                (root/'acceptance.json').write_text(json.dumps(output,indent=2))
                print(json.dumps({'acceptance':asdict(result),'steps':report['steps'],'sources':report['sources'],
                                  'model':output['model']['usage'],'checkpoint':bool(output['checkpoint'])}),flush=True)
                if not result.finished or report['sources'].get('agent',0)+report['sources'].get('program',0)<5 or not output['checkpoint']:
                    raise RuntimeError('native acceptance did not meet all gates')
            else:
                config = PilotConfig(opponent_id=identity['policy_id'],opponent_sha256=identity['weights_sha256'],
                                     environment_revision=identity['environment_source_sha256'],codex_version='0.167.0',
                                     deadline_unix=args.deadline,calibration_seeds=tuple(
                                         pilot_gt.private_seed(gt.config['seed_key'],'calibration',i) for i in range(5)))
                print(json.dumps({'event':'pilot_start','deadline_unix':args.deadline,'opponent_id':identity['policy_id']}),flush=True)
                result = PilotRunner(root/'experiment',config,adapter).run()
                print(json.dumps(result),flush=True)
                from report_harmonies_pilot import summarize
                rows, summary = summarize(root/"experiment")
                (root/"experiment/summary.json").write_text(json.dumps(summary, indent=2))
        finally:
            for server in servers:
                server.shutdown(); server.server_close()
            gt.close(); gateway.close()


if __name__ == '__main__':
    main()
