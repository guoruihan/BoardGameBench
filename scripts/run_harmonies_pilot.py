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
    parser.add_argument('--managed-acceptance', action='store_true', help='Force a one-call native quota exit while a managed job publishes')
    parser.add_argument('--frozen-submission', type=Path,
                        help='Evaluate this existing trusted frozen checkpoint without development')
    parser.add_argument('--deployment-profiles', nargs='+', choices=('L10', 'L20', 'L40'),
                        default=('L10', 'L20', 'L40'))
    parser.add_argument('--protocol-version')
    parser.add_argument('--model-id', choices=('gpt-6-astra','gpt-5.6-sol'), default='gpt-6-astra')
    args = parser.parse_args()
    if args.managed_acceptance:
        args.acceptance = True
    if args.acceptance and args.frozen_submission:
        parser.error('--acceptance and --frozen-submission are mutually exclusive')
    if len(set(args.deployment_profiles)) != len(args.deployment_profiles):
        parser.error('--deployment-profiles must be unique')
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
    upstream = json.loads(args.private_upstream.read_text())
    if upstream['model'].split('/')[-1] != args.model_id or upstream.get('reasoning_effort','medium') != 'medium':
        raise ValueError('private route and public model configuration differ')
    gateway = pilot_gateway.Gateway(root/'gateway', upstream)
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
                                     sdk_dir=sdk, gateway_account=account, gateway_status=status, gateway_close=gateway.close_model,
                                     development_phase='smoke' if args.acceptance else 'development', model_id=args.model_id)
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
                if args.managed_acceptance:
                    budget = Budget(1, 1, 30000, 8192, 8192, 180, 2, 60, 4)
                    (work/'finalize_gate.py').write_text(
                        "import time,pathlib,submission\n"
                        "from pilot_client import PilotClient\n"
                        "time.sleep(5)\n"
                        "client=PilotClient();client.new_game()\n"
                        "client.run(lambda obs,actions: 0,max_actions=5)\n"
                        "p=pathlib.Path('candidate');p.mkdir()\n"
                        "(p/'play.py').write_text('print(42)\\n')\n"
                        "print(submission.publish(directory='candidate',entrypoint='play.py'),flush=True)\n")
                    prompt = ("Infrastructure fault injection only. Execute exactly this command using your shell tool now: "
                              "python3 -c \"import jobs; print(jobs.start(['python3','finalize_gate.py']))\". "
                              "Do not inspect or edit files, do not wait, do not add any other commands. "
                              "The prewritten script performs the authorized smoke checks. "
                              "After that tool returns, say launched. A one-call model quota deliberately refuses the next request.")
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
                if args.managed_acceptance:
                    events = [json.loads(line) for line in (base/'jobs-0/events.jsonl').read_text().splitlines()]
                    exited = [e for e in events if e['event']=='controller_exit']
                    completed = [e for e in events if e['event']=='job_exit' and e['exit_code']==0]
                    latest = json.loads((base/'checkpoints/latest.json').read_text())
                    if (not exited or not completed or completed[-1]['time'] <= exited[-1]['time']
                            or latest['published_unix'] <= exited[-1]['time']
                            or output['model']['usage']['calls'] != 1):
                        raise RuntimeError('managed quota-exit publication gate failed')
                if (not args.managed_acceptance and not result.finished) or report['sources'].get('agent',0)+report['sources'].get('program',0)<5 or not output['checkpoint']:
                    raise RuntimeError('native acceptance did not meet all gates')
            else:
                config = PilotConfig(opponent_id=identity['policy_id'],opponent_sha256=identity['weights_sha256'],
                                     environment_revision=identity['environment_source_sha256'],codex_version='0.167.0',
                                     deadline_unix=args.deadline, model_id=args.model_id,
                                     deployment_profiles=tuple(args.deployment_profiles),
                                     **({'protocol_version': args.protocol_version} if args.protocol_version else {}),
                                     calibration_seeds=tuple(
                                         pilot_gt.private_seed(gt.config['seed_key'],'calibration',i) for i in range(5)))
                print(json.dumps({'event':'pilot_start','deadline_unix':args.deadline,'opponent_id':identity['policy_id']}),flush=True)
                runner = PilotRunner(root/'experiment',config,adapter)
                result = (runner.run_frozen(args.frozen_submission) if args.frozen_submission
                          else runner.run())
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
