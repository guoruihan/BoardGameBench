"""Audit hashes, actual learner source equality and preserved V1 artifacts."""
import ast
from pathlib import Path

import torch

from boardbench.artifacts.store import read_json, write_json, digest, file_hashes, source_id, validate_checkpoint


def main():
    root=Path('.').resolve()
    plan=read_json('outputs/v11/experiment_plan/plan.json')
    report={'status':'passed','runtime_source_id':source_id(),'training':{},'policies':{},'v1_preserved':True}
    for game,spec in plan['games'].items():
        for algorithm in ('ppo','imitation'):
            if algorithm=='imitation' and not spec['imitation_episodes']:
                continue
            hashes=[]
            for seed in plan['training_seeds']:
                run=root/'outputs/v11'/game/f'{algorithm}_{seed}'
                config=read_json(run/'config.json')
                summary=read_json(run/'summary.json')
                assert summary['status'] in ('completed','wall_budget') and summary['parameter_changed']
                records=read_json(run/'checkpoints.json')
                initial=torch.load(run/records[0]['path'],map_location='cpu',weights_only=True)['state_dict']
                changed={}
                for checkpoint in records:
                    assert digest(run/checkpoint['path'])==checkpoint['sha256']
                    weights=torch.load(run/checkpoint['path'],map_location='cpu',weights_only=True)
                    if checkpoint['episodes']:
                        names=[name for name,value in weights['state_dict'].items() if not torch.equal(value,initial[name])]
                        assert any(name.startswith('actor') for name in names)
                        changed[checkpoint['path']]=names
                modules=['solvers/neural.py','solvers/action_features.py','environments/numerical.py',
                         'environments/board.py','environments/micro.py','environments/take_it_easy.py','environments/harmonies.py',
                         'training.py' if algorithm=='ppo' else 'imitation.py']
                fingerprints={relative:digest(run/'source/src/boardbench'/relative) for relative in modules}
                if algorithm=='imitation':
                    source=(run/'source/src/boardbench/solvers/board.py').read_text()
                    for node in ast.parse(source).body:
                        if isinstance(node,ast.FunctionDef) and node.name in ('ranked_actions','micro_value','tie_value'):
                            fingerprints['teacher:'+node.name]=ast.dump(node,include_attributes=False)
                hashes.append(fingerprints)
                report['training'][f'{game}/{algorithm}/{seed}']={
                    'actual_full_training_source_id':source_id(run/'source'),
                    'learner_modules':{k:v for k,v in fingerprints.items() if not k.startswith('teacher:')},
                    'changed_tensor_names':changed,'status':summary['status'],
                    'network':config['network'],'hidden':config['hidden']}
            assert all(x==hashes[0] for x in hashes),f'learner code changed across repetitions: {game}/{algorithm}'
        for path in (root/'outputs/v11'/game/'policies').glob('*/manifest.json'):
            m=read_json(path)
            assert digest(path.parent/m['config_path'])==m['config_sha256']
            assert digest(path.parent/m['artifact_path'])==m['artifact_sha256']
            validate_checkpoint(path.parent/m['checkpoint_path'])
            report['policies'][f'{game}/{m["policy_id"]}']='passed'
    assert len(report['training'])==15 and len(report['policies'])==39
    report['same_actual_learner_code_within_each_three_seed_group']=True
    for path in (root/'outputs/v1').glob('*/policies/*/export/checkpoints/ep_000000/manifest.json'):
        m=read_json(path)
        actual=file_hashes(path.parent)
        actual.pop('manifest.json')
        assert actual==m['files']
    old_zip=root/'deliverables/boardbench-v1-final.zip'
    assert digest(old_zip)=='753fbd3424b9192cecbeef14b357f0a1e987c884ab42d8e84d16f6391729bf35'
    write_json('runs/v11_artifact_audit.json',report)
    print('PASS: 15 training runs, 39 policies, same learner code, V1 untouched')


if __name__=='__main__':
    main()
