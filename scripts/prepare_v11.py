"""Materialize the predeclared V1.1 experiment matrix, without running any test."""
from pathlib import Path
from boardbench.artifacts.store import read_json, write_json, digest


def main():
    path=Path('configs/v11_plan.json')
    plan=read_json(path)
    out=Path('outputs/v11/experiment_plan')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'plan.json',plan)
    write_json(out/'provenance.json',{'plan_sha256':digest(path),'test_used':False})
    validation=list(range(plan['validation_start'],plan['validation_start']+plan['validation_count']))
    test=list(range(plan['test_start'],plan['test_start']+plan['test_count']))
    for game,spec in plan['games'].items():
        for seed in plan['training_seeds']:
            for algorithm,episodes in [('ppo',spec['ppo_episodes']),('imitation',spec['imitation_episodes'])]:
                if not episodes:
                    continue
                config=dict(algorithm=algorithm,task_id=game,training_seed=seed,episodes=episodes,
                            train_env_seed_start=spec['train_start'],validation_seeds=validation,test_seeds=test,
                            device='cuda',hidden=spec['hidden'],network=spec['ppo_network'],threads=2,
                            batch_episodes=32,checkpoint_every=episodes//4,epochs=4,minibatch_size=256,
                            learning_rate=.001 if algorithm=='imitation' else .0003,
                            max_wall_seconds=plan['per_training_run_wall_seconds'])
                write_json(out/f'{game}_{algorithm}_{seed}.json',config)
    print(out)


if __name__=='__main__':
    main()
