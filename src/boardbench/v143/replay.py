"""Public offline transitions plus independently checked per-seat PPO rewards."""
from boardbench.environments.harmonies_duo import HarmoniesDuo
from boardbench.v140.encoding import action_map


def replay(game,objective='margin'):
    if objective not in ('score','margin'):raise ValueError('unknown reward objective')
    env=HarmoniesDuo();obs,_=env.reset(game['seed']);pending={};count=0
    for row in game['decisions']:
        seat=row['seat']
        if obs['terminated'] or seat!=obs['active_player']: raise ValueError('seat/terminal mismatch')
        score=obs['score_breakdown'];potential=score[seat]['total']-(score[1-seat]['total'] if objective=='margin' else 0)
        trainable=row['trainable']
        if trainable!=(seat in game['learner_seats']):raise ValueError('wrong learner seat')
        if trainable:
            if seat in pending:
                old,reward=pending[seat]
                if abs((potential-old)/150-reward)>1e-10:raise ValueError('inter-decision reward mismatch')
            pending[seat]=(potential,row['reward_to_next_own_decision'])
        elif 'reward_to_next_own_decision' in row:raise ValueError('history action entered PPO')
        action=action_map(obs)[row['action']];next_obs=env.step(action).observation;count+=1
        yield dict(observation=obs,action=action,next_observation=next_obs,acting_player=seat,
            terminated=next_obs['terminated'],trainable=trainable,behavior_logp=row['logp'],
            reward_to_next_own_decision=row.get('reward_to_next_own_decision'))
        obs=next_obs
    if not count or not obs['terminated']:raise ValueError('incomplete match')
    for seat,(old,reward) in pending.items():
        final=obs['scores'][seat]-(obs['scores'][1-seat] if objective=='margin' else 0)
        if abs((final-old)/150-reward)>1e-10:raise ValueError('terminal reward mismatch')
    if (obs['scores'][game['seat']]!=game['score'] or obs['scores'][1-game['seat']]!=game['opponent_score']):
        raise ValueError('recorded score mismatch')
    env.validate()
