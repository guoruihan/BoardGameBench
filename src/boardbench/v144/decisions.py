"""Small auditable result-driven experiment choices, not an unbounded scheduler."""
from copy import deepcopy
from .evaluation import paired_delta


def first_round():
    return [dict(name='score_current',objective='score',mode='current',gae_lambda=.95,replica=0),
            dict(name='score_history',objective='score',mode='league',gae_lambda=.95,replica=0),
            dict(name='score_fullreturn',objective='score',mode='current',gae_lambda=1.,replica=0),
            dict(name='margin_control',objective='margin',mode='current',gae_lambda=.95,replica=0)]


def next_round(records,round_index):
    """records contain config, fresh DEVELOPMENT result, retained policy and SHA."""
    eligible=[r for r in records if r['config']['objective']=='score']
    if not eligible:raise ValueError('no own-score parent')
    winner=max(eligible,key=lambda r:r['evaluation']['score_mean'])
    control=next(r for r in records if r['config']['objective']=='margin')
    contrast=paired_delta(winner['evaluation'],control['evaluation'])
    convincing=contrast['metrics']['score']['ci95'][0]>0
    c=winner['config']
    inherited={k:c[k] for k in ('mode','gae_lambda','learning_rate','entropy')}
    parent=dict(warm_start=winner['policy'],warm_start_sha256=winner['sha256'])
    base=dict(**inherited,**parent,objective='score',replica=0)
    # One exact hyperparameter continuation, one opponent-distribution change,
    # and one credit/exploration response. All round warm starts use fresh Adam.
    arms=[dict(base,name='score_continue'),dict(base,name='score_diverse')]
    arms[1]['mode']='diverse'
    if c['mode']=='diverse':arms[1]['mode']='current'  # Actual distribution contrast.
    challenger=dict(base,name='score_credit' if round_index==1 else 'score_explore')
    if round_index==1:
        challenger['gae_lambda']=.95 if c['gae_lambda']==1. else 1.
        rationale='flip full-return versus bootstrapped credit assignment around the measured winner'
    elif convincing:
        challenger.update(learning_rate=c['learning_rate']*.5,entropy=c['entropy']*.5,replica=1)
        rationale='reliable score advantage: test smaller updates and reduced exploration with a new RNG replica'
    else:
        challenger.update(entropy=min(.02,c['entropy']*2),replica=1)
        rationale='no reliable score advantage: test greater exploration with a new RNG replica'
    arms.append(challenger)
    ctrl={k:control['config'][k] for k in ('mode','gae_lambda','learning_rate','entropy')}
    arms.append(dict(ctrl,name='margin_control',objective='margin',replica=0,
        warm_start=control['policy'],warm_start_sha256=control['sha256']))
    return arms,dict(round_index=round_index,parent=winner['policy'],parent_score=winner['evaluation']['score_mean'],
        parent_animal=winner['evaluation']['animal_mean'],paired_vs_margin=contrast,
        rationale=rationale,opponent_variant=arms[1]['mode'],uses_confirmation=False,
        warm_start_semantics='weights only; fresh Adam, training RNG and history, fixed evaluation opponents unchanged')
