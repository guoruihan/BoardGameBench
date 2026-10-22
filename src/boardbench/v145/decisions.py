"""Development-driven mechanism choices. Confirmation never enters this module."""
from boardbench.v144.evaluation import paired_delta

KEYS=('mode','progress_scale','sil_weight','gae_lambda','learning_rate','entropy')


def first_round():
    base=dict(objective='score',gae_lambda=.95,progress_scale=0.,sil_weight=0.,replica=0)
    return [dict(base,name='control',mode='current'),dict(base,name='fixed',mode='fixed'),
        dict(base,name='plan_credit',mode='fixed',progress_scale=1.),
        dict(base,name='own_replay',mode='fixed',sil_weight=.05)]


def next_round(records,round_index):
    if round_index not in (1,2) or len(records)<2:raise ValueError('two completed development candidates required')
    if any(any(not 245000000<=g['seed']<246000000 for g in r['evaluation']['per_game']) for r in records):
        raise ValueError('confirmation cannot drive decisions')
    winner=max(records,key=lambda r:r['evaluation']['score_mean'])
    control=next((r for r in records if r['config']['name']=='control'),None)
    if control is None:control=min(records,key=lambda r:r['config']['progress_scale']+r['config']['sil_weight'])
    def arm(record,name):
        return dict(**{k:record['config'][k] for k in KEYS},name=name,objective='score',replica=0,
            warm_start=record['policy'],warm_start_sha256=record['sha256'])
    keep=arm(control,'control');keep.update(mode='current',progress_scale=0.,sil_weight=0.)
    base=arm(winner,'continue');contrast=paired_delta(winner['evaluation'],control['evaluation'])
    if all(base[k]==keep[k] for k in (*KEYS,'warm_start_sha256')):base['replica']=1
    if round_index==1:
        combined={**base,'name':'combine'}
        # Test the missing mechanism around the observed winner, not a new grid.
        if not base['progress_scale']:combined['progress_scale']=1.
        else:combined['sil_weight']=.05 if not base['sil_weight'] else 0.
        mixed={**base,'name':'opponent_mix','mode':'mixed' if base['mode']!='mixed' else 'fixed'}
        arms=[keep,base,combined,mixed]
        reason='keep measured winner; add missing animal-credit/replay mechanism and contrast fixed/moving opponent distribution'
    else:
        animal=contrast['metrics']['animal'];score=contrast['metrics']['score']
        credit={**base,'name':'credit_test','gae_lambda':1. if base['gae_lambda']<1. else .95}
        tune={**base,'name':'exploration_test','replica':1}
        if score['ci95'][0]>0 and animal['delta']>0:
            tune.update(learning_rate=base['learning_rate']*.5,entropy=base['entropy']*.5)
            reason='positive paired score and animal signal: smaller updates; independently test full-return credit'
        elif base['progress_scale'] and animal['delta']<=0:
            tune['progress_scale']=0.
            reason='animal-credit signal did not improve animals: ablate potential; independently test return horizon'
        else:
            tune['entropy']=min(.02,max(.005,base['entropy']*2))
            reason='no reliable joint score/animal signal: increase exploration; independently test return horizon'
        arms=[keep,base,credit,tune]
    return arms,dict(parent=winner['policy'],parent_score=winner['evaluation']['score_mean'],
        parent_animal=winner['evaluation']['animal_mean'],paired_vs_control=contrast,rationale=reason,
        uses_confirmation=False,warm_start_semantics='weights only; fresh Adam/RNG/replay; fixed anchor never updated')
