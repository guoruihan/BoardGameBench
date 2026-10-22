"""Public owned-animal plan potential in raw score units; no hidden futures."""
import numpy as np
from boardbench.v134.plans import GOAL_DIM,PLAN_DIM
from boardbench.v140.encoding import BASE


def plan_potential(x):
    plans=np.asarray(x)[...,3*BASE:3*BASE+PLAN_DIM-1].reshape(*np.asarray(x).shape[:-1],2,GOAL_DIM)
    # Existing planner has at most two compatible goals. A market-only card
    # earns no potential: initially nobody owns a card, so Phi(initial)=0.
    present=plans[...,368:400].sum(-1)>0
    owned=plans[...,-1]<.5
    return (10*plans[...,-2]*present*owned).sum(-1)


def credit(raw_rewards,values,potentials,scale,lam):
    from boardbench.v127.training import advantages
    phi=np.asarray(potentials,dtype=np.float64)*scale/150
    raw=np.asarray(raw_rewards,dtype=np.float64)
    if len(phi)!=len(raw) or abs(phi[0])>1e-10:raise ValueError('complete own episode with zero initial potential required')
    shaped=raw+np.diff(np.r_[phi,0.])
    if abs(shaped.sum()-raw.sum())>1e-10:raise RuntimeError('potential feedback changed total return')
    adv,targets=advantages(shaped.tolist(),values,lam)
    mc=np.cumsum(raw[::-1])[::-1]-phi
    return shaped,adv,targets,mc


def visible_cards(x):
    # Public one-hot market IDs: board23*17, bundles18, counters18 =>427.
    ids=[]
    for offset in (0,BASE):ids.extend(np.asarray(x)[offset+427:offset+526].reshape(3,33).argmax(-1).tolist())
    return set(ids)-{0}
