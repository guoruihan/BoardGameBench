import pytest
from boardbench.families.registry import game_class
from boardbench.families.reference_search import TerminalReference
from boardbench.families.public_world import sampled_world


def test_terminal_search_complete_round_and_real_engine_untouched():
    c=game_class('toy_battle');e=c();e.reset(17221);before=e.snapshot()
    search=TerminalReference(c,{'kind':'heuristic'},seat=0,seconds=900,max_rounds=1,width=3)
    a=search.decide(e)
    assert a in e.legal_actions() and e.snapshot()==before
    assert search.last['completed_rounds']==1
    assert all(x['samples']==1 and x['terminal_scores'] is not None for x in search.last['candidates'])
    assert search.spent<900


def test_incomplete_branch_not_used_as_terminal_target():
    c=game_class('toy_battle');e=c();e.reset(17221)
    s=TerminalReference(c,{'kind':'heuristic'},seat=0,seconds=900,max_rounds=1,max_simulation_steps=0)
    s.decide(e)
    assert s.last['fallback'] and s.last['completed_rounds']==0
    assert all(x['mean'] is None and x['samples']==0 for x in s.last['candidates'])


def test_root_sample_independent_of_live_rng():
    import random
    c=game_class('splendor');a=c();a.reset(72);b=c();b.restore(a.snapshot());b.rng=random.Random(998)
    for d in b.state['decks']:d.reverse()
    assert sampled_world(a,0,91).snapshot()==sampled_world(b,0,91).snapshot()

@pytest.mark.parametrize('game',['machi_koro','el_dorado'])
def test_scoreless_races_use_real_victory_without_fabricating_score(game):
    c=game_class(game);e=c();e.reset(17221)
    s=TerminalReference(c,{'kind':'heuristic'},seat=0,seconds=900,max_rounds=1,width=2)
    result,steps,counts=s._round(sampled_world(e,0,91),e.legal_actions()[:2],0,s.clock()+15)
    assert result is not None
    values,scores=result
    assert all(v in (0.,.5,1.) for v in values)
    assert all(v==[None,None] for v in scores)


def test_report_retains_missing_and_incomplete_schedule(tmp_path):
    import json
    from boardbench.families.search_experiment import report
    jobs=[dict(game=g,budget=b) for g in ('calico','machi_koro') for b in (30,900) for _ in range(8)]
    (tmp_path/'schedule.json').write_text(json.dumps(dict(jobs=jobs)))
    rows=report(tmp_path)
    assert len(rows)==4
    assert all(r['scheduled']==8 and r['complete']==0 and r['statuses']['no_result']==8 for r in rows)
    assert all(r['rl_score'] is None and r['search_score'] is None for r in rows)


def test_rl_continuation_does_not_call_heuristic_inside_simulation():
    c=game_class('toy_battle');e=c();e.reset(17221)
    s=TerminalReference(c,{'kind':'heuristic'},seat=0,seconds=900,max_rounds=1,width=2,continuation='rl')
    def forbidden(*args):raise AssertionError('Heuristic continuation used in RL-only ablation')
    s.expert.decide=forbidden;s.progress.decide=forbidden
    result,steps,_=s._round(sampled_world(e,0,99),e.legal_actions()[:2],0,s.clock()+10)
    assert result is not None and steps>2
