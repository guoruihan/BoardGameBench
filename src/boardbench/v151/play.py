"""Public-only CLI for explicit assistant actions and a frozen RL opponent."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from boardbench.environments.harmonies import Harmonies, STACKS, PATTERNS
from boardbench.environments.harmonies_duo import HarmoniesDuo

COLORS = dict(m='gray', u='blue', w='brown', g='green', y='yellow', r='red')
SHORT = {v:k for k,v in COLORS.items()}
MODEL = None
MODEL_KEY = None


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, data):
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def solver(config):
    global MODEL, MODEL_KEY
    import torch
    from boardbench.v145.policy import SelfplaySolver
    torch.set_num_threads(1)
    path = Path(config['model_path'])
    key = (str(path.resolve()), config['model_sha256'])
    if MODEL is None or MODEL_KEY != key:
        if hashlib.sha256(path.read_bytes()).hexdigest() != config['model_sha256']:
            raise ValueError('opponent SHA256 mismatch')
        s = SelfplaySolver(device='cpu'); s.load_weights(path); MODEL = s.model
        MODEL_KEY = key
    return SelfplaySolver(model=MODEL, device='cpu')


class Session:
    def __init__(self, path):
        self.path = Path(path); self.d = json.loads(self.path.read_text())
        self.env = Harmonies() if self.d['mode'] == 'solo' else HarmoniesDuo()
        self.env.load_state(self.d['snapshot'])
        self.ai = None
        if self.d['mode'] == 'duo':
            self.ai = solver(self.d['opponent'])
            for p, state in zip(self.ai.planners, self.d['planners']):
                state = deepcopy(state)
                for goal in state['active'] or []:
                    goal['targets'] = {int(k):v for k,v in goal['targets'].items()}
                p.__dict__.update(state)

    @classmethod
    def create(cls, path, mode, seed, seat, opponent):
        path = Path(path)
        if path.exists(): raise FileExistsError(path)
        env = Harmonies() if mode == 'solo' else HarmoniesDuo()
        env.reset(seed)
        d = dict(mode=mode, seed=seed, seat=seat, opponent=opponent if mode=='duo' else None,
                 created_unix=time.time(), snapshot=env.save_state(), records=[], submissions=[],
                 planners=[dict(horizon=2, active=None, started=0) for _ in range(2)])
        write(path,d); obj=cls(path); obj.advance();obj.save();return obj

    def save(self):
        self.d['snapshot']=self.env.save_state()
        if self.ai:self.d['planners']=[deepcopy(p.__dict__) for p in self.ai.planners]
        self.d['updated_unix']=time.time();write(self.path,self.d)

    def step(self, action, actor, note=''):
        obs=self.env.observe()
        if actor not in ('assistant', 'rl_opponent'):
            raise ValueError('unknown actor')
        if action not in obs['legal_actions']:raise ValueError(f'illegal action: {action}')
        if self.d['mode']=='solo' and actor!='assistant':
            raise ValueError('solo has no opponent')
        if self.d['mode']=='duo' and (actor=='assistant')!=(obs['active_player']==self.d['seat']):
            raise ValueError('not assistant turn' if actor=='assistant' else 'not opponent turn')
        self.env.step(action)
        self.d['records'].append(dict(actor=actor, action=action, observation=obs,
                                      observation_sha256=digest(obs), note=note, time=time.time()))

    def advance(self):
        if self.ai is None:return
        for _ in range(768):
            obs=self.env.observe()
            if obs['terminated'] or obs['active_player']==self.d['seat']:return
            self.step(self.ai.decide(obs),'rl_opponent')
        raise RuntimeError('opponent action limit')

    def action(self, text):
        if text.startswith('a'):
            card, cell=map(int,text[1:].split('@'))
            options=[a for a in self.env.legal_actions() if a['type']=='place_animal'
                     and a['instance_id']==card and a['anchor_cell_id']==cell]
            if not options:raise ValueError('animal placement does not match')
            return options[0]  # Equivalent orientation witness only, never choose a location.
        if text=='e':return dict(type='end_turn')
        if text[0]=='d':return dict(type='end_turn',discard_card_slot=int(text[1:]))
        if text[0]=='b':return dict(type='choose_bundle',bundle_id=int(text[1:]))
        if text[0]=='c':return dict(type='take_card',market_slot=int(text[1:]))
        if text[0] in COLORS:return dict(type='place_token',color=COLORS[text[0]],cell_id=int(text[1:]))
        raise ValueError('unknown command')

    def submit(self, commands, note, expected):
        if expected!=self.env.s['decision_index']:raise ValueError('stale observation revision')
        if not note.strip():raise ValueError('decision note required')
        if not isinstance(commands,list) or not commands or any(not isinstance(c,str) or not c for c in commands):
            raise ValueError('nonempty command list required')
        # Reject malformed batches before any mutation, including duplicate end commands.
        if any(c[0] in ('e','d') for c in commands[:-1]):
            raise ValueError('end_turn must be the final command')
        entry=dict(revision=expected,commands=commands,note=note,time=time.time())
        self.d['submissions'].append(entry)
        try:
            for command in commands:
                self.step(self.action(command),'assistant',note)
            self.advance()
            entry['status']='accepted'
        except Exception as e:
            entry['status']='partial_error';entry['error']=str(e)
            raise
        finally:self.save()

    def view(self):
        obs=self.env.observe();seat=self.d['seat'];duo=self.d['mode']=='duo'
        own=obs['players'][seat] if duo else obs
        def board(person):
            return ' '.join(f'{i}:{"".join(SHORT[c] for c in STACKS[b["stack_id"]]) or "."}'
                            +(f'*{b["animal"]}' if b['animal'] is not None else '') for i,b in enumerate(person['board']))
        score=obs['score_breakdown'][seat] if duo else obs['score_breakdown']
        out=dict(game=self.path.stem,rev=obs['decision_index'],turn=own['turn_index'],ended=obs['terminated'],
                 score=score,board=board(own),active={c['card_id']:c['placed_count'] for c in own['active_cards']},
                 completed={c['card_id']:c['placed_count'] for c in own['completed_cards']},
                 bundles={i:''.join(SHORT[c] for c in b) for i,b in enumerate(obs['token_market']) if b},
                 market={i:c for i,c in enumerate(obs['animal_market']) if c is not None},
                 pending={SHORT[c]:n for c,n in own['pending_tokens'].items() if n},
                 took=[own['bundle_taken'],own['card_taken']],remaining_tokens=obs['remaining_tokens'],
                 remaining_cards=obs['remaining_cards'])
        available=[c['card_id'] for c in own['active_cards']]+[c for c in obs['animal_market'] if c]
        ready={}
        for card in available:
            anchors=[i for i in range(23) if own['board'][i]['animal'] is None and any(
                all(own['board'][j]['stack_id'] in allowed for j,allowed in pattern)
                for pattern in PATTERNS[card][i].values())]
            if anchors:ready[card]=anchors
        out['matching_anchors']=ready
        if duo:
            other=obs['players'][1-seat]
            out.update(seat=seat,opponent_score=obs['score_breakdown'][1-seat],opponent_board=board(other),
                       opponent_cards={c['card_id']:c['placed_count'] for c in other['active_cards']},
                       final_round=obs['final_round'],winners=obs['winners'])
        return out

    def audit(self):
        env=Harmonies() if self.d['mode']=='solo' else HarmoniesDuo();env.reset(self.d['seed'])
        opponent=solver(self.d['opponent']) if self.d['mode']=='duo' else None
        for row in self.d['records']:
            assert env.observe()==row['observation']
            assert digest(row['observation'])==row['observation_sha256']
            assert row['actor'] in ('assistant','rl_opponent')
            if self.d['mode']=='duo':
                assert (row['actor']=='assistant')==(env.s['active_player']==self.d['seat'])
            else:
                assert row['actor']=='assistant'
            if row['actor']=='rl_opponent':
                assert opponent.decide(row['observation'])==row['action'], 'frozen opponent decision mismatch'
            assert row['action'] in env.legal_actions();env.step(row['action'])
        assert env.save_state()==self.env.save_state()
        obs=env.observe();duo=self.d['mode']=='duo';seat=self.d['seat']
        own=obs['players'][seat] if duo else obs
        other=obs['players'][1-seat] if duo else None
        return dict(actions=len(self.d['records']),assistant_actions=sum(r['actor']=='assistant' for r in self.d['records']),
                    assistant_turns=sum(r['actor']=='assistant' and r['action']['type']=='end_turn' for r in self.d['records']),
                    animal_cubes=sum(c['animal'] is not None for c in own['board']),
                    opponent_animal_cubes=sum(c['animal'] is not None for c in other['board']) if duo else None,
                    completed_cards=len(own['completed_cards']),
                    rejected_submissions=sum(e.get('status')=='partial_error' for e in self.d['submissions']),
                    frozen_opponent_verified=duo,natural_terminal=env.s['terminated'],
                    session_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),final=self.view())


def summarize(manifest, reports):
    """Aggregate all predeclared games, using the engine's animal-cube tie break."""
    rows=[]
    for game in manifest['games']:
        report=reports[game['id']];final=report['final'];score=final['score']
        if not report['natural_terminal']:
            raise ValueError('summary requires every declared game to finish')
        row=dict(game=game['id'],mode=game['mode'],seed=game['seed'],seat=game['seat'],
                 total=score['total'],animals=score['animals'],terrain=score['total']-score['animals'],
                 breakdown=score,turns=report['assistant_turns'],animal_cubes=report['animal_cubes'],
                 completed_cards=report['completed_cards'],rejected_submissions=report['rejected_submissions'],
                 session_sha256=report['session_sha256'])
        if game['mode']=='duo':
            opponent=final['opponent_score'];winners=final['winners']
            row.update(opponent_total=opponent['total'],opponent_animals=opponent['animals'],
                       opponent_terrain=opponent['total']-opponent['animals'],
                       opponent_animal_cubes=report['opponent_animal_cubes'],
                       outcome='draw' if len(winners)==2 else 'win' if game['seat'] in winners else 'loss')
        rows.append(row)
    aggregates={}
    for mode in ('solo','duo'):
        group=[r for r in rows if r['mode']==mode]
        if not group:continue
        stats=dict(games=len(group),mean={key:sum(r[key] for r in group)/len(group)
                   for key in ('total','animals','terrain','turns','animal_cubes')},
                   min_score=min(r['total'] for r in group),max_score=max(r['total'] for r in group))
        if mode=='duo':
            stats['outcomes']={key:sum(r['outcome']==key for r in group) for key in ('win','draw','loss')}
            stats['opponent_mean']={key:sum(r['opponent_'+key] for r in group)/len(group)
                                    for key in ('total','animals','terrain','animal_cubes')}
        aggregates[mode]=stats
    return dict(schema='boardbench.v151.assistant_play.v1',games=rows,aggregates=aggregates,
                total_actions=sum(r['actions'] for r in reports.values()),
                assistant_actions=sum(r['assistant_actions'] for r in reports.values()),
                rejected_submissions=sum(r['rejected_submissions'] for r in reports.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--init',action='store_true')
    p.add_argument('--requests');p.add_argument('--view',nargs='*');p.add_argument('--audit',action='store_true');a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    if a.init:
        config=json.loads(Path('configs/harmonies_duo_play.json').read_text())
        manifest=dict(version=151,controller='current conversational assistant',assistance='public rules, matching, legality only',
                      no_search=True,seed_scheme='fresh manual-play 151100000+i solo; 151200000+i duo',opponent=config,
                      games=[dict(id=f'{mode}{i}',mode=mode,seed=(151100000 if mode=='solo' else 151200000)+i,
                                  seat=0 if mode=='solo' else i%2) for mode in ['solo','duo'] for i in range(8)])
        if (root/'manifest.json').exists():raise FileExistsError('experiment already initialized')
        write(root/'manifest.json',manifest)
        for g in manifest['games']:Session.create(root/(g['id']+'.json'),g['mode'],g['seed'],g['seat'],config)
        print('Created16 fixed games; no assistant decisions made.')
    if a.requests:
        for request in json.loads(Path(a.requests).read_text()):
            s=Session(root/(request['game']+'.json'))
            try:s.submit(request['commands'],request['note'],request['rev'])
            except Exception as e:print(json.dumps(dict(game=request['game'],error=str(e)),ensure_ascii=False))
            print(json.dumps(s.view(),ensure_ascii=False))
    if a.view is not None:
        for name in a.view:print(json.dumps(Session(root/(name+'.json')).view(),ensure_ascii=False))
    if a.audit:
        m=json.loads((root/'manifest.json').read_text());reports={}
        for g in m['games']:
            session=Session(root/(g['id']+'.json'))
            if any(session.d[k]!=g[k] for k in ('mode','seed','seat')):
                raise ValueError('session does not match predeclared manifest')
            if g['mode']=='duo' and session.d['opponent']!=m['opponent']:
                raise ValueError('session opponent does not match manifest')
            reports[g['id']]=session.audit()
        write(root/'audit.json',reports)
        print(json.dumps({k:dict(actions=v['actions'],ended=v['natural_terminal'],score=v['final']['score']['total']) for k,v in reports.items()}))
        if all(r['natural_terminal'] for r in reports.values()):
            summary=summarize(m,reports);write(root/'summary.json',summary)
            print(json.dumps(summary['aggregates']))

if __name__=='__main__':main()
