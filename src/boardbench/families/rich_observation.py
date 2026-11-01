"""Versioned, typed player-observation and action encoding for all families.

The old 24 heuristic features remain optional advice. They are NOT the state.
Every field supplied by observe(), including visible history, is retained in a
typed tree. Paths retain dictionary keys AND array indices; strings use an exact
static vocabulary or UTF-8 bytes, never hashing or an unknown-token bucket.
No engine, snapshot, RNG or environment seed is accepted by this interface.
"""
from functools import lru_cache
import ast
import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np

from .registry import game_class

VERSION = 'family_public_tree_v1'
DEPTH = 16
NULL, BOOL, NUMBER, SYMBOL, TEXT, BYTE, LIST, DICT = range(8)
FLOAT = 8
# Each row: 16 dictionary-key IDs, 16 list indices (index+1), type, symbol,
# numeric value. Non-float32 numbers use an exact textual representation.
WIDTH = DEPTH*2+3
STATIC_FIELDS = {
    'cascadia': ('scoring_cards',),
    'calico': ('coordinates','playable_indices','goal_indices','board_ids','cat_rules','goal_rules'),
    'patchwork': ('patches','income_events','end'),
    'harmonies': (), 'century_spice_road': (), 'jaipur': (), 'splendor': (),
    'gizmos': (), 'machi_koro': (), 'clever': (), 'century_a_new_world': (),
    'stone_age': (), 'targi': (), 'dominion': (), 'el_dorado': ('board',),
    'toy_battle': ('map','special_base_effects'),
}


def _strings(value, found):
    if isinstance(value, str):
        if len(value) <= 160:
            found.add(value)
    elif isinstance(value, dict):
        for k,v in value.items():
            _strings(k,found);_strings(v,found)
    elif isinstance(value, (list,tuple,set)):
        for v in value:_strings(v,found)


@lru_cache(maxsize=16)
def vocabulary(game):
    """Static source/component symbols only; independent of sampled games."""
    cls = game_class(game)
    module = importlib.import_module(cls.__module__)
    found = {'game_id','config_id','rules_version','player','current_player',
             'terminated','decision_index','state','action','catalog',
             'CARDS','TILES','MERCHANTS','ORDERS','NOBLES','LANDMARKS','TRIBES',
             'GOODS','BUILDINGS','BONUSES','EXPLORATIONS','LOCATIONS','DATA'}
    paths = [Path(module.__file__), Path(__file__).with_name('base.py')]
    if game == 'harmonies':
        from boardbench.environments import harmonies, harmonies_duo
        for m in (harmonies,harmonies_duo):
            paths.append(Path(m.__file__))
            for name,v in vars(m).items():
                if not name.startswith('__'):_strings(v,found)
    for name,v in vars(module).items():
        if not name.startswith('__'):_strings(v,found)
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Constant):_strings(node.value,found)
            elif isinstance(node,ast.keyword) and node.arg:found.add(node.arg)
    # Reserve byte symbols 1..256 for arbitrary text. IDs start at257.
    symbols = sorted(found)
    result = {v:i+257 for i,v in enumerate(symbols)}
    identity = hashlib.sha256(json.dumps(symbols,ensure_ascii=False).encode()).hexdigest()
    return result,identity


class PublicTreeEncoder:
    def __init__(self, game, *, cache=True, symbols=None):
        if game not in STATIC_FIELDS:
            raise ValueError('Game not qualified for the rich observation interface')
        self.game = game
        if symbols is None:
            self.symbols,self.vocabulary_sha256 = vocabulary(game)
        else:
            if not isinstance(symbols,list) or not all(isinstance(x,str) for x in symbols) or sorted(set(symbols))!=symbols:
                raise ValueError('Expected a sorted, unique frozen symbol list')
            self.symbols={v:i+257 for i,v in enumerate(symbols)}
            self.vocabulary_sha256=hashlib.sha256(json.dumps(symbols,ensure_ascii=False).encode()).hexdigest()
        self.cache = cache
        self._paths = {}
        self._static = {}
        self._catalog_cache = {}
        self._action_cache = {}

    def _catalog(self,observation):
        """Game-specific public component facts, not hidden component locations."""
        module=importlib.import_module(game_class(self.game).__module__)
        if self.game=='harmonies':
            from boardbench.environments import harmonies as module
            ids={c['card_id'] for p in observation['players']
                 for c in p['active_cards']+p['completed_cards']}
            ids.update(c for c in observation['animal_market'] if c is not None)
            return {'CARDS':{c:module.CARDS[c] for c in sorted(ids)}}
        if self.game=='cascadia':
            ids={c['tile'] for b in observation['boards'] for c in b}
            ids.update(c for c in observation['market_tiles'] if c is not None)
            if observation['pending_tile'] is not None:ids.add(observation['pending_tile'])
            return {'TILES':{c:module.TILES[c] for c in sorted(ids)}}
        if self.game in ('calico','patchwork','toy_battle'):
            return {} # Observation already includes the complete fixed rules tables.
        if self.game=='stone_age':
            cards={c for c in observation['market'] if c is not None}
            cards.update(c for p in observation['players'] for c in (p['cards'] or []))
            cards.update(e['card'] for e in observation['public_history'] if e['type']=='card_acquired')
            buildings={c for c in observation['buildings'] if c is not None}
            buildings.update(c for p in observation['players'] for c in p['buildings'])
            return {'CARDS':{c:module.CARDS[c] for c in sorted(cards)},
                    'BUILDINGS':{c:module.BUILDINGS[c] for c in sorted(buildings)}}
        if self.game=='clever':return {'DATA':module.DATA}
        # These fixed catalogs are small, static public rules. Include each
        # completely so even numeric IDs and future legal market types have
        # their actual costs/effects, not only arbitrary identity codes.
        names={
            'century_spice_road':('MERCHANTS','ORDERS'), 'jaipur':('DATA',),
            'splendor':('CARDS','NOBLES'), 'gizmos':('CARDS',),
            'machi_koro':('CARDS','LANDMARKS'),
            'century_a_new_world':('CARDS','BONUSES','EXPLORATIONS','LOCATIONS'),
            'targi':('TRIBES','GOODS'), 'dominion':('CARDS',), 'el_dorado':('CARDS',),
        }[self.game]
        return {k:getattr(module,k) for k in names}

    @property
    def metadata(self):
        cls=game_class(self.game)
        return dict(version=VERSION,game=self.game,config=cls.config_id,
                    rules_version=cls.rules_version,path_depth=DEPTH,
                    vocabulary_sha256=self.vocabulary_sha256,
                    vocabulary_size=len(self.symbols)+257,
                    static_fields=list(STATIC_FIELDS[self.game]))

    def _path(self, path):
        if len(path)>DEPTH:
            raise ValueError('Public tree exceeds versioned path depth; refusing truncation')
        if self.cache and path in self._paths:return self._paths[path]
        row=np.zeros(WIDTH,np.float32)
        for i,(kind,value) in enumerate(path):
            if kind=='key':
                if value not in self.symbols:
                    raise ValueError(f'Unregistered public dictionary key: {value!r}')
                row[i]=self.symbols[value]
            else:
                if not 0<=value<2**24-1:raise ValueError('Array index outside exact float32 range')
                row[DEPTH+i]=value+1
        if self.cache:
            # Bounded per-encoder cache; no accumulation of complete histories.
            if len(self._paths)>=32768:self._paths.clear()
            self._paths[path]=row
        return row

    def _walk(self,value,path,rows):
        row=self._path(path).copy()
        if value is None:row[-3]=NULL
        elif isinstance(value,bool):row[-3]=BOOL;row[-1]=int(value)
        elif isinstance(value,(int,float)):
            if not math.isfinite(value):raise ValueError('Nonfinite public number')
            if abs(value)<1e30 and float(np.float32(value))==value:
                row[-3]=NUMBER if isinstance(value,int) else FLOAT;row[-1]=value
            else:
                # Preserve large bitboards or non-float32 quantities exactly.
                row[-3]=TEXT;row[-1]=-1 if isinstance(value,int) else -2
                rows.append(row)
                for i,b in enumerate(repr(value).encode()):
                    byte=self._path(path+(('index',i),)).copy()
                    byte[-3]=BYTE;byte[-2]=b+1;rows.append(byte)
                return
        elif isinstance(value,str):
            if value in self.symbols:row[-3]=SYMBOL;row[-2]=self.symbols[value]
            else:
                raw=value.encode('utf-8');row[-3]=TEXT;row[-1]=len(raw);rows.append(row)
                for i,b in enumerate(raw):
                    byte=self._path(path+(('index',i),)).copy()
                    byte[-3]=BYTE;byte[-2]=b+1;rows.append(byte)
                return
        elif isinstance(value,dict):
            row[-3]=DICT;row[-1]=len(value);rows.append(row)
            for key in sorted(value,key=lambda k:(type(k).__name__,str(k))):
                if type(key) is int:
                    # Integer dictionary keys differ from list indices.
                    self._walk(value[key],path+(('key','state'),('index',key)),rows)
                elif isinstance(key,str):self._walk(value[key],path+(('key',key),),rows)
                else:raise ValueError('Unsupported public dictionary key type')
            return
        elif isinstance(value,(list,tuple)):
            row[-3]=LIST;row[-1]=len(value);rows.append(row)
            for i,v in enumerate(value):self._walk(v,path+(('index',i),),rows)
            return
        else:raise ValueError(f'Unsupported public value {type(value)}')
        rows.append(row)

    def encode(self,observation):
        cls=game_class(self.game)
        if any(observation.get(k)!=v for k,v in
               dict(game_id=self.game,config_id=cls.config_id,rules_version=cls.rules_version).items()):
            raise ValueError('Player observation game/schema mismatch')
        if any(k in observation for k in ('rng','snapshot','private','environment_seed')):
            raise ValueError('Privileged provenance is not a player observation')
        # Legal actions have their own lossless candidate stream. All other
        # observation fields are retained, including visible history.
        items={k:v for k,v in observation.items() if k!='legal_actions'}
        parts=[]
        for key in sorted(items):
            value=items[key];path=(('key',key),)
            if self.cache:
                packed=json.dumps(value,sort_keys=True,separators=(',',':'))
                entry=self._static.get(key)
                if entry is None or entry[0]!=packed:
                    # Public histories grow by appending events. Reuse only an
                    # exactly matching serialized prefix; reset/mutation falls
                    # through to the reference traversal. Never trust identity
                    # of mutable engine objects or a sampled environment seed.
                    if (entry is not None and key in ('history','public_history') and isinstance(value,list)
                            and entry[2] is not None and len(value)>entry[2]
                            and (entry[2]==0 or packed.startswith(entry[0][:-1]+','))):
                        part=[]
                        for i in range(entry[2],len(value)):
                            self._walk(value[i],path+(('index',i),),part)
                        header=entry[1][:1].copy();header[0,-1]=len(value)
                        array=np.concatenate((header,entry[1][1:],np.asarray(part,np.float32).reshape(-1,WIDTH)))
                    else:
                        part=[];self._walk(value,path,part);array=np.asarray(part,np.float32)
                    entry=(packed,array,len(value) if isinstance(value,list) else None);self._static[key]=entry
                parts.append(entry[1])
            else:
                rows=[];self._walk(value,path,rows)
                parts.append(np.asarray(rows,np.float32).reshape(-1,WIDTH))
        catalog=self._catalog(observation)
        identity=json.dumps(catalog,sort_keys=True,separators=(',',':'))
        part=self._catalog_cache.get(identity) if self.cache else None
        if part is None:
            rows=[];self._walk(catalog,(('key','catalog'),),rows)
            part=np.asarray(rows,np.float32).reshape(-1,WIDTH)
            if self.cache:
                if len(self._catalog_cache)>=32:self._catalog_cache.clear()
                self._catalog_cache[identity]=part
        parts.append(part)
        return np.concatenate(parts)

    def encode_action(self,action):
        packed=json.dumps(action,sort_keys=True,separators=(',',':')) if self.cache else None
        if packed in self._action_cache:return self._action_cache[packed].copy()
        rows=[];self._walk(action,(('key','action'),),rows)
        result=np.asarray(rows,np.float32).reshape(-1,WIDTH)
        if self.cache:
            if len(self._action_cache)>=4096:self._action_cache.clear()
            self._action_cache[packed]=result.copy()
        return result

    def encode_actions(self,actions):
        chunks=[self.encode_action(a) for a in actions]
        owners=np.repeat(np.arange(len(chunks),dtype=np.int64),[len(a) for a in chunks])
        return (np.concatenate(chunks) if chunks else np.zeros((0,WIDTH),np.float32),owners)


def numeric_inputs(rows):
    """Separate categorical paths from normalized numeric values, without clipping."""
    rows=np.asarray(rows,np.float32)
    if rows.ndim!=2 or rows.shape[1]!=WIDTH or not np.isfinite(rows).all():
        raise ValueError('Malformed typed public rows')
    keys=rows[:,:DEPTH].astype(np.int64)
    symbols=rows[:,-2].astype(np.int64)
    numbers=np.concatenate((rows[:,DEPTH:2*DEPTH],rows[:,-1:]),axis=1)
    numbers=np.sign(numbers)*np.log1p(np.abs(numbers))/5
    return keys,rows[:,-3].astype(np.int64),symbols,numbers
