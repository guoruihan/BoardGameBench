"""Original 45 tribe and 19 goods cards, independently transcribed from
published original components (Yucata card IDs 0..44; provenance in sources).
Costs and gains use dates, salt, pepper, gold. Effects are executable identifiers,
not reproduced card prose. No artwork or third-party implementation is included.
"""
RESOURCE_NAMES = ('dates', 'salt', 'pepper', 'gold')
# type, VP, alternative payment vectors, effect
_ROWS = [
 ('oasis',2,[(0,0,1,1)],''),
 ('oasis',3,[(1,1,0,1)],''),
 ('oasis',3,[(2,1,2,0)],''),
 ('camel',3,[(1,1,0,1)],''),
 ('camel',2,[(1,2,1,0)],''),
 ('camel',3,[(1,2,2,0)],''),
 ('well',3,[(1,1,0,1)],''),
 ('well',2,[(2,1,1,0)],''),
 ('well',3,[(2,1,2,0)],''),
 ('camp',2,[(0,0,1,1)],''),
 ('camp',3,[(1,0,1,1)],''),
 ('camp',3,[(2,2,1,0)],''),
 ('targia',3,[(1,0,1,1)],''),
 ('targia',2,[(1,1,2,0)],''),
 ('targia',3,[(1,2,2,0)],''),
 ('oasis',1,[(0,0,1,1)],'pairs_well'),
 ('camel',1,[(0,1,0,1)],'pairs_camp'),
 ('well',1,[(0,1,0,1)],'pairs_targia'),
 ('camp',1,[(1,0,0,1)],'pairs_oasis'),
 ('targia',1,[(0,0,1,1)],'pairs_camel'),
 ('camel',1,[(0,0,0,1)],'majority'),
 ('camp',1,[(1,0,1,0)],'ten_cards'),
 ('targia',1,[(0,0,2,1)],'diverse_rows'),
 ('camel',2,[(2,2,1,0)],'rightmost'),
 ('camel',1,[(0,0,0,1)],'discount_camel'),
 ('well',1,[(1,2,0,0)],'discount_well'),
 ('camp',1,[(1,0,2,0),(0,0,0,1)],'discount_camp'),
 ('targia',1,[(1,2,0,0),(0,0,0,1)],'discount_targia'),
 ('oasis',1,[(0,2,1,0)],'discount_oasis'),
 ('oasis',1,[(0,1,1,1)],'raid_immunity'),
 ('camp',2,[(1,0,0,1)],'noble_discount'),
 ('targia',1,[(1,0,0,1)],'raid_point'),
 ('well',1,[(1,0,0,1)],'robber_placement'),
 ('oasis',1,[(0,2,1,0)],'bonus_salt'),
 ('camel',2,[(0,0,2,1)],'bonus_pepper'),
 ('targia',2,[(2,0,0,1)],'bonus_dates'),
 ('oasis',2,[(1,0,1,1)],'no_goods'),
 ('well',2,[(1,2,1,0)],'display_goods'),
 ('well',2,[(0,1,1,1)],'hand_surcharge'),
 ('camp',1,[(1,0,0,1)],'trader_discount'),
 ('well',1,[(1,0,0,1)],'bank_gold'),
 ('targia',2,[(1,1,2,0)],'choice_bonus'),
 ('camel',1,[(2,1,0,0)],'empty_goods'),
 ('oasis',2,[(0,1,1,1)],'rearrange'),
 ('camp',2,[(1,1,2,0)],'two_pawns'),
]
TRIBES = {i:dict(kind=t,vp=v,costs=c,effect=e) for i,(t,v,c,e) in enumerate(_ROWS)}
# Unique physical IDs; missing sprite IDs refer to copies of the preceding face.
GOODS = {1:(1,0,0,0,0),2:(1,0,0,0,0),3:(0,1,0,0,0),4:(0,1,0,0,0),
         5:(0,0,1,0,0),6:(0,0,1,0,0),7:(2,0,0,0,0),8:(0,2,0,0,0),
         9:(0,0,2,0,0),10:(0,1,1,0,0),11:(1,0,1,0,0),12:(1,1,0,0,0),
         13:None,14:None,15:None,16:(0,0,0,1,0),17:(0,0,0,1,0),
         18:(0,0,0,1,0),19:(0,0,0,0,1)}
BORDER = {1:'noble',2:'dates',3:'salt',4:'raid',5:'trader',6:'pepper',7:'dates',
          8:'raid',9:'fata_morgana',10:'silversmith',11:'pepper',12:'raid',
          13:'caravan',14:'tribal_expansion',15:'salt',16:'raid'}
# row/column in 5x5 board, clockwise from the first top non-corner card.
COORDS = {1:(0,1),2:(0,2),3:(0,3),4:(0,4),5:(1,4),6:(2,4),7:(3,4),
          8:(4,4),9:(4,3),10:(4,2),11:(4,1),12:(4,0),13:(3,0),14:(2,0),
          15:(1,0),16:(0,0)}
OPPOSITE = {1:11,2:10,3:9,5:15,6:14,7:13,9:3,10:2,11:1,13:7,14:6,15:5}
