// Pure public-board kernel. Original Python transitions and RNG are unchanged.
#include <algorithm>
#include <cstdint>
#include "tables.h"

static uint32_t neighbors(uint32_t bits) {
    uint32_t result=0;
    while(bits) { int i=__builtin_ctz(bits); bits &= bits-1; result |= adjacency[i]; }
    return result;
}

static void terrain(const int16_t* ids, int16_t* out) {
    uint32_t fields=0, water=0, mountains=0;
    for(int i=0;i<6;++i) out[i]=0;
    for(int i=0;i<23;++i) {
        int s=ids[i];
        if(s==8) out[0]+=1; else if(s==9) out[0]+=3; else if(s==10) out[0]+=7;
        if(s==2) fields |= 1u<<i;
        else if(s==1) water |= 1u<<i;
        else if(s>=3 && s<=5) mountains |= 1u<<i;
    }
    for(int i=0;i<23;++i) {
        int s=ids[i];
        if(s>=3 && s<=5 && (adjacency[i]&mountains)) out[1]+=(1<<(s-2))-1;
        else if(s>=12) {
            uint32_t ns=adjacency[i]; int colors=0;
            while(ns) { int j=__builtin_ctz(ns); ns &= ns-1; colors |= tops[ids[j]]; }
            if(__builtin_popcount(static_cast<unsigned>(colors))>=3) out[3]+=5;
        }
    }
    while(fields) {
        uint32_t component=fields & -fields;
        while(true) {
            uint32_t expanded=component | (neighbors(component)&fields);
            if(expanded==component) break;
            component=expanded;
        }
        if(__builtin_popcount(component)>=2) out[2]+=5;
        fields &= ~component;
    }
    uint32_t starts=water;
    while(starts) {
        uint32_t frontier=starts & -starts; starts ^= frontier;
        uint32_t visited=frontier; int length=0;
        while(frontier) { ++length; frontier=neighbors(frontier)&water&~visited; visited |= frontier; }
        out[5]=std::max<int>(out[5],length);
    }
    const int river[]={0,0,2,5,8,11,15};
    out[4]=out[5]<=6 ? river[out[5]] : 15+4*(out[5]-6);
}

static double landscape(const int16_t* ids, int animals) {
    int16_t score[6]; terrain(ids,score);
    int total=animals; for(int i=0;i<5;++i) total+=score[i];
    const double bonus[]={0.,0.,0.,.25,.5,0.,.7,1.8,0.,0.,0.,1.,0.,0.,0.};
    double sum=0.; for(int i=0;i<23;++i) sum+=bonus[ids[i]];
    return total+sum; // Preserve Python's order of double additions.
}

static double potential(const int16_t* ids, uint32_t occupied, int card) {
    double best=0.;
    for(int i=offsets[card-1];i<offsets[card];++i) {
        const auto &p=patterns[i];
        if(occupied & (1u<<p.anchor)) continue;
        int matched=0; bool valid=true;
        for(int j=p.begin;j<p.begin+p.size;++j) {
            const auto &r=requirements[j]; unsigned bit=1u<<ids[r.cell];
            if(r.allowed&bit) ++matched;
            else if((occupied&(1u<<r.cell)) || !(r.reachable&bit)) {valid=false;break;}
        }
        if(valid) best=std::max(best,static_cast<double>(matched)/p.size);
    }
    return best;
}

extern "C" int harmonies_terrain(const int16_t* ids, int16_t* out) {
    if(!ids || !out) return 1;
    for(int i=0;i<23;++i) if(ids[i]<0 || ids[i]>=15) return 2;
    terrain(ids,out); return 0;
}

extern "C" int harmonies_values(const int16_t* ids, uint32_t occupied, int animals,
                                int n, const int16_t* actions, double* out) {
    if(!ids || !actions || !out || n<0 || n>256 || occupied>=(1u<<23) || animals<0 || animals>10000) return 1;
    for(int i=0;i<23;++i) if(ids[i]<0 || ids[i]>=15) return 2;
    bool tokens=false;
    for(int i=0;i<n;++i) {
        int kind=actions[3*i], a=actions[3*i+1], b=actions[3*i+2];
        if(kind<0 || kind>4) return 3;
        if(kind==1 && (a<1 || a>32)) return 3;
        if(kind==2) {if(a<0 || a>=23 || b<0 || b>=15) return 3; tokens=true;}
    }
    double baseline=tokens ? landscape(ids,animals) : 0.;
    int16_t copy[23];std::copy(ids,ids+23,copy);
    for(int i=0;i<n;++i) {
        int kind=actions[3*i], a=actions[3*i+1], b=actions[3*i+2];out[i]=0.;
        if(kind==1) out[i]=.2+potential(ids,occupied,a);
        else if(kind==2) {copy[a]=b;out[i]=landscape(copy,animals)-baseline;copy[a]=ids[a];}
    }
    return 0;
}
