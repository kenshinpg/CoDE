"""Label-free, probe-free single-group interventions with exact-count controls."""
import hashlib,random,math
def stable(seed,key):return hashlib.sha256(f'{seed}|{key}'.encode()).hexdigest()
def geometry(facts,groups,cited,operation):
    F=set(facts);E=set(cited)
    if not E<=F:raise ValueError('Nonpatient citation')
    flat=[f for fs in groups.values() for f in fs]
    if len(flat)!=len(set(flat)) or set(flat)!=F:raise ValueError('Groups must partition observed facts')
    if operation not in ('add','delete'):raise ValueError('Unknown operation')
    out={}
    for g,fs in sorted(groups.items()):
        G=set(fs);change=(G-E) if operation=='add' else (G&E)
        if not change:continue
        after=(E|G) if operation=='add' else (E-G)
        assert len(E^after)==len(change) and after<=F
        out[g]={'fact_ids':sorted(after),'changed_fact_ids':sorted(change),'actual_fact_count':len(change),'partial_group':0<len(E&G)<len(G)}
    return out
def select(candidates,weights,direction,seed,repeats=10,epsilon=1e-10):
    if weights is None:return {'available':False,'reason':'selector_unavailable'}
    if direction not in ('support','counter','unsigned'):raise ValueError('Unknown direction')
    if any(not math.isfinite(float(weights[g])) for g in candidates):raise ValueError('Nonfinite weight')
    mag={g:(-float(weights[g]) if direction=='counter' else float(weights[g])) for g in candidates}
    if direction=='unsigned' and any(v < -epsilon for v in mag.values()):raise ValueError('Signed value given as unsigned')
    mag={g:v for g,v in mag.items() if v>epsilon}
    if not candidates:return {'available':False,'reason':'empty_operation_candidates'}
    if not mag:return {'available':False,'reason':'no_directional_candidates'}
    high=min(mag,key=lambda g:(-mag[g],g));n=candidates[high]['actual_fact_count']
    pool=sorted(g for g in mag if g!=high and candidates[g]['actual_fact_count']==n)
    if not pool:return {'available':False,'reason':'no_same_count_alternative','high':high,'actual_fact_count':n,'directional_candidates':len(mag)}
    low=min(pool,key=lambda g:(mag[g],g));rng=random.Random(str(seed));randoms=[rng.choice(pool) for _ in range(repeats)]
    return {'available':True,'high':high,'low':low,'random':randoms,'actual_fact_count':n,'pool_size':len(pool),'random_unique':len(set(randoms)),'directional_candidates':len(mag),'high_low_tie':abs(mag[high]-mag[low])<=epsilon,'high_magnitude':mag[high],'low_magnitude':mag[low]}
