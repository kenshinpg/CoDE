"""Deterministic admission grouping and text-view policies, independent of models."""
import hashlib,json,random,re
from collections import Counter,defaultdict

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def original_links(o):
    if o['source_type']=='radiology':
        return [(r['hpo_id'],r.get('label') or r.get('hpo_label') or r.get('hpo_name')) for r in o.get('concept_links',[]) if r.get('hpo_id')]
    c=o.get('concept') or {};return [(c['id'],c.get('label'))] if c.get('id') else []
def choose(o,blocked_groups):
    if o['source_type']=='laboratory':return 'lab_separate_view',None,None
    if (o['admission_id'],o['representation_group_id']) in blocked_groups:return 'diagnosis_keyword_group',None,None
    if o['source_type']=='radiology' and o.get('provenance',{}).get('unit_kind')!='parent':return 'radiology_atom_not_selected',None,None
    links={c for c,n in original_links(o)}
    if not links:return 'no_original_concept',None,None
    if len(links)!=1:return 'multiple_original_concepts',None,None
    a=o.get('attributes',{}).get('assertion',{}).get('value')
    if a not in ('present','absent'):return 'assertion_not_selected',None,None
    return 'candidate',next(iter(links)),a

def grouped_split(labels,subjects,config):
    groups=defaultdict(list)
    for a in sorted(labels):
        if not subjects.get(a):raise ValueError('Missing subject grouping')
        groups[subjects[a]].append(a)
    classes=sorted(set(labels.values()));fractions=config['split_fractions'];parts=list(fractions)
    if abs(sum(fractions.values())-1)>1e-12 or any(x<=0 for x in fractions.values()):raise ValueError('Split fractions invalid')
    totals=Counter(labels.values());targets={p:{c:totals[c]*fractions[p] for c in classes} for p in parts}
    current={p:Counter() for p in parts};assignment={};rng=random.Random(config['seed'])
    order=sorted(groups);rng.shuffle(order);order.sort(key=lambda g:-len(groups[g]))
    for g in order:
        counts=Counter(labels[a] for a in groups[g]);scores={}
        for p in parts:
            # Increment in squared deviation from class targets, normalized by class size.
            scores[p]=sum(((current[p][c]+n-targets[p][c])**2-(current[p][c]-targets[p][c])**2)/totals[c] for c,n in counts.items())
        p=min(parts,key=lambda p:(scores[p],parts.index(p)))
        current[p].update(counts)
        for a in groups[g]:assignment[a]=p
    if any(current[p][c]==0 for p in parts for c in classes):raise ValueError('A class has no support in a split')
    return assignment,{p:dict(current[p]) for p in parts}
