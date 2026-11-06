"""Pure-data contracts for B/C/E; no fitting or changed estimators."""
from collections import Counter, defaultdict
import math

TOL = 2e-5

def identity_audit(split, parts):
    index = {r['admission_id']: r for r in split}
    if len(index) != len(split): raise ValueError('Duplicate split admission')
    subjects = defaultdict(list)
    patterns = defaultdict(set)
    bypart = {}
    private = []
    for part, records in parts.items():
        ids = set()
        for r in records:
            pid = r['patient_id']; s = index[pid]
            if pid in ids or s['partition'] != part or r['partition'] != part or s['reference_class'] != r['label']:
                raise ValueError('Partition/label identity mismatch')
            if not s['subject_group']: raise ValueError('Missing subject group')
            ids.add(pid); subjects[s['subject_group']].append(s)
            # Recompute equality from actual model input, independently of stored pattern IDs.
            patterns[tuple(sorted(r['tokens']))].add(part)
            private.append(dict(s,pattern_id=r['pattern_id']))
        if ids != {k for k,v in index.items() if v['partition']==part}: raise ValueError('Incomplete partition')
        bypart[part] = dict(admissions=len(ids),subjects=len({index[k]['subject_group'] for k in ids}),
                            diseases=dict(Counter(index[k]['reference_class'] for k in ids)))
    if sum(len(x) for x in parts.values()) != len(split): raise ValueError('Incomplete split census')
    cross_subject = sum(len({r['partition'] for r in v})>1 for v in subjects.values())
    pairs = {a+'__'+b:sum(a in p and b in p for p in patterns.values())
             for a,b in [('train','development'),('train','test'),('development','test')]}
    out = dict(admissions=len(split),subjects=len(subjects),partitions=bypart,
        repeated_subjects=sum(len(v)>1 for v in subjects.values()),
        cross_disease_subjects=sum(len({r['reference_class'] for r in v})>1 for v in subjects.values()),
        cross_partition_subjects=cross_subject,full_input_pattern_overlap=pairs,
        subject_basis='inferred subject groups; authority requires source confirmation',
        clinical_leakage_cleared=False,development_exposure_cleared=False)
    if cross_subject or any(pairs.values()): raise ValueError('Cross-partition subject or full-input pattern detected')
    return out,private

def index_records(records, keys):
    out = {tuple(r[k] for k in keys):r for r in records}
    if len(out)!=len(records): raise ValueError('Duplicate comparison records')
    return out

def equal(a,b,path='',tol=TOL):
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys(): raise ValueError('Key drift: '+path)
        return max([equal(a[k],b[k],path+'/'+str(k),tol) for k in a]+[0.])
    if isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b): raise ValueError('Length drift: '+path)
        return max([equal(x,y,path,tol) for x,y in zip(a,b)]+[0.])
    if a is None or b is None or isinstance(a,(str,bool)) or isinstance(b,(str,bool)):
        if a!=b: raise ValueError('State drift: '+path)
        return 0.
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        d=abs(a-b)
        if not math.isfinite(d) or d>tol: raise ValueError('Numeric drift: '+path)
        return d
    if a!=b: raise ValueError('Value drift: '+path)
    return 0.

def merge_rq3(old,new):
    ix=index_records(old,['explanation_id','probe']); nx=index_records(new,['explanation_id','probe'])
    if ix.keys()!=nx.keys(): raise ValueError('RQ3 census drift')
    merged=[]; maximum=0.; changes=Counter(); max_tabr=Counter()
    for key,o in ix.items():
        n=nx[key]; left={k:v for k,v in o.items() if k not in ['scores','coverage_unavailability']}
        right={k:v for k,v in n.items() if k not in ['scores','coverage_unavailability']}
        # Endpoints, sets and metadata are exact; probabilities use the original tolerance.
        for k in left:
            maximum=max(maximum,equal(left[k],right[k],k,TOL if k.endswith('probabilities') else 0))
        maximum=max(maximum,equal({k:v for k,v in o['scores'].items() if not k.startswith('tabr_')},
                                  {k:v for k,v in n['scores'].items() if not k.startswith('tabr_')},'nonTabR'))
        equal({k:v for k,v in o['coverage_unavailability'].items() if not k.startswith('tabr_')},
              {k:v for k,v in n['coverage_unavailability'].items() if not k.startswith('tabr_')})
        r=dict(o,scores=dict(o['scores']),coverage_unavailability=dict(o['coverage_unavailability']))
        for k in ['tabr_coverage','tabr_absolute','tabr_retained']:
            x,y=o['scores'][k],n['scores'][k]
            changes[k+'_availability'] += (x is None)!=(y is None)
            if x is not None and y is not None:
                max_tabr[k]=max(max_tabr[k],abs(x-y));changes[k+'_over_tolerance']+=abs(x-y)>TOL
            r['scores'][k]=y
        r['coverage_unavailability']['tabr_coverage']=n['coverage_unavailability']['tabr_coverage']
        merged.append(r)
    return merged,dict(rows=len(merged),non_tabr_max_delta=maximum,tabr_max_delta=dict(max_tabr),changes=dict(changes),
                       original_sets_endpoints_and_non_tabr_values_preserved=True)

def merge_rq1(old,new):
    keys=['patient_id','selector','probe','fraction'];ix=index_records(old,keys);nx=index_records(new,keys)
    if ix.keys()!=nx.keys(): raise ValueError('RQ1 census drift')
    out=[]; changes=Counter();maximum=0.
    for key,o in ix.items():
        n=nx[key]
        if o['selector']!='tabr':
            maximum=max(maximum,equal(o,n,'RQ1 nonTabR',0));out.append(o);continue
        for k in ['patient_id','subject_group','probe','fraction','both_full_correct','selector']: equal(o[k],n[k],k,0)
        changes['selection_rows']+=1
        changes['selected_set_changed']+=o.get('selected_tokens')!=n.get('selected_tokens')
        changes['availability_changed']+=o['available']!=n['available']
        changes['outcome_changed']+=o.get('correct')!=n.get('correct')
        if o.get('selected_tokens')==n.get('selected_tokens'): equal(o,n,'RQ1 unchanged selection',0)
        out.append(n)
    return out,dict(changes,non_tabr_max_delta=maximum,non_tabr_rows_preserved=True,
                    random_controls='Original seed and fact-count matching; unchanged selections must reproduce exactly')

def unavailable_audit(parts, scores):
    vocabulary={t for r in parts['train'] for t in r['tokens']}
    test={r['patient_id']:r for r in parts['test']};unseen={p:set(r['tokens'])-vocabulary for p,r in test.items()}
    rs=[r for r in scores if r['probe']=='encoder'];index_records(rs,['explanation_id'])
    if {r['patient_id'] for r in rs}-test.keys(): raise ValueError('RQ2 outside test')
    result=[];private=[]
    for disease in sorted({r['label'] for r in test.values()}):
        ids={p for p,r in test.items() if r['label']==disease};bad={p for p in ids if unseen[p]}
        mapped=[r for r in rs if r['patient_id'] in ids and r['mapping_available']]
        excluded=[r for r in mapped if not r['probe_available']]
        explained={r['patient_id'] for r in excluded}
        result.append(dict(disease=disease,test_admissions=len(ids),full_input_unavailable_admissions=len(bad),
            unseen_token_occurrences=sum(len(unseen[p]) for p in ids),unique_unseen_tokens=len(set().union(*(unseen[p] for p in ids))),
            mapped_explanations=len(mapped),probe_excluded_explanations=len(excluded),probe_excluded_admissions=len(explained),
            excluded_without_full_unseen=len(explained-bad),full_unseen_without_mapped_exclusion=len(bad-explained)))
    for p,r in test.items():
        if unseen[p]:private.append(dict(patient_id=p,disease=r['label'],unseen_tokens=sorted(unseen[p])))
    totals={k:sum(r[k] for r in result) for k in result[0] if k not in ['disease','unique_unseen_tokens']} if result else {}
    return dict(by_disease=result,totals=totals,training_vocabulary_size=len(vocabulary),
        unseen_occurrences_by_partition={part:sum(len(set(r['tokens'])-vocabulary) for r in rr) for part,rr in parts.items()},
        note='Full-input training-token support and complete-mapping probe exclusion are separately counted'),private
