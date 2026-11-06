"""Role-restricted citations using the existing adjudication scoring/statistics contracts."""
import copy
from collections import Counter
from . import adjudication_sensitivity as legacy

SETS = {'all': 'E_all', 'first': 'E_first', 'support': 'E_support'}


def resolve(roles, keys):
    index = {(k['dataset'], k['case']): k for k in keys}
    if len(index) != len(keys):
        raise ValueError('Duplicate review key')
    seen = set(); out = []
    for r in roles:
        key = r['dataset'], r['case']
        if key in seen or key not in index:
            raise ValueError('Duplicate or unknown role case')
        seen.add(key); k = index[key]
        names = list(SETS.values()) + ['E_oppose', 'negative_facts', 'negative_with_argument']
        for name in names:
            if len(r[name]) != len(set(r[name])) or not set(r[name]) <= set(r['E_all']):
                raise ValueError('Role set outside E_all or duplicate: ' + name)
        if set(r['E_first']) != set(r['E_support']) | set(r['E_oppose']):
            raise ValueError('First-diagnosis union mismatch')
        if not set(r['negative_with_argument']) <= set(r['negative_facts']) or r['uncertain']:
            raise ValueError('Unresolved role or inconsistent negative-use set')
        if k['stratum'] not in ['random', 'targeted']:
            raise ValueError('Unknown audit stratum')
        mapped = {v: sorted(k['aliases'][alias] for alias in r[field]) for v, field in SETS.items()}
        if any(len(x) != len(set(x)) for x in mapped.values()):
            raise ValueError('Alias collision')
        if set(mapped['all']) != set(k['final_fact_ids']):
            raise ValueError('All citations differ from the adjudicated reference')
        out.append(dict(k, audit_id=k['case'], sets=mapped, role_first_diagnosis=r['first_diagnosis'],
                        random_sample=k['stratum']=='random', targeted_sample=k['stratum']=='targeted',
                        status='complete', unresolved_fact_ids=[], negative_count=len(r['negative_facts']),
                        oppose_count=len(r['E_oppose']),
                        support_and_oppose_count=len(set(r['E_support']) & set(r['E_oppose'])),
                        negative_argument_count=len(r['negative_with_argument'])))
    if seen != set(index):
        raise ValueError('Missing role case')
    return out


def make_ledger(reviews, original, facts):
    ledger=[]; lookup={}
    for review in reviews:
        r=copy.deepcopy(review); e=original[r['explanation_id']]
        if e['diagnosis'] != r['role_first_diagnosis']:
            raise ValueError('Role first diagnosis differs from frozen proposal')
        r['automatic_fact_ids']=list(e['cited_fact_ids'])
        for version in SETS:
            rr=dict(r,final_fact_ids=r['sets'][version])
            ll, _ = legacy.make_ledger([rr], original, facts)
            for row, old_version in zip(ll,['automatic','adjudicated']):
                if old_version=='automatic' and version!='all':
                    continue
                v='automatic' if old_version=='automatic' else version
                eid=r['explanation_id']+'::roles20261023::'+v
                row['explanation_id']=eid; ledger.append(row); lookup[eid]=(r,v)
    return ledger,lookup


def cook(scored,lookup,old,labels):
    expected={(eid,p) for eid in lookup for p in legacy.PROBES}
    actual=[(r['explanation_id'],r['probe']) for r in scored]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Unexpected, duplicate or missing role scores')
    result=[]; checks={}
    for v in SETS:
        selected={eid:(dict(r,final_fact_ids=r['sets'][v]),'automatic' if variant=='automatic' else 'adjudicated')
                  for eid,(r,variant) in lookup.items() if variant in ['automatic',v]}
        rr,check=legacy.cook([r for r in scored if r['explanation_id'] in selected],selected,old,labels)
        for row in rr:
            if row['version']=='adjudicated':
                row['version']=v; result.append(row)
        checks[v]=check
    return result,checks


def make_panels(records):
    idx={(r['dataset'],r['explanation_id'],r['probe'],r['version']):r for r in records}
    if len(idx)!=len(records):raise ValueError('Duplicate role analysis record')
    panels=[]; flows=[]
    # Every pair uses availability under ALL THREE sets, not separate changing populations.
    for restriction in ['first','support']:
        rr=[]
        for base in records:
            if base['version']!='all':continue
            triplet=[idx.get((base['dataset'],base['explanation_id'],base['probe'],v)) for v in SETS]
            if any(x is None for x in triplet):raise ValueError('Incomplete three-set record')
            for source,version in [(triplet[0],'automatic'),(triplet[1 if restriction=='first' else 2],'adjudicated')]:
                r=copy.deepcopy(source); r['version']=version
                for field in ['both_full_correct','mapping_available','target_supported','probe_available']:
                    r[field]=all(x[field] for x in triplet)
                for m in legacy.CORE_SCORES:
                    if any(x['scores'].get(m) is None for x in triplet):r['scores'][m]=None
                rr.append(r)
        pp,ff=legacy.make_panels(rr)
        meta=dict(base_set='all',restricted_set=restriction,common_scope='all_first_support')
        for p in pp:
            p.update(meta,panel_id=len(panels)); panels.append(p)
        flows.extend(dict(f,**meta) for f in ff)
    return panels,flows


def rename_result(r):
    replacements=[('adjudicated_fixed_endpoint','restricted_fixed_all_endpoint'),
                  ('adjudicated_updated','restricted_updated'),
                  ('change_fixed_endpoint','change_fixed_all_endpoint'),
                  ('automatic','all'),('adjudicated','restricted')]
    out={}
    for k,v in r.items():
        for before,after in replacements:
            if k==before or k.startswith(before+'_'):
                k=after+k[len(before):];break
        out[k]=v
    return out


def finish_panel(panel,draws,reps,seed):
    rr,tt=legacy.finish_panel(panel,draws,reps,seed)
    return [rename_result(r) for r in rr],[rename_result(r) for r in tt]


def census(reviews):
    out=[]
    for ds in ['DDXPlus','TRACE']:
        for stratum in ['random','targeted']:
            rr=[r for r in reviews if r['dataset']==ds and r['stratum']==stratum]
            out.append(dict(dataset=ds,audit_group=stratum,explanations=len(rr),
                            patients=len({r['patient_id'] for r in rr}),
                            **{v+'_citations':sum(len(r['sets'][v]) for r in rr) for v in SETS},
                            **{v+'_empty':sum(not r['sets'][v] for r in rr) for v in SETS},
                            opposed_citations=sum(r['oppose_count'] for r in rr),
                            explanations_with_opposition=sum(r['oppose_count']>0 for r in rr),
                            support_and_oppose_citations=sum(r['support_and_oppose_count'] for r in rr),
                            negative_facts=sum(r['negative_count'] for r in rr),
                            negative_with_argument=sum(r['negative_argument_count'] for r in rr)))
    return out
