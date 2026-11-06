"""Census and outcome-independent grouped split for the fixed LANTERN 10k pool.

No training or model calls. Extraction availability is never used to allocate
patients. Source files are read-only. Reports do not include explanation text.
"""
import argparse,hashlib,json,math,tarfile,time,traceback
from collections import Counter,defaultdict
from pathlib import Path

def rows(p):
    with p.open(encoding='utf-8-sig') as f:
        for s in f:
            if s.strip():yield json.loads(s)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stable(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def unique(p,key):
    out={}
    for r in rows(p):
        if r[key] in out:raise ValueError('Duplicate '+key+' in '+str(p))
        out[r[key]]=r
    return out

def grouped_split(items,seed):
    """Largest groups first; squared stratum/global allocation error, no scores."""
    names=['train','development','test'];ratios=[.7,.15,.15]
    groups=defaultdict(list);totals=Counter()
    for r in items:
        groups[r['pattern_id']].append(r);totals[(r['disease'],r['component'])]+=1
    counts=[Counter() for _ in names];sizes=[0,0,0];assigned={};train_diseases=set()
    for pattern,rs in sorted(groups.items(),key=lambda kv:(-len(kv[1]),stable([seed,kv[0]]))):
        delta=Counter((r['disease'],r['component']) for r in rs)
        # Guarantee fitting coverage when a disease appears in this fixed pool;
        # test presence cannot be promised for single-pattern diseases.
        choices=[0] if any(r['disease'] not in train_diseases for r in rs) else range(3)
        def cost(i):
            value=0.
            for s,n in delta.items():
                target=totals[s]*ratios[i];before=counts[i][s]-target
                value+=((before+n)**2-before**2)/max(target,1.)
            target=len(items)*ratios[i];before=sizes[i]-target
            return value+((before+len(rs))**2-before**2)/max(target,1.)
        choice=min(choices,key=lambda i:(cost(i),i));counts[choice].update(delta);sizes[choice]+=len(rs)
        if choice==0:train_diseases.update(r['disease'] for r in rs)
        for r in rs:assigned[r['patient_id']]=names[choice]
    return assigned

def run(a):
    sample=a.sample;export=a.export
    paths={'patients':sample/'data/patients_public.jsonl','dictionary':sample/'data/evidence_dictionary.json',
           'cases':sample/'selected_cases_private.jsonl','response_selection':sample/'selected_responses_private.jsonl',
           'generation':sample/'generation/results.jsonl','sets':export/'cosi_sets_40000.jsonl',
           'repairs':export/'citation_repairs.jsonl','manifest':export/'manifest.json','report':export/'report.json'}
    stamps={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths.values()}
    manifest=json.loads(paths['manifest'].read_text(encoding='utf-8-sig'))
    hashes={}
    for name,p in paths.items():
        hashes[str(p)]=sha(p)
        if name in ('sets','repairs','report'):
            assert manifest['artifacts'][p.name]==hashes[str(p)],'Export changed: '+name
    print('Input hashes verified',flush=True)
    patients=unique(paths['patients'],'patient_id');cases=unique(paths['cases'],'patient_id')
    assert set(patients)==set(cases) and len(patients)==10000
    sets=unique(paths['sets'],'explanation_id');assert len(sets)==40000
    public={};items=[];component=Counter();original=Counter();original_groups=defaultdict(set)
    for pid,p in patients.items():
        c=cases[pid];assert c['component'] in ('coverage','challenge')
        facts=[f for f in p['facts'] if f.get('visible',True) and f['coverage_scope']=='symptoms_antecedents']
        tokens=sorted({f['provenance']['source_token'] for f in facts});assert len(tokens)>0
        pat=hashlib.sha256(json.dumps(tokens,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        assert p['pattern_id']==pat
        public[pid]={'all_facts':{f['fact_id'] for f in p['facts']},'scored_facts':{f['fact_id'] for f in facts}}
        items.append(dict(patient_id=pid,pattern_id=pat,disease=c['disease'],component=c['component'],original_partition=c['partition']))
        component[c['component']]+=1;original[c['partition']]+=1;original_groups[pat].add(c['partition'])
    assert component=={'coverage':6000,'challenge':4000}
    assignment=grouped_split(items,a.seed)
    for r in items:r['partition']=assignment[r['patient_id']]
    seen=set();per_patient=Counter();statuses=Counter();flow=defaultdict(Counter);ledger=[]
    for g in rows(paths['generation']):
        eid=g['id'];assert eid in sets and eid not in seen;seen.add(eid)
        s=sets[eid];pid=s['patient_id'];assert g['patient_id']==pid and pid in patients
        c=cases[pid];assert s['partition']==c['partition'] and s['component']==c['component']
        available=s['reviewed_available'];status=s['repair_status'];assert available==(status=='COMPLETE')
        cited=s['new_strict_cited_fact_ids']
        if available:assert isinstance(cited,list) and len(cited)==len(set(cited)) and set(cited)<=public[pid]['all_facts']
        else:assert cited is None
        text=g['parsed']['explanation'];diagnoses=g['parsed']['ranked_diagnoses'];assert isinstance(text,str) and len(diagnoses)>0
        r=dict(explanation_id=eid,patient_id=pid,partition=assignment[pid],original_partition=c['partition'],disease=c['disease'],component=c['component'],repair_status=status,reviewed_available=available,text_sha256=stable(text),diagnoses_sha256=stable(diagnoses),word_count=len(text.split()),diagnosis_correct=diagnoses[0]==c['disease'],
               cited_fact_ids=sorted(set(cited)&public[pid]['scored_facts']) if available else None)
        ledger.append(r);per_patient[pid]+=1;statuses[status]+=1
        for key in [r['partition'],r['partition']+'|'+r['component'],r['partition']+'|'+r['disease'],'correct' if r['diagnosis_correct'] else 'wrong']:
            flow[key][status]+=1
    assert seen==set(sets) and set(per_patient)==set(patients) and set(per_patient.values())=={4}
    overlay_seen=set()
    for o in rows(paths['repairs']):
        eid=o['explanation_id'];assert eid in sets and eid not in overlay_seen;overlay_seen.add(eid)
        s=sets[eid];assert o['patient_id']==s['patient_id'] and o['status']==s['repair_status']
        if s['reviewed_available']:assert o['strict_cited_fact_ids']==s['new_strict_cited_fact_ids']
    assert overlay_seen==seen
    for p in paths.values():assert (p.stat().st_size,p.stat().st_mtime_ns)==stamps[str(p)],'Input mutated during census'
    with (a.out/'patient_split_PRIVATE.jsonl').open('w',encoding='utf-8') as f:
        for r in sorted(items,key=lambda r:r['patient_id']):f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (a.out/'explanation_ledger_PRIVATE.jsonl').open('w',encoding='utf-8') as f:
        for r in sorted(ledger,key=lambda r:r['explanation_id']):f.write(json.dumps(r,ensure_ascii=False)+'\n')
    strata=defaultdict(Counter);splits=Counter();patterns=defaultdict(set)
    for r in items:
        splits[r['partition']]+=1;strata[r['disease']+'|'+r['component']][r['partition']]+=1;patterns[r['pattern_id']].add(r['partition'])
    assert all(len(v)==1 for v in patterns.values())
    result=dict(status='census_and_split_created',patients=len(items),explanations=len(ledger),statuses=statuses,patient_splits=splits,
        observed_diseases=len({r['disease'] for r in items}),unique_patterns=len(patterns),original_cross_partition_patterns=sum(len(x)>1 for x in original_groups.values()),new_cross_partition_patterns=0,
        original_patient_partitions=original,disease_component_split=strata,availability=flow,seed=a.seed,target_ratios=[.7,.15,.15],
        sampling='All 10k predefined patients retained; no extraction/diagnosis-success sampling; complete-pattern grouping with disease/component stratification.',
        training_policy='All sources and probes refit using new train patients only; development for protocol checks; test is retrospectively designated, not previously unseen.',
        semantic_validation='Pending independent mapping review; COMPLETE is not semantic certification.',
        input_sha256=hashes,split_sha256=sha(a.out/'patient_split_PRIVATE.jsonl'),ledger_sha256=sha(a.out/'explanation_ledger_PRIVATE.jsonl'))
    dump(a.out/'cohort_report.json',result)
    return {'status':'complete','patients':10000,'explanations':40000,'patient_splits':splits,'next':'Review census and propagate these fixed IDs into new training and all RQ adapters; no experiments launched.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',type=Path,default=Path('inputs/ddxplus_sample'))
    p.add_argument('--export',type=Path,default=Path('inputs/ddxplus_mapping'))
    p.add_argument('--out',type=Path,default=Path('artifacts/code10k_protocol_v1'));p.add_argument('--seed',type=int,default=26100770)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);result={};start=time.time()
    try:result=run(a)
    except Exception:result={'status':'failed','error':traceback.format_exc()};raise
    finally:
        result['elapsed_seconds']=time.time()-start;result['script_sha256']=sha(Path(__file__));dump(a.out/'run_manifest.json',result)
        archive=a.out.with_name(a.out.name+'_reports.tar.gz')
        with tarfile.open(archive,'w:gz') as t:
            for f in a.out.glob('*.json*'):t.add(f,arcname=f.name)
        print('Return report:',archive,flush=True)
if __name__=='__main__':main()
