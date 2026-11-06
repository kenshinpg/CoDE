"""Deterministic shared lab view. Server execution only; no model dependencies."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import time

NUM=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
EXACT=re.compile(r'^\s*=?\s*('+NUM+r')(?:\s+([A-Za-zµμ%#/][A-Za-zµμ0-9%/.*^_()-]*))?\s*$')
STATES=['LOW','WITHIN_RANGE','HIGH']
PHRASES={'LOW':'below the supplied reference interval','WITHIN_RANGE':'within the supplied reference interval','HIGH':'above the supplied reference interval'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try:yield json.loads(line)
                except ValueError as e:raise ValueError(f'{p}:{n}: {e}') from e

def dump(p,r):
    Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def parse(raw):
    if raw is None or isinstance(raw,str) and not raw.strip():return None,None,'missing'
    if isinstance(raw,bool) or not isinstance(raw,(str,int,float)):return None,None,'unsupported_type'
    m=EXACT.fullmatch(str(raw))
    if not m:return None,None,'not_exact_scalar'
    try:x=Decimal(m[1])
    except InvalidOperation:return None,None,'invalid_number'
    if not x.is_finite():return None,None,'nonfinite'
    return x,m[2],None

def signature(r):
    # Evidence ledger binds the exact source observation, including project and source locator.
    return hashlib.sha256(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def classify(r,evidence=None,unit_policy='strict'):
    raw=r['raw']['value'];v=r['value']
    x,ux,ex=parse(raw);lo,ul,el=parse(v.get('reference_lower'));hi,uh,eh=parse(v.get('reference_upper'))
    reasons=[]
    for name,err in [('value',ex),('lower',el),('upper',eh)]:
        if err:reasons.append(name+'_'+err)
    candidate=None
    if not reasons:
        if lo>hi:reasons.append('reversed_reference_bounds')
        else:candidate='LOW' if x<lo else 'HIGH' if x>hi else 'WITHIN_RANGE'
    known=[u for u in (ux,ul,uh) if u is not None]
    comparable=False;basis='unconfirmed'
    if len(set(known))>1:reasons.append('explicit_unit_mismatch')
    elif len(known)==3:
        comparable=True;basis='identical_explicit_unit_strings_no_conversion'
    elif evidence:
        if evidence.get('source_observation_sha256')!=signature(r):raise ValueError('Stale unit evidence: '+r['observation_id'])
        if evidence.get('decision')=='confirmed_same_scale':
            if not evidence.get('basis') or not evidence.get('evidence_reference'):
                raise ValueError('Unit evidence requires basis and evidence_reference')
            comparable=True;basis='record_bound_evidence:'+evidence['basis']
    confirmed=comparable
    assumed=False
    if not comparable and not evidence and unit_policy=='source_paired_assumption' and 'explicit_unit_mismatch' not in reasons:
        comparable=True;assumed=True;basis='source_paired_value_reference_same_scale_assumed_not_verified'
    if not comparable:reasons.append('unit_comparability_unconfirmed')
    return dict(candidate_state=candidate,state=candidate if comparable and not reasons else None,
                reasons=reasons,unit_comparability=dict(confirmed=confirmed,assumed=assumed,accepted_for_view=comparable,basis=basis,
                value_unit=ux,lower_unit=ul,upper_unit=uh),
                parsed_decimal_strings=dict(value=str(x) if x is not None else None,
                lower=str(lo) if lo is not None else None,upper=str(hi) if hi is not None else None))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--unit-evidence',type=Path)
    p.add_argument('--unit-policy',choices=['strict','source_paired_assumption'],default='source_paired_assumption')
    a=p.parse_args();source=a.source.resolve();a.out.mkdir(parents=True,exist_ok=False)
    inputs={n:sha(source/n) for n in ['observations_PRIVATE.jsonl','labels_PRIVATE.jsonl','feature_dictionary.json']}
    labels={}
    for r in rows(source/'labels_PRIVATE.jsonl'):
        if r['admission_id'] in labels:raise ValueError('Duplicate admission labels')
        labels[r['admission_id']]=r['reference_class']
    inventory=json.loads((source/'feature_dictionary.json').read_text(encoding='utf-8-sig'))['laboratory']
    dictionary={str(r['itemid']):r for r in inventory}
    if len(dictionary)!=len(inventory):raise ValueError('Duplicate dictionary item IDs')
    evidence={};evidence_sha=None
    if a.unit_evidence:
        evidence_sha=sha(a.unit_evidence)
        for r in rows(a.unit_evidence):
            if r['observation_id'] in evidence:raise ValueError('Duplicate evidence')
            evidence[r['observation_id']]=r
    counts=Counter();by_item={};by_disease=defaultdict(Counter);seen=set();keys=set();samples=Counter()
    patients={aid:[] for aid in labels};run='lab_discrete_v2_'+str(time.time_ns())
    view_id='lab_reference_state_v2_'+a.unit_policy
    with (a.out/'derivations_PRIVATE.jsonl').open('x',encoding='utf-8') as audit, (a.out/'examples_PRIVATE.jsonl').open('x',encoding='utf-8') as examples:
        for r in rows(source/'observations_PRIVATE.jsonl'):
            if r['source_type']!='laboratory':continue
            oid=r['observation_id'];aid=r['admission_id'];item=str(r['concept']['id'])
            if oid in seen or aid not in labels or (aid,item) in keys:raise ValueError('Duplicate/orphan snapshot '+oid)
            seen.add(oid);keys.add((aid,item))
            result=classify(r,evidence.get(oid),a.unit_policy);state=result['state'];meta=dictionary.get(item,{})
            if not meta.get('label'):
                result['reasons'].append('dictionary_label_missing');result['state']=state=None
            fact_id=oid+':'+view_id
            record=dict(observation_id=oid,admission_id=aid,source_observation=r,
                source_observation_sha256=signature(r),derivation=result,
                representation_group_id=r['representation_group_id'],fact_id=fact_id,
                rule='x<L:LOW; L<=x<=U:WITHIN_RANGE; x>U:HIGH',unit_conversion=False)
            audit.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            counts['observations']+=1;counts['candidate/'+str(result['candidate_state'])]+=1
            counts['visible/'+str(state)]+=1
            counts['unit_basis/'+result['unit_comparability']['basis']]+=1
            if state:counts['visible_unit_basis/'+result['unit_comparability']['basis']]+=1
            for reason in result['reasons']:counts['reason/'+reason]+=1
            by_disease[labels[aid]]['observations']+=1
            by_disease[labels[aid]]['visible']+=int(state is not None)
            st=by_item.setdefault(item,dict(itemid=item,label=meta.get('label'),fluid=meta.get('fluid'),observations=0,visible=0,states=Counter(),reasons=Counter()))
            st['observations']+=1;st['visible']+=int(state is not None);st['states'][str(state)]+=1;st['reasons'].update(result['reasons'])
            if state:
                # Whitelist export: no raw value, reference interval, unit evidence or labels.
                label=meta['label'];fluid=meta.get('fluid')
                display=label+(f' ({fluid})' if fluid else '')
                patients[aid].append(dict(fact_id=fact_id,feature_id='LAB_'+item,state=state,
                    display_text=display+': '+PHRASES[state],representation_group_id=r['representation_group_id'],
                    source_observation_ids=[oid]))
            category=state or '|'.join(result['reasons'])
            if samples[category]<5:
                examples.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');samples[category]+=1
            if counts['observations']%10000==0:print(f"Processed {counts['observations']:,} lab observations",flush=True)
    if not seen:raise ValueError('Empty laboratory source')
    if set(evidence)-seen:raise ValueError('Unknown observation IDs in unit evidence')
    for n,h in inputs.items():
        if sha(source/n)!=h:raise ValueError('Source changed: '+n)
    if a.unit_evidence and sha(a.unit_evidence)!=evidence_sha:raise ValueError('Evidence changed')
    with (a.out/'patient_lab_view_PRIVATE.jsonl').open('x',encoding='utf-8') as f:
        for aid,facts in patients.items():
            f.write(json.dumps(dict(admission_id=aid,view_id=view_id,facts=sorted(facts,key=lambda x:x['feature_id'])),ensure_ascii=False)+'\n')
    dump(a.out/'feature_dictionary.json',dict(view_id=view_id,features=[dict(feature_id='LAB_'+i,label=dictionary.get(i,{}).get('label'),fluid=dictionary.get(i,{}).get('fluid'),states=STATES) for i in sorted(by_item)]))
    dump(a.out/'item_statistics.json',list(by_item.values()))
    report=dict(dataset='TRACE-2400-v1',run_id=run,source=str(source),counts=dict(counts),
        admissions=len(patients),admissions_with_visible_labs=sum(bool(f) for f in patients.values()),
        unique_items=len(by_item),by_disease_PRIVATE=dict(by_disease),inputs=inputs,
        unit_evidence_sha256=evidence_sha,code_sha256=sha(Path(__file__)),view_id=view_id,unit_policy=a.unit_policy,
        assumption='Source-package paired value and reference interval share a scale; not independently verified' if a.unit_policy=='source_paired_assumption' else None,
        visible_items=sum(s['visible']>0 for s in by_item.values()),
        readiness='VIEW_READY_FOR_REVIEW' if any(patients.values()) else 'NO_VISIBLE_FACTS_UNIT_OR_VALUE_EVIDENCE_REQUIRED',
        notes=['Candidate states are not model input.','No diagnosis labels, precise values or bounds in patient view.',
        'No normal/abnormal patient inference. No HPO or LOINC mapping.','Snapshot, not first/last/all measurements.',
        'No admission removed. Empty facts remain explicit.','Sample examples are not a prevalence sample.'])
    dump(a.out/'report.json',report)
    (a.out/'stage_record.md').write_text('# 检验离散视图服务器阶段\n\n'+
        '处理完成；验收状态：'+report['readiness']+'。\n\n'+
        f"观测{len(seen)}条，住院{len(patients)}次，具有可见检验的住院{report['admissions_with_visible_labs']}次。\n\n"+
        '单位政策：'+a.unit_policy+'。源包同量纲假设不是逐条验证；显式冲突仍排除。完整统计与输入哈希见report.json。不得将生成文件等同正式实验验收通过。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
