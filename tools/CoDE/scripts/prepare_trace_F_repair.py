"""Prepare a separate approved test-F repair overlay. Does not run scoring.

init exports the existing 85 proposals with blank confirmation columns.
check validates approval completeness without opening model inputs.
prepare verifies frozen files and emits affected patient/ledger overlays.
"""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from code_eval.trace2400.f_repair import APPROVAL_FIELDS,validate_actions,patch_patients,rebind,index


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f)) if p.suffix.lower()=='.csv' else [json.loads(s) for s in f if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def emit(p,rs):
    with p.open('w',encoding='utf-8') as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rs):
    fields=list(dict.fromkeys(k for r in rs for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rs:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def run(a):
    proposals=rows(a.proposals)
    index(proposals,['admission_id','fact_id'])
    if len(proposals)!=85:raise ValueError('Expected the 85-item proposal package')
    # Output is always a new directory; never overwrite a previous preparation.
    a.out.mkdir(parents=True,exist_ok=False)
    if a.mode=='init':
        template=[dict(r,**{k:('PENDING' if k=='approval_status' else '') for k in APPROVAL_FIELDS}) for r in proposals]
        table(a.out/'source_fact_corrections_confirmation_PRIVATE.csv',template)
        dump(a.out/'F_repair_manifest.json',dict(status='awaiting_human_actions',proposals=85,
             proposal_sha256=sha(a.proposals),scoring_started=False,F_changed=False))
        print('Prepared 85 existing proposals with blank confirmation fields; no actions applied.');return
    if a.approved is None:raise ValueError('--approved is required')
    approvals=rows(a.approved)
    pending=validate_actions(proposals,approvals)
    status=dict(status='pending_human_confirmation' if pending else 'actions_validated_not_scored',
        proposal_sha256=sha(a.proposals),approval_sha256=sha(a.approved),pending_ids=pending,
        F_changed=False,models_refit=False,scoring_started=False)
    dump(a.out/'F_repair_manifest.json',status)
    if a.mode=='check':
        print(json.dumps(dict(status=status['status'],pending=len(pending))));return
    if pending:raise ValueError('Human confirmation incomplete; see F_repair_manifest.json')
    if a.view is None or a.auto_ledger is None or a.human_sets is None:
        raise ValueError('prepare requires --view, --auto-ledger and --human-sets')
    manifest=read(a.view/'manifest.json')
    names=['patients_model_input_PRIVATE.jsonl','test_records_PRIVATE.jsonl','split_PRIVATE.jsonl',
           'evidence_dictionary.json','train_records_PRIVATE.jsonl','development_records_PRIVATE.jsonl']
    bindings={}
    for name in names:
        p=a.view/name;h=sha(p)
        if manifest['output_sha256'].get(name)!=h:raise ValueError('Frozen view identity differs: '+name)
        bindings[name]=h
    patients=rows(a.view/names[0]);test=rows(a.view/names[1]);split=rows(a.view/names[2]);dictionary=read(a.view/names[3])
    original={r['patient_id']:r for r in patients}
    affected,records,audit,transitions=patch_patients(patients,test,split,dictionary,proposals,approvals)
    corrected={r['patient_id']:r for r in affected}
    decisions=rows(a.citation_decisions) if a.citation_decisions else []
    decision_index=index(decisions,['mapping','explanation_id','old_fact_id'])
    used_decisions=set();ledgers=[];citation_pending=[]
    for mapping,path in [('automatic',a.auto_ledger),('human_exact',a.human_sets)]:
        for r in rows(path):
            if mapping=='human_exact' and r['dataset']!='TRACE':continue
            pid=r['patient_id'];eid=r['explanation_id']
            if pid not in corrected:continue
            ids=r.get('cited_fact_ids') if mapping=='automatic' else r['human_return_exact_E']
            if ids is not None and not set(ids)<={f['fact_id'] for f in original[pid]['facts']}:
                raise ValueError('Original citation is outside frozen patient F')
            result=rebind(ids,pid,eid,mapping,transitions,{f['fact_id'] for f in corrected[pid]['facts']},decision_index)
            for fid in ids or []:
                k=(mapping,eid,fid)
                if k in decision_index:used_decisions.add(k)
            updated=dict(r,repair_mapping=mapping,original_cited_fact_ids=ids,
                         confirmed_corrected_fact_ids=result['confirmed_ids'],
                         replacement_mapping_pending=result['pending'],repair_binding_available=result['available'],
                         removed_old_citations=result['removed'],generated_from_original_F=True)
            if mapping=='automatic':
                updated['cited_fact_ids']=result['confirmed_ids'] if result['available'] else None
                updated['reviewed_available']=r['reviewed_available'] and result['available']
            ledgers.append(updated)
            for p in result['pending']:
                citation_pending.append(dict(mapping=mapping,patient_id=pid,explanation_id=eid,**p,
                    decision='UNRESOLVED',confirmed_by='',confirmed_date='',evidence_reference=''))
    if used_decisions!=decision_index.keys():raise ValueError('Citation decisions contain unmatched identities')
    # All four natural explanations must remain represented for every touched admission.
    auto=[r for r in ledgers if r['repair_mapping']=='automatic']
    for pid in corrected:
        if len([r for r in auto if r['patient_id']==pid])!=4:raise ValueError('Incomplete frozen rollout census')
    emit(a.out/'corrected_patients_overlay_PRIVATE.jsonl',affected)
    emit(a.out/'corrected_test_records_overlay_PRIVATE.jsonl',records)
    emit(a.out/'rebound_explanation_overlay_PRIVATE.jsonl',ledgers)
    table(a.out/'F_repair_actions_PRIVATE.csv',audit)
    table(a.out/'replacement_citations_to_confirm_PRIVATE.csv',citation_pending)
    work=[]
    for r in records:
        changed=r['fact_repair_status']=='CHANGED'
        work.append(dict(patient_id=r['patient_id'],input_changed=changed,empty_F=r['empty_F'],
             RQ1='recompute_selection_controls_and_full_eligibility' if changed else 'reuse_frozen',
             RQ2='recompute_full_attribution_denominators_E_scores_and_endpoints' if changed else 'reuse_frozen',
             RQ3='fixed_original_sets_and_regenerated_budget_sets_separate' if changed else 'reuse_frozen',
             original_cluster_id=r['pattern_id'],corrected_pattern_id=r['corrected_pattern_id']))
    table(a.out/'F_repair_scoring_worklist_PRIVATE.csv',work)
    status.update(status='overlay_prepared_NOT_SCORED',frozen_view_sha256=bindings,
        automatic_ledger_sha256=sha(a.auto_ledger),human_sets_sha256=sha(a.human_sets),
        citation_decisions_sha256=sha(a.citation_decisions) if a.citation_decisions else None,
        touched_admissions=len(records),changed_admissions=sum(r['fact_repair_status']=='CHANGED' for r in records),
        deferred_items=sum(a['approved_action']=='DEFER' for a in approvals),
        pending_replacement_citations=len(citation_pending),
        frozen_F_modified=False,corrected_overlay_created=True,
        original_generation_inputs_changed=False,formal_tables_replaced=False,
        limitation='Prepared data overlay only. Frozen scoring and paired result aggregation remain to be executed.',
        output_sha256={p.name:sha(p) for p in a.out.iterdir() if p.is_file() and p.name!='F_repair_manifest.json'})
    # Guard against any accidental write to the original view.
    if any(sha(a.view/n)!=h for n,h in bindings.items()):raise ValueError('Original view unexpectedly changed')
    dump(a.out/'F_repair_manifest.json',status)
    print(json.dumps(dict(status=status['status'],changed_admissions=status['changed_admissions'],pending_citations=len(citation_pending))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['init','check','prepare'])
    for name in ['proposals','approved','view','auto-ledger','human-sets','citation-decisions','out']:
        p.add_argument('--'+name,type=Path,required=name in ('proposals','out'))
    run(p.parse_args())
