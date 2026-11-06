"""Bind the eight AI-reviewed negative citations to signed state replacements."""
import hashlib
from collections import Counter
from datetime import date
from .f_repair import index


def verify_rematches(approved, reviews, documents, ledger=None, transitions=None):
    actions={str(r['admission_id']):r for r in approved if r['approved_action']=='REPLACE'}
    if len(actions)!=2:raise ValueError('Expected two approved state replacements')
    ri=index(reviews,['patient_id','explanation_id']);di=index(documents,['patient_id','explanation_id'])
    if len(ri)!=8 or ri.keys()!=di.keys():raise ValueError('Rematch document/review census')
    if Counter(str(r['patient_id']) for r in reviews)!=Counter({p:4 for p in actions}):raise ValueError('Four explanations per replacement admission required')
    if ledger is not None:
        expected={(str(r['patient_id']),r['explanation_id']) for r in ledger if str(r['patient_id']) in actions}
        if ri.keys()!=expected:raise ValueError('Rematch differs from frozen explanation ledger')
    audit=[]
    for key,r in ri.items():
        d=di[key];act=actions[key[0]];target=d['target']
        if d['task_id']!=r['explanation_id']:raise ValueError('Task identity mismatch')
        fields={'review_id':act['correction_id'],'old_fact_id':act['fact_id'],
                'new_fact_id':act['replacement_fact_id'],'feature_id':act['replacement_feature_id'],
                'new_state':act['replacement_state']}
        for name,value in fields.items():
            if r[name]!=value or target[name]!=value:raise ValueError('Rematch target differs: '+name)
        if str(target['patient_id'])!=key[0]:raise ValueError('Cross-admission target')
        if transitions is not None and transitions[key[0],r['old_fact_id']]['new_fact_id']!=r['new_fact_id']:
            raise ValueError('Derived replacement identity differs from reviewed target')
        for name in ['input_view_id','input_digest','rollout_index','source_file','source_line','source_row_sha256']:
            if r[name]!=d[name]:raise ValueError('Document provenance differs: '+name)
        text=d['text'];h=hashlib.sha256(text.encode('utf-8')).hexdigest()
        if r['text_sha256']!=h or d['text_sha256']!=h:raise ValueError('Full-text hash mismatch')
        extent=r['reviewed_text_range']
        if extent['start']!=0 or extent['end']!=len(text) or extent['offset_unit']!='Unicode code points':
            raise ValueError('Full-text review extent mismatch')
        for q in r.get('context_checks',[]):
            if not 0<=q['start']<q['end']<=len(text) or text[q['start']:q['end']]!=q['text']:raise ValueError('Context span mismatch')
        if r['decision']!='DOES_NOT_CITE_NEW' or r['match_status']!='not_referenced' or r['new_fact_inclusion_in_E'] is not False:
            raise ValueError('This update supports the eight verified negative citations only')
        if r['quotes'] or r['target_spans'] or r['target_fact_pending'] is not False:raise ValueError('Negative citation status inconsistent')
        if r['reviewer_kind']!='AI' or r['human_rematch_confirmed'] is not False or r['review_method']!='AI_full_text_review':
            raise ValueError('Rematch reviewer provenance changed')
        if r['confirmed_by']!='Codex_AI' or not r['source_fact_action_human_approved']:raise ValueError('Action/rematch provenance mismatch')
        date.fromisoformat(r['confirmed_date'])
        if not r.get('reason') or not r.get('evidence_reference'):raise ValueError('Missing review rationale')
        old_cited=None
        if ledger is not None:
            e=next(x for x in ledger if (str(x['patient_id']),x['explanation_id'])==key)
            old_cited=r['old_fact_id'] in (e['cited_fact_ids'] or [])
        audit.append(dict(patient_id=key[0],explanation_id=key[1],correction_id=act['correction_id'],
            old_fact_id=r['old_fact_id'],new_fact_id=r['new_fact_id'],decision=r['decision'],
            review_method=r['review_method'],confirmed_by=r['confirmed_by'],human_rematch_confirmed=False,
            status='AI_TARGET_REMATCH_COMPLETED',old_fact_in_consumed_E=old_cited,
            new_fact_added=False,text_sha256=h,evidence_reference=r['evidence_reference']))
    return audit
