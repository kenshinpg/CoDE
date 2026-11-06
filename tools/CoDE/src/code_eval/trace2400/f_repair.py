"""Explicitly approved, test-only fact patches. Pure data adapter; no model fit.

Frozen files are never edited. A replacement fact does not inherit the old
explanation citation without a separate explicit citation decision.
"""
import copy
import hashlib
import json
from collections import Counter
from datetime import date

APPROVAL_FIELDS = ['approval_status','approved_action','confirmed_by','confirmed_date',
                   'decision_reason','replacement_feature_id','replacement_state',
                   'replacement_display_text','replacement_encoding_verified',
                   'input_scope_verified','approved_evidence_reference']
ACTIONS = {'REMOVE','REPLACE','KEEP','DEFER','WITHHOLD'}


def normalize_approved(rows):
    """Translate the signed LANTERN action vocabulary; retain the signed fields."""
    out=[]
    action_map={'REMOVE_FROM_DERIVED_F':'REMOVE','REPLACE_STATE':'REPLACE',
                'WITHHOLD_UNRESOLVED_IN_STRICT_VIEW':'WITHHOLD'}
    for raw in rows:
        if raw.get('confirmation_status')!='APPROVED':raise ValueError('Unsigned action')
        if raw['approved_action'] not in action_map:raise ValueError('Unknown signed action')
        r=dict(raw, signed_approved_action=raw['approved_action'],approval_status='CONFIRMED',
               approved_action=action_map[raw['approved_action']],confirmed_date=raw['confirmation_date'],
               decision_reason=raw['confirmation_reason'],
               approved_evidence_reference=raw['verified_source_evidence'])
        if r['approved_action']=='REPLACE':
            r.update(replacement_feature_id=raw['feature_id'],replacement_state=raw['proposed_new_state'],
                replacement_display_text=raw['proposed_new_display_text'],replacement_fact_id=raw['proposed_new_fact_id'],
                replacement_encoding_verified='YES',input_scope_verified='YES')
            if not r['replacement_fact_id']:raise ValueError('Approved replacement identity missing')
        out.append(r)
    return out


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                     separators=(',',':')).encode()).hexdigest()


def index(records, fields):
    out = {}
    for r in records:
        key = tuple(str(r[k]) for k in fields)
        if key in out:
            raise ValueError('Duplicate identity: '+repr(key))
        out[key] = r
    return out


def validate_actions(proposals, approvals):
    expected = index(proposals, ['correction_id'])
    received = index(approvals, ['correction_id'])
    if expected.keys() != received.keys():
        raise ValueError('Approval table does not cover the exact proposal identities')
    pending=[]
    for key,p in expected.items():
        a=received[key]
        for col in ['admission_id','fact_id','feature_id','old_state']:
            if str(p[col]) != str(a[col]):
                raise ValueError('Approval identity differs: '+repr(key)+' '+col)
        status=a.get('approval_status','').strip()
        if status in ('','PENDING'):
            pending.append(p['correction_id']);continue
        if status not in ('CONFIRMED','DEFERRED'):
            raise ValueError('Unknown approval_status: '+status)
        action=a.get('approved_action','').strip()
        if action not in ACTIONS or (status=='DEFERRED') != (action=='DEFER'):
            raise ValueError('Status/action mismatch: '+repr(key))
        for col in ['confirmed_by','confirmed_date','decision_reason']:
            if not a.get(col,'').strip():raise ValueError('Missing '+col+' '+repr(key))
        date.fromisoformat(a['confirmed_date'])
        if action=='REPLACE':
            for col in ['replacement_feature_id','replacement_state','replacement_display_text','approved_evidence_reference']:
                if not a.get(col,'').strip():raise ValueError('Missing replacement '+col)
            if a.get('replacement_encoding_verified')!='YES' or a.get('input_scope_verified')!='YES':
                raise ValueError('Replacement encoding and original input scope must be verified')
        if action=='KEEP' and (a.get('input_scope_verified')!='YES' or not a.get('approved_evidence_reference','').strip()):
            raise ValueError('KEEP requires confirmed source evidence within original input scope')
    return pending


def validate_fact(f,dictionary):
    meta=dictionary['features'].get(f['feature_id'])
    if meta is None or meta.get('type')!='C' or f['state'] not in meta.get('values',{}):
        raise ValueError('Replacement cannot be encoded by frozen dictionary: '+f['feature_id']+' '+f['state'])
    if meta['group_id']!=f['group_id']:
        raise ValueError('Frozen group identity differs')


def patch_patients(patients, test_records, split, dictionary, proposals, approvals):
    pending=validate_actions(proposals,approvals)
    if pending:raise ValueError('Unconfirmed actions; cannot construct corrected F: '+','.join(pending))
    pp=index(copy.deepcopy(patients),['patient_id']); rr=index(test_records,['patient_id'])
    ss=index(split,['admission_id']); old=copy.deepcopy(pp)
    audits=[]; transitions={}; touched=set()
    for a in approvals:
        pid=str(a['admission_id']); fid=a['fact_id']; k=(pid,)
        if k not in ss or ss[k]['partition']!='test' or k not in rr:
            raise ValueError('Repair outside frozen test partition: '+pid)
        fs=index(old[k]['facts'],['fact_id']); current=index(pp[k]['facts'],['fact_id'])
        if (fid,) not in fs or (fid,) not in current:raise ValueError('Old fact missing or multiply acted on')
        f=fs[(fid,)]
        if f['feature_id']!=a['feature_id'] or f['state']!=a['old_state']:
            raise ValueError('Old feature/state differs from frozen F')
        action=a['approved_action'];new_id=fid
        if action in ('REMOVE','REPLACE','WITHHOLD'):
            pp[k]['facts']=[x for x in pp[k]['facts'] if x['fact_id']!=fid]
            new_id=None
        if action=='REPLACE':
            feature=a['replacement_feature_id'];state=a['replacement_state']
            if feature==f['feature_id'] and state==f['state']:
                raise ValueError('Display-only change is not an encoded semantic repair')
            meta=dictionary['features'].get(feature,{})
            nf=dict(fact_id=a.get('replacement_fact_id') or 'FR_'+digest([pid,fid,feature,state])[:24],feature_id=feature,state=state,
                    display_text=a['replacement_display_text'],group_id=meta.get('group_id'),supersedes_fact_id=fid)
            validate_fact(nf,dictionary)
            same=[x for x in pp[k]['facts'] if x['feature_id']==feature and x['state']==state]
            if len(same)>1:raise ValueError('Duplicate target fact in current view')
            if same:new_id=same[0]['fact_id']
            else:pp[k]['facts'].append(nf);new_id=nf['fact_id']
        touched.add(pid)
        transitions[pid,fid]=dict(action=action,new_fact_id=new_id,correction_id=a['correction_id'])
        audits.append(dict(a,derived_fact_id=new_id,changed=action in ('REMOVE','REPLACE','WITHHOLD'),
                           applied_to_corrected_view=action in ('REMOVE','REPLACE','WITHHOLD'),
                           original_F_modified=False,
                           semantic_unresolved=action in ('WITHHOLD','DEFER'),
                           category_provenance='inherited review-table category; not newly independently classified'))
    affected=[]; record_out=[]
    for pid in sorted(touched):
        k=(pid,);p=pp[k];r=copy.deepcopy(rr[k]);facts=p['facts']
        index(facts,['fact_id'])
        feature_states={}
        for f in facts:
            validate_fact(f,dictionary)
            if f['feature_id'] in feature_states and feature_states[f['feature_id']]!=f['state']:
                raise ValueError('Repair produced conflicting states in one encoded feature')
            feature_states[f['feature_id']]=f['state']
        oldtokens=sorted({f['feature_id']+'_@_'+f['state'] for f in old[k]['facts']})
        if oldtokens!=sorted(r['tokens']):raise ValueError('Frozen fact/token binding differs')
        tokens=sorted({f['feature_id']+'_@_'+f['state'] for f in facts})
        r.update(tokens=tokens,original_pattern_id=r['pattern_id'],corrected_pattern_id=digest(tokens),
                 fact_repair_status='CHANGED' if tokens!=oldtokens else 'UNCHANGED',empty_F=not tokens)
        # Keep original pattern_id for split/cluster/retrieval exclusion identity.
        p.update(original_F_sha256=digest(old[k]['facts']),corrected_F_sha256=digest(facts),
                 generated_from_original_F=True)
        affected.append(p);record_out.append(r)
    return affected,record_out,audits,transitions


def rebind(ids, pid, eid, mapping, transitions, allowed, decisions):
    """Returns a lower-bound confirmed set AND explicit unresolved state."""
    if ids is None:
        return dict(original_ids=None,confirmed_ids=None,pending=[],removed=[],available=False)
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate citation')
    result=set();pending=[];removed=[]
    for fid in ids:
        t=transitions.get((pid,fid))
        if t is None or t['action'] in ('KEEP','DEFER'):
            if fid not in allowed:raise ValueError('Unchanged citation missing from corrected F')
            result.add(fid)
        elif t['action'] in ('REMOVE','WITHHOLD'):removed.append(fid)
        else:
            decision=decisions.get((mapping,eid,fid))
            if decision is None or decision.get('decision')=='UNRESOLVED':
                pending.append(dict(old_fact_id=fid,new_fact_id=t['new_fact_id']))
                continue
            if decision.get('decision') not in ('CITES_NEW','DOES_NOT_CITE_NEW'):
                raise ValueError('Unknown replacement citation decision')
            if not decision.get('confirmed_by') or not decision.get('confirmed_date') or not decision.get('evidence_reference'):
                raise ValueError('Citation decision lacks confirmation/evidence')
            date.fromisoformat(decision['confirmed_date'])
            if decision.get('new_fact_id')!=t['new_fact_id']:
                raise ValueError('Citation decision target mismatch')
            if decision['decision']=='CITES_NEW':result.add(t['new_fact_id'])
            else:removed.append(fid)
    if not result<=allowed:raise ValueError('Corrected citation outside corrected F')
    return dict(original_ids=ids,confirmed_ids=sorted(result),pending=pending,removed=removed,available=not pending)
