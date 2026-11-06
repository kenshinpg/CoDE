"""Lossless parent/atom linkage. No model calls, semantic rules, or clinical merging."""
import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter

def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
def dump(f,r):f.write(json.dumps(r,ensure_ascii=False)+'\n')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def identity(a,b):
    return all(a[k]==b[k] for k in ('code_sha256','system','schema','batch','input_sha256'))
def reuse(a):
    old,new=read(a.pilot/'manifest.json'),read(a.full/'manifest.json')
    if not identity(old,new):raise ValueError('Source/prompt/schema/code/batch changed; cannot reuse')
    for root,m in ((a.pilot,old),(a.full,new)):
        if sha(root/'tasks_PRIVATE.jsonl')!=m['tasks_sha256']:raise ValueError('Task hash mismatch')
    before={r['id']:r for r in rows(a.pilot/'tasks_PRIVATE.jsonl')}
    after={r['id']:r for r in rows(a.full/'tasks_PRIVATE.jsonl')}
    latest={r['id']:r for r in rows(a.pilot/'results_PRIVATE.jsonl')}
    selected=[]
    for k,r in latest.items():
        if r['status']!='COMPLETE':continue
        if k not in before or before[k]!=after.get(k):raise ValueError('Non-identical task '+k)
        selected.append(r)
    with (a.full/'results_PRIVATE.jsonl').open('x',encoding='utf-8') as f:
        for r in selected:dump(f,r)
    write(a.full/'reuse_manifest.json',dict(pilot=str(a.pilot.resolve()),pilot_manifest_sha256=sha(a.pilot/'manifest.json'),pilot_results_sha256=sha(a.pilot/'results_PRIVATE.jsonl'),reused_complete_tasks=len(selected),note='Only identical COMPLETE tasks reused. Failed tasks remain pending.'))
    print('Reused COMPLETE tasks:',len(selected),flush=True)

def build(a):
    manifest=read(a.run/'manifest.json')
    if sha(a.run/'tasks_PRIVATE.jsonl')!=manifest['tasks_sha256']:raise ValueError('Task hash mismatch')
    # Reuse original strict validation without modifying or rerunning extraction.
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_atoms',a.atom_code)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    if sha(a.atom_code)!=manifest['code_sha256']:raise ValueError('Wrong frozen atom code')
    tasks=list(rows(a.run/'tasks_PRIVATE.jsonl'))
    latest={r['id']:r for r in rows(a.run/'results_PRIVATE.jsonl')} if (a.run/'results_PRIVATE.jsonl').exists() else {}
    if set(latest)-{t['id'] for t in tasks}:raise ValueError('Foreign task IDs')
    docs={r['document_id']:r for r in rows(a.run/'documents_PRIVATE.jsonl')}
    a.out.mkdir(parents=True,exist_ok=False);counts=Counter();ids=set();parents=set();sources=Counter()
    run='bridge_'+str(time.time_ns())
    with (a.out/'mapping_units_PRIVATE.jsonl').open('x',encoding='utf-8') as uf,(a.out/'parent_ledger_PRIVATE.jsonl').open('x',encoding='utf-8') as pf:
        def emit(unit):
            uid=unit['unit_id']
            if uid in ids:raise ValueError('Duplicate unit '+uid)
            ids.add(uid);e=unit['entity'];text=docs[unit['document_id']]['text']
            s,t=e['raw_start'],e['raw_end']
            if not 0<=s<t<=len(text) or text[s:t]!=e['raw_quote']:raise ValueError('Span mismatch '+uid)
            dump(uf,dict(schema_version='radiology_mapping_units_v1',run_id=run,**unit));counts[unit['unit_kind']]+=1
        for task in tasks:
            response=latest.get(task['id']);items={}
            if response and response['status']=='COMPLETE':items={i['parent_id']:i for i in mod.validate(response['parsed'],task)['items']}
            for parent in task['parents']:
                pid=parent['parent_id']
                if pid in parents:raise ValueError('Duplicate parent '+pid)
                parents.add(pid);item=items.get(pid);disposition=item['status'] if item else 'unresolved_task';sources[disposition]+=1
                base=dict(document_id=task['document_id'],patient_id=task['patient_id'],parent_mention_id=pid,representation_group_id=pid,context=parent['context'],section=parent['section'],original_entity=parent['original_entity'],assertion_correction_applied=False,task_id=task['id'],disposition_is_filter=False,clinical_evidence_group_frozen=False)
                parent_uid=pid+':parent'
                emit(dict(**base,unit_id=parent_uid,unit_kind='parent',entity=parent['original_entity'],retrieval_queries=[dict(text=parent['quote'],source='original_quote')],proposed_assertion=None,proposed_role=None,derived_from_unit_ids=[],decomposition_status=disposition))
                children=[]
                for j,data in enumerate(item['atoms'] if item else []):
                    atom=mod.Atom.model_validate(data);start=parent['start']+mod.locate(parent['quote'],atom);uid=pid+':atom'+str(j)
                    queries=[dict(text=atom.quote,source='original_subspan')]
                    if atom.query!=atom.quote:queries.append(dict(text=atom.query,source='model_contextual_query_unverified'))
                    emit(dict(**base,unit_id=uid,unit_kind='atom',entity=dict(mention_id=uid,raw_start=start,raw_end=start+len(atom.quote),raw_quote=atom.quote,assertion=parent['original_entity'].get('assertion')),retrieval_queries=queries,proposed_assertion=atom.assertion,proposed_role=atom.role,derived_from_unit_ids=[parent_uid],decomposition_status=disposition))
                    children.append(uid)
                dump(pf,dict(parent_id=pid,parent_unit_id=parent_uid,child_unit_ids=children,document_id=task['document_id'],decomposition_status=disposition,source_retained=True,mapping_eligible=True))
    if len(parents)!=manifest['parents']:raise ValueError('Incomplete parent coverage')
    report=dict(run_id=run,source_run=str(a.run.resolve()),counts=dict(counts),parent_dispositions=dict(sources),expected_parents=manifest['parents'],retained_parents=len(parents),parent_coverage=1.0,all_spans_checked=True,documents=len(docs),zero_entity_documents=sum(d['entity_count']==0 for d in docs.values()),input_hashes={n:sha(a.run/n) for n in ('manifest.json','tasks_PRIVATE.jsonl','documents_PRIVATE.jsonl')},code_sha256=sha(__file__),semantic_acceptance=False,hpo_mapping_performed=False,note='All original parents retained. Atom queries augment candidate retrieval only. Parent/atom units are alternative or supplementary representations, not independent scoring facts. Multi-concept mappings must retain support spans; count after CoDE view rules.')
    if (a.run/'results_PRIVATE.jsonl').exists():report['input_hashes']['results_PRIVATE.jsonl']=sha(a.run/'results_PRIVATE.jsonl')
    write(a.out/'report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    q=s.add_parser('reuse');q.add_argument('--pilot',type=Path,required=True);q.add_argument('--full',type=Path,required=True)
    q=s.add_parser('build');q.add_argument('--run',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--atom-code',type=Path,default=Path('scripts/radiology_atoms_v1/atoms.py'))
    a=p.parse_args();globals()[a.cmd](a)
if __name__=='__main__':main()
