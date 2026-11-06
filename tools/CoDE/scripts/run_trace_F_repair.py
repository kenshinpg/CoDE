"""Approved test-F repair using the frozen TRACE RQ1/RQ2/RQ3 workers.

Only affected admissions are scored. Original files and models are read-only.
Natural explanations retain their original text/diagnosis; E is projected onto
unchanged facts. State-replacement admissions also have an explicit pending-E
view. Controlled sets are regenerated under the original sampling protocol.
"""
import argparse, copy, csv, json, os, shutil, subprocess, sys, tarfile, time, traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from prepare_trace_F_repair import read,rows,dump,emit,table,sha
from code_eval.trace2400.f_repair import normalize_approved,validate_actions,patch_patients
from code_eval.trace2400.f_rematch import verify_rematches


def path(v):return Path(v) if Path(v).is_absolute() else ROOT/v


def config(a):
    c=read(a.config); c['datasets']['TRACE']={k:str(path(v)) for k,v in c['datasets']['TRACE'].items()}
    return c,c['datasets']['TRACE']


def verify(a):
    c,cfg=config(a); bindings={}
    lock=read(a.input/'input_lock.json')
    for n,h in lock.items():
        if sha(a.input/n)!=h:raise ValueError('Repair input changed: '+n)
        bindings[str(a.input/n)]=h
    for n,h in c['worker_code_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen worker changed: '+n)
    for key in ['rq2_scores','rq3_scores']:
        p=Path(cfg[key]);h=sha(p)
        if h!=c['accepted_sha256']['TRACE_'+key]:raise ValueError('Formal input changed: '+key)
        bindings[str(p)]=h
    v=Path(cfg['view']);m=read(v/'manifest.json')
    for n in ['patients_model_input_PRIVATE.jsonl','test_records_PRIVATE.jsonl','split_PRIVATE.jsonl',
              'evidence_dictionary.json','train_records_PRIVATE.jsonl','development_records_PRIVATE.jsonl']:
        p=v/n;h=sha(p)
        if m['output_sha256'].get(n)!=h:raise ValueError('Frozen view changed: '+n)
        bindings[str(p)]=h
    fm=read(Path(cfg['models'])/'fitted_sources.json')
    for n,h in c['accepted_TRACE_model_sha256'].items():
        p=Path(fm['models'][n]['path'])
        if fm['models'][n]['sha256']!=h or sha(p)!=h:raise ValueError('Frozen model changed: '+n)
        bindings[str(p)]=h
    for k in ['ledger','rq1_selections']:
        p=Path(cfg[k]);bindings[str(p)]=sha(p)
    return cfg,bindings


def prepare(a):
    cfg,bindings=verify(a);v=Path(cfg['view'])
    signed=rows(a.input/'source_fact_corrections_approved.csv')
    approved=normalize_approved(signed);proposals=rows(a.input/'source_fact_corrections_PRIVATE.csv')
    if len(approved)!=85 or validate_actions(proposals,approved):raise ValueError('Approval census')
    if Counter(r['approved_action'] for r in approved)!=Counter(REMOVE=81,REPLACE=2,WITHHOLD=2):raise ValueError('Signed action counts changed')
    patients=rows(v/'patients_model_input_PRIVATE.jsonl');test=rows(v/'test_records_PRIVATE.jsonl')
    patched,records,audit,transitions=patch_patients(patients,test,rows(v/'split_PRIVATE.jsonl'),
             read(v/'evidence_dictionary.json'),proposals,approved)
    changed={r['patient_id'] for r in records if r['fact_repair_status']=='CHANGED'}
    if len(records)!=28 or changed!={r['patient_id'] for r in records}:raise ValueError('Expected 28 changed test admissions')
    if any(r['empty_F'] for r in records):raise ValueError('Empty corrected F requires explicit protocol handling')
    by={p['patient_id']:p for p in patched};pending={str(r['admission_id']) for r in approved if r['approved_action']=='REPLACE'}
    old=[r for r in rows(Path(cfg['ledger'])) if r['patient_id'] in changed]
    if Counter(r['patient_id'] for r in old)!=Counter({p:4 for p in changed}):raise ValueError('Original rollout census')
    rematch_file=a.input/'rematch_review_PRIVATE.jsonl'
    rematch=[]
    if rematch_file.exists():
        rematch=verify_rematches(approved,rows(rematch_file),rows(a.input/'eight_original_explanations_PRIVATE.jsonl'),old,transitions)
        table(a.out/'replacement_E_rematch_acceptance_PRIVATE.csv',rematch)
    new=[];citation_tasks=[]
    for r in old:
        allowed={f['fact_id'] for f in by[r['patient_id']]['facts']};ids=r['cited_fact_ids']
        rr=dict(r,cited_fact_ids=None if ids is None else sorted(set(ids)&allowed),
            original_cited_fact_ids=ids,generated_from_original_F=True,
            replacement_reassessment_pending=r['patient_id'] in pending and not rematch,
            replacement_rematch_method='AI_full_text_review' if rematch and r['patient_id'] in pending else None)
        new.append(rr)
        if r['patient_id'] in pending:
            for act in approved:
                if act['approved_action']=='REPLACE' and str(act['admission_id'])==r['patient_id']:
                    citation_tasks.append(dict(explanation_id=r['explanation_id'],patient_id=r['patient_id'],
                        correction_id=act['correction_id'],old_fact_id=act['fact_id'],
                        new_fact_id=transitions[r['patient_id'],act['fact_id']]['new_fact_id'],
                        new_feature_id=act['replacement_feature_id'],new_state=act['replacement_state'],
                        status='AI_TARGET_REMATCH_COMPLETED' if rematch else 'PENDING_NEW_FACT_REFERENCE_REVIEW',
                        decision='DOES_NOT_CITE_NEW' if rematch else None,human_rematch_confirmed=False))
    emit(a.out/'corrected_patients_overlay_PRIVATE.jsonl',patched)
    emit(a.out/'corrected_test_overlay_PRIVATE.jsonl',records)
    emit(a.out/'original_ledger_PRIVATE.jsonl',old);emit(a.out/'projected_ledger_PRIVATE.jsonl',new)
    table(a.out/'F_repair_actions_PRIVATE.csv',audit);table(a.out/'replacement_E_review_PRIVATE.csv',citation_tasks)
    cls=Path(cfg['ledger']).parent/'class_shap.json';value=read(cls)
    if value['train_sha256']!=sha(v/'train_records_PRIVATE.jsonl') or value['tree_sha256']!=bindings[read(Path(cfg['models'])/'fitted_sources.json')['models']['lightgbm']['path']]:
        raise ValueError('Frozen class SHAP binding')
    if sha(cls)!=read(cls.parent/'evaluation_lock.json')['class_shap_sha256']:raise ValueError('Class SHAP identity')
    shutil.copyfile(cls,a.out/'class_shap.json')
    dump(a.out/'evaluation_lock.json',dict(class_shap_sha256=sha(cls)))
    bindings[str(cls)]=sha(cls)
    dump(a.out/'input_binding.json',dict(files=bindings,config_sha256=sha(a.config),
        approval_sha256=sha(a.input/'source_fact_corrections_approved.csv'),frozen_code=read(a.config)['worker_code_sha256']))
    dump(a.out/'repair_manifest.json',dict(admissions=28,original_test_admissions=len(test),actions=dict(Counter(r['approved_action'] for r in approved)),
        action_confirmation='R01 2026-10-18; APPROVED',semantic_unresolved_withheld=2,
        pending_replacement_explanations=0 if rematch else len(citation_tasks),pending_admissions=[] if rematch else sorted(pending),
        replacement_admissions=sorted(pending),AI_rematched_explanations=len(rematch),human_rematch_confirmed=False,
        changed_admissions=sorted(changed),models_trained=False,F_original_overwritten=False,
        natural_E_policy=('original citations projected onto surviving facts; eight new-fact references reviewed by AI as absent'
                          if rematch else 'original citations projected onto unchanged corrected facts; replacement citations await review'),
        rq3_policy='regenerate sets with original seed, budget and exact-count sampler; no old/new subset-ID pairing',
        source_explanations_generated_from_original_F=True))
    return cfg


def worker(a):
    import numpy as np
    import run_trace2400_evaluation as ev
    import run_trace2400_rq2 as natural
    _,cfg=config(a)
    args=SimpleNamespace(view=Path(cfg['view']),models=Path(cfg['models']),first=Path(cfg['first']),
        partition='test',threads=a.threads,out=a.out,shard=0,shards=1)
    bank,retrieval,labels,enc,means,allquery=ev.setup(args)
    corrected={r['patient_id']:r for r in rows(a.out/'corrected_test_overlay_PRIVATE.jsonl')}
    ids=sorted(corrected)[a.shard::a.shards]
    # Retrieval exclusion was constructed with ORIGINAL train+test identity.
    original_patients=rows(Path(cfg['view'])/'patients_model_input_PRIVATE.jsonl')
    patched={r['patient_id']:r for r in rows(a.out/'corrected_patients_overlay_PRIVATE.jsonl')}
    model_token_audit=[]
    for pid in ids:
        for name,(_,vocab) in bank.neural.items():
            model_token_audit.append(dict(patient_id=pid,source=name,
                original_oov=sorted(set(allquery[pid]['tokens'])-vocab.keys()),
                corrected_oov=sorted(set(corrected[pid]['tokens'])-vocab.keys())))
    folder=a.out/f'worker_{a.shard}';folder.mkdir(exist_ok=True)
    emit(folder/'model_token_audit_PRIVATE.jsonl',model_token_audit)
    if not ids:
        dump(folder/'complete.json',dict(patients=[],outputs={}));return
    originals=rows(a.out/'original_ledger_PRIVATE.jsonl'); projected=rows(a.out/'projected_ledger_PRIVATE.jsonl')
    native_rows=natural.rows
    for variant in ['original','corrected']:
        query={pid:dict(allquery[pid],**({'tokens':corrected[pid]['tokens']} if variant=='corrected' else {})) for pid in ids}
        # The return tuple uses frozen encodings/weights; only query tokens change.
        def setup(_):return bank,retrieval,labels,enc,means,query
        ev.setup=setup;natural.setup=setup
        patients=[patched.get(r['patient_id'],r) if variant=='corrected' else r for r in original_patients]
        natural.rows=lambda p:patients if Path(p).resolve()==(Path(cfg['view'])/'patients_model_input_PRIVATE.jsonl').resolve() else native_rows(p)
        for rq in ['rq1','rq2','rq3']:
            dest=folder/variant/rq;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'shard_0/run_manifest.json').exists():
                # Whole-run resume verifies this file and output hashes before workers start.
                if read(dest/'shard_0/run_manifest.json')['status']=='complete':continue
                raise ValueError('Incomplete prior worker folder; use a fresh output directory')
            args.out=dest
            print('START',a.shard,variant,rq,'patients',len(ids),flush=True)
            if rq=='rq2':
                ledger=[r for r in (originals if variant=='original' else projected) if r['patient_id'] in ids]
                emit(dest/'explanation_ledger.jsonl',ledger)
                dump(dest/'evaluation_lock.json',dict(explanation_ledger_sha256=sha(dest/'explanation_ledger.jsonl')))
                natural.worker(args)
            elif rq=='rq1':ev.worker_rq1(args)
            else:ev.worker_rq3(args)
            print('DONE',a.shard,variant,rq,flush=True)
    dump(folder/'complete.json',dict(patients=ids,outputs={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='complete.json'}))


def launch(a):
    ids=a.gpus.split(',');jobs=[]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate GPU')
    try:
        for i,gpu in enumerate(ids):
            cmd=[sys.executable,'-u',__file__,'--config',str(a.config),'--input',str(a.input),'--out',str(a.out),
                 '--shard',str(i),'--shards',str(len(ids)),'--threads',str(a.threads)]
            log=(a.out/f'worker_{i}.log').open('a',encoding='utf-8')
            jobs.append((subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,
                CODE_TRACE_TABR_QUERY_MODE='single',OMP_NUM_THREADS=str(a.threads)),stdout=log,stderr=subprocess.STDOUT),log))
        while any(p.poll() is None for p,_ in jobs):
            print('F repair workers',[(i,p.poll()) for i,(p,_) in enumerate(jobs)],flush=True);time.sleep(20)
        if any(p.returncode for p,_ in jobs):raise RuntimeError('Worker failed; controlled report contains logs')
    finally:
        for p,l in jobs:
            if p.poll() is None:p.terminate();p.wait()
            l.close()


def run(a):
    import torch,lightgbm,numpy
    if not torch.cuda.is_available():raise RuntimeError('Use the existing C550 vendor torch environment')
    if not a.resume:
        a.out.mkdir(parents=True,exist_ok=False)
        print('Stage 1: approved actions, frozen inputs and separate corrected-F preparation',flush=True)
        prepare(a)
    else:
        current=read(a.out/'input_binding.json')
        if current['config_sha256']!=sha(a.config):raise ValueError('Resume config changed')
        verify(a)
        for p,h in current['files'].items():
            if sha(Path(p))!=h:raise ValueError('Resume input changed')
        previous=read(a.out/'run_manifest.json')
        for p,h in previous.get('code_sha256',{}).items():
            if sha(ROOT/p)!=h:raise ValueError('Resume implementation changed: '+p)
        for p,h in previous.get('output_sha256',{}).items():
            if sha(a.out/p)!=h:raise ValueError('Resume output changed: '+p)
    print('Stage 2: original replay and corrected scoring for 28 affected admissions',flush=True)
    launch(a)
    print('Scoring complete; replay audit and paired summaries',flush=True)
    from summarize_trace_F_repair import summarize
    result=summarize(a)
    for p,h in read(a.out/'input_binding.json')['files'].items():
        if sha(Path(p))!=h:raise ValueError('Frozen original unexpectedly changed')
    return dict(result,models_trained=False,api_calls=0,original_files_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/human_return_20261016.json')
    p.add_argument('--input',type=Path,default=ROOT/'inputs/source_F_repair_joint_20261018')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/trace_F_repair_v2')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--shard',type=int);p.add_argument('--shards',type=int);p.add_argument('--resume',action='store_true')
    p.add_argument('--bootstrap',type=int,default=5000)
    a=p.parse_args()
    for k in ['config','input','out']:setattr(a,k,getattr(a,k).resolve())
    if a.shard is not None:worker(a)
    else:
        state={}
        try:state=run(a)
        except Exception:state=dict(status='failed',error=traceback.format_exc());raise
        finally:
            if a.out.exists():
                state['output_sha256']={str(f.relative_to(a.out)):sha(f) for f in a.out.rglob('*') if f.is_file() and f.name!='run_manifest.json'}
                state['code_sha256']={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'scripts/summarize_trace_F_repair.py',ROOT/'src/code_eval/trace2400/f_repair.py',ROOT/'src/code_eval/trace2400/f_rematch.py']}
                dump(a.out/'run_manifest.json',state)
                archive=a.out.with_name(a.out.name+'_reports_PRIVATE.tar.gz')
                with tarfile.open(archive,'w:gz') as t:
                    for f in a.out.rglob('*'):
                        if f.is_file():t.add(f,arcname=str(f.relative_to(a.out)))
                print('Return controlled report:',archive,flush=True)
