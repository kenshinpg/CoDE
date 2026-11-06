"""202 explicit human sets and source-F exposure; frozen scoring workers reused."""
import argparse,os,sys,time,traceback,subprocess,shutil,tarfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from code_eval.human_return import read,rows,sha,dump,emit,table,validate_reviews,make_ledger,cook,summarize,complete,score_drift_audit

def verified_copy(source,dest,relative,lock):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe saved relative path')
    src=source/rel
    expected=lock.get('output_sha256',{}).get(rel.as_posix())
    if expected is None or sha(src)!=expected:raise ValueError('Saved output identity changed: '+rel.as_posix())
    target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)

def reuse_source_audit(a,provenance):
    source=a.resume_from
    if read(source/'input_binding.json')!=provenance:raise ValueError('Resume input identity changed')
    lock=read(source/'run_manifest.json')
    names=['source_F_impact.csv','source_F_population_impact.csv']
    names+=sorted(k for k in lock.get('output_sha256',{}) if k.startswith('source_F/'))
    required={'source_F/source_flags_PRIVATE.json','source_F/summary.json'}
    if not required<=set(names):raise ValueError('Saved source-F audit incomplete')
    for name in names:verified_copy(source,a.out,name,lock)
    if read(a.out/'source_F/summary.json')['status']!='complete':raise ValueError('Saved source-F audit has missing records')
    dump(a.out/'resume_provenance.json',dict(source=str(source),source_manifest_sha256=sha(source/'run_manifest.json'),
        source_audit_files_reused=names,dataset_shards_reused=[]))
    return read(a.out/'source_F/source_flags_PRIVATE.json')

def reuse_dataset_shards(a,ds):
    source=a.resume_from
    if source is None or not (source/ds).exists():return False
    lock=read(source/'run_manifest.json');folder=source/ds
    for name in ['explanation_ledger.jsonl','class_shap.json','evaluation_lock.json']:
        rel=ds+'/'+name
        if lock['output_sha256'].get(rel)!=sha(folder/name) or sha(folder/name)!=sha(a.out/rel):
            raise ValueError('Resume prepared dataset changed: '+rel)
    shards=sorted(folder.glob('shard_*'))
    if not shards:raise ValueError('Existing dataset has no completed shards; use a new run for that dataset')
    for shard in shards:
        if read(shard/'run_manifest.json')['status']!='complete':raise ValueError('Saved worker incomplete: '+str(shard))
        prefix=ds+'/'+shard.name+'/'
        files=sorted(k for k in lock['output_sha256'] if k.startswith(prefix))
        if prefix+'scores.jsonl' not in files:raise ValueError('Saved score shard is not hash-bound')
        for name in files:verified_copy(source,a.out,name,lock)
        shutil.copyfile(shard/'run_manifest.json',a.out/ds/shard.name/'run_manifest.json')
    note=read(a.out/'resume_provenance.json');note['dataset_shards_reused'].append(dict(dataset=ds,shards=len(shards)))
    dump(a.out/'resume_provenance.json',note)
    print('Reused verified',ds,'score shards:',len(shards),'; zero scoring calls for this dataset',flush=True)
    return True

def path(v):return Path(v) if Path(v).is_absolute() else ROOT/v
def configs(p):
    c=read(p)
    for d,values in c['datasets'].items():c['datasets'][d]={k:str(path(v)) for k,v in values.items()}
    return c

def verify_files(a,c):
    lock=read(a.review/'input_lock.json');seen={}
    for f,h in lock['files'].items():
        if sha(a.review/f)!=h:raise ValueError('Human input changed: '+f)
    for ds,cfg in c['datasets'].items():
        for k in ['rq2_scores','rq3_scores'] if ds=='TRACE' else ['rq2_scores']:
            p=Path(cfg[k]);h=sha(p)
            if h!=c['accepted_sha256'][ds+'_'+k]:raise ValueError('Formal score input drift: '+ds+' '+k)
            seen[str(p)]=h
        if ds=='TRACE':
            fm=read(Path(cfg['models'])/'fitted_sources.json')
            for name,h in c['accepted_TRACE_model_sha256'].items():
                if fm['models'][name]['sha256']!=h or sha(Path(fm['models'][name]['path']))!=h:raise ValueError('TRACE fitted model identity changed: '+name)
    for f,h in c['worker_code_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('Frozen worker implementation changed: '+f)
    return dict(human_input_lock=lock,formal_score_files=seen,worker_code=c['worker_code_sha256'])

def dataset_args(a,cfg,ds,out):
    common=dict(out=out,threads=a.threads,shard=a.shard,shards=a.shards,partition='test')
    if ds=='DDXPlus':common.update({k:Path(cfg[k]) for k in ['models','protocol','sample','rq1','config']})
    else:common.update({k:Path(cfg[k]) for k in ['view','models','first']})
    return SimpleNamespace(**common)

def prepare_dataset(a,c,ds,reviews):
    cfg=c['datasets'][ds];out=a.out/ds;out.mkdir(exist_ok=False)
    original={r['explanation_id']:r for r in rows(Path(cfg['ledger']))}
    rr=[r for r in reviews if r['dataset']==ds]
    if ds=='DDXPlus':
        from run_code10k_rq1 import verify
        verify(dataset_args(a,cfg,ds,out))
        pp=rows(Path(cfg['sample'])/'data/patients_public.jsonl')
        facts={r['patient_id']:{f['fact_id'] for f in r['facts'] if f.get('visible',True) and f['coverage_scope']=='symptoms_antecedents'} for r in pp}
        cls=Path(cfg['rq1'])/'class_shap.json'
    else:
        from run_trace2400_evaluation import inputs
        inputs(dataset_args(a,cfg,ds,out))
        pp=rows(Path(cfg['view'])/'patients_model_input_PRIVATE.jsonl');facts={r['patient_id']:{f['fact_id'] for f in r['facts']} for r in pp}
        cls=Path(cfg['ledger']).parent/'class_shap.json'
    ledger,lookup=make_ledger(rr,original,facts)
    emit(out/'explanation_ledger.jsonl',ledger);shutil.copyfile(cls,out/'class_shap.json')
    dump(out/'evaluation_lock.json',dict(explanation_ledger_sha256=sha(out/'explanation_ledger.jsonl'),class_shap_sha256=sha(out/'class_shap.json'),
        baseline_ledger_sha256=sha(Path(cfg['ledger'])),human_input_sha256=sha(a.review/'human_return_evidence_sets_PRIVATE.jsonl')))
    return lookup

def launch(a,ds):
    ids=a.gpus.split(',');jobs=[]
    try:
        for i,g in enumerate(ids):
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--dataset',ds,'--shard',str(i),'--shards',str(len(ids)),
                '--out',str(a.out),'--review',str(a.review),'--config',str(a.config),'--threads',str(a.threads)]
            log=(a.out/ds/f'worker_{i}.log').open('w',encoding='utf-8')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=g,CODE_TRACE_TABR_QUERY_MODE='single',OMP_NUM_THREADS=str(a.threads))
            jobs.append((subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
        while any(p.poll() is None for p,l in jobs):
            print(ds,'review scoring',[(i,p.poll()) for i,(p,l) in enumerate(jobs)],flush=True);time.sleep(10)
        if any(p.returncode for p,l in jobs):raise RuntimeError(ds+' worker failed; inspect worker logs')
    finally:
        for p,l in jobs:
            if p.poll() is None:p.terminate();p.wait()
            l.close()

def finish_dataset(a,c,ds,lookup,flags):
    cfg=c['datasets'][ds];out=a.out/ds;scored=[]
    for f in sorted(out.glob('shard_*')):
        if read(f/'run_manifest.json')['status']!='complete':raise ValueError('Incomplete score shard')
        scored+=rows(f/'scores.jsonl')
    old={(r['explanation_id'],r['probe']):r for r in rows(Path(cfg['rq2_scores']))}
    labels=read(out/'class_shap.json')['labels']
    audit=score_drift_audit(scored,lookup,old)
    dump(out/'automatic_score_drift_PRIVATE.json',audit)
    table(out/'automatic_score_drift_summary.csv',[dict(method=m,**v) for m,v in audit['methods'].items()])
    quarantine=set()
    if a.quarantine_voting_drift and ds=='DDXPlus' and any(m.startswith('voting_') for m in audit['mismatching_methods']):
        quarantine={'voting_coverage','voting_absolute','voting_retained'}
        print('DDXPlus Voting replay drift: all three Voting scores withheld from BOTH mapping versions; raw values retained.',flush=True)
    cooked,check=cook(scored,lookup,old,labels,flags if ds=='TRACE' else None,quarantined_methods=quarantine)
    dump(out/'baseline_check.json',check);emit(out/'case_results_PRIVATE.jsonl',cooked)
    return cooked

def annotate_human_impact(a,records):
    import csv
    p=a.out/'source_F_impact.csv'
    with p.open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
    originals={(r['dataset'],r['explanation_id'],r['probe']):r for r in records if r['version']=='automatic'}
    for x in rr:
        fid=x['fact_id'];pid=x['patient_id']
        if not fid:continue
        for probe in ['encoder','lightgbm']:
            target=[r for r in records if r['dataset']=='TRACE' and r['version']=='human_exact' and r['probe']==probe and r['patient_id']==pid]
            for name,methods in [('quantity',['fact_coverage','group_coverage']),('cohd_vs_group',['cohd_coverage','group_coverage']),('shap_vs_group',['tree_positive_shap','group_coverage']),('shap_vs_A',['tree_positive_shap','lightgbm_absolute'])]:
                chosen=[r for r in target if r['both_full_correct'] and r['diagnosis_correct'] and r['probe_available'] and (name=='quantity' or r['target_supported']) and all(r['scores'].get(m) is not None for m in methods)]
                hit=[r for r in chosen if fid in r['fact_ids']]
                x[probe+'_'+name+'_human_EXACT_explanations']=len(hit)
                x[probe+'_'+name+'_human_EXACT_updated_failures']=sum(r['reference_failure'] for r in hit)
                common=[r for r in hit if originals[r['dataset'],r['explanation_id'],r['probe']]['probe_available'] and all(originals[r['dataset'],r['explanation_id'],r['probe']]['scores'].get(m) is not None for m in methods)]
                x[probe+'_'+name+'_human_EXACT_common_explanations']=len(common)
                x[probe+'_'+name+'_human_EXACT_common_original_failures']=sum(originals[r['dataset'],r['explanation_id'],r['probe']]['reference_failure'] for r in common)
    table(p,rr)

def run(a):
    c=configs(a.config);reviews=rows(a.review/'human_return_evidence_sets_PRIVATE.jsonl');census=validate_reviews(reviews)
    if census!={'DDXPlus':{'explanations':102,'complete':91,'random':50,'targeted':53,'overlap':1},'TRACE':{'explanations':100,'complete':25,'random':60,'targeted':40,'overlap':0}}:raise ValueError('Human return census changed')
    provenance=verify_files(a,c)
    if a.summary_only:
        previous=read(a.out/'input_binding.json');runlock=read(a.out/'run_manifest.json')
        if previous!=provenance:raise ValueError('Summary input identity changed')
        for ds in ['DDXPlus','TRACE']:
            f=a.out/ds/'case_results_PRIVATE.jsonl'
            if runlock.get('output_sha256',{}).get(f.relative_to(a.out).as_posix())!=sha(f):raise ValueError('Saved case records not verified')
    else:
        import torch,lightgbm
        if not torch.cuda.is_available():raise RuntimeError('Use the existing C550 torch environment')
        if torch.cuda.device_count()<len(a.gpus.split(',')):raise RuntimeError('Requested GPUs are not visible')
    dump(a.out/'input_binding.json',provenance);dump(a.out/'review_census.json',census)
    if a.resume_from:
        print('Stage 1: reuse hash-verified completed source-F audit',flush=True)
        flags=reuse_source_audit(a,provenance)
    else:
        from audit_human_source_F import run as source_run
        print('Stage 1: frozen source-F exposure and candidate eligibility audit',flush=True)
        flags=source_run(a,c['datasets']['TRACE'],reviews)
    allrecords=[]
    for ds in ['DDXPlus','TRACE']:
        print('Stage 2:',ds,'automatic and human EXACT subset scoring',flush=True)
        if a.summary_only:
            allrecords+=rows(a.out/ds/'case_results_PRIVATE.jsonl')
            continue
        lookup=prepare_dataset(a,c,ds,reviews)
        if not reuse_dataset_shards(a,ds):launch(a,ds)
        allrecords+=finish_dataset(a,c,ds,lookup,flags)
    rule_meta=read(a.review/'human_rule_metadata.json')
    for r in allrecords:r['reported_rule_change_types']=rule_meta.get(r['dataset']+'|'+r['explanation_id'],[])
    emit(a.out/'human_review_case_results_PRIVATE.jsonl',allrecords);annotate_human_impact(a,allrecords)
    print('Stage 3: common-sample fixed/updated-endpoint sensitivity',flush=True)
    stats,trans=summarize(allrecords,a.bootstrap,261006)
    table(a.out/'human_review_subset_sensitivity.csv',stats);table(a.out/'endpoint_transitions.csv',trans)
    # Aggregate at exactly the same sample intersections as each reported comparison.
    meaningful=[r for r in stats if r['analysis_scope']=='formal_RQ2_scope' and r['proposal_stratum']=='correct' and r['probe']=='encoder'
        and (r['method'],r['reference_method']) in [('cohd_coverage','group_coverage'),('tree_positive_shap','group_coverage'),('tree_positive_shap','lightgbm_absolute')]]
    table(a.out/'key_sensitivity_comparisons.csv',meaningful)
    readme='''# Human explicit-citation-set sensitivity

The input contains 202 reviewed explanations (DDXPlus 102; TRACE 100). The human EXACT sets are used. All adjudicated_E fields remain null and are excluded from the analysis.

Random and targeted review memberships are retained separately; the overlapping DDXPlus explanation is stored once per version/probe and contributes once to each of its two review strata. COMPLETE requires the reported COMPLETE status, no unresolved fact IDs, and no incomplete flag. All-reviewed results use explicitly confirmed citations even when the remainder is unresolved.

Models, training splits, original automatic mappings and full patient facts F remain frozen. Each subset's automatic predictions and scores are checked against accepted formal scores before results are summarized. TRACE uses the accepted single-query TabR inference mode. No new model fitting, LLM generation, A/B re-analysis, or full-test main-table replacement is performed.

The baseline_check.json files state any score quarantine. With the explicit Voting-quarantine option, an observed DDXPlus Voting replay mismatch withholds its entire score family from BOTH automatic and human mappings, across the whole reviewed DDXPlus sample. Those entries remain unavailable in sensitivity comparisons. Raw recomputed values are preserved in quarantined_raw_scores; original-versus-replayed discrepancies are recorded in automatic_score_drift_PRIVATE.json. This is an unresolved replay failure of a comparison control, not evidence of annotation sensitivity or permission to alter its formal results. All other scores and endpoint checks retain their original strict thresholds. Similar-case TabR and the matched no-retrieval control remain separately checked.

## Tables

- source_F_impact.csv: controlled source-item IDs, exact membership in F/splits and formal RQ2, human EXACT exposure, and same-feature audit candidates. Same-feature counts do not identify further confirmed errors.
- source_F_population_impact.csv: unique affected records/patients/failure events in each original RQ1/RQ2/RQ3 common population. Full-F exposure includes the denominator, attribution and full-input eligibility even when the erroneous fact is absent from E.
- source_F/source_omission_candidates_PRIVATE.jsonl: 35 bundled omission/view-exclusion candidates. Source-span context matching is recorded; it does not establish mention identity or eligibility. None are added to F.
- human_review_subset_sensitivity.csv: each method's AP and selected pairwise method contrasts on common automatic/human records; intervals and availability counts. key_sensitivity_comparisons.csv extracts the main three comparisons.
- endpoint_transitions.csv: prediction/endpoint changes, failure patient counts, set changes and per-method unscorable counts.
- human_review_case_results_PRIVATE.jsonl: original IDs, both sets, changes, scores, class probabilities, endpoints and source-review flags.

## Interpretation

review_scope describes the fixed reviewed sample; formal_RQ2_scope applies the unchanged full-input dual-probe correctness filter. Correct, incorrect and all proposals are separate. Score comparisons use common availability for both methods and mapping versions. Targeted review estimates describe the selected failure sample.

fixed_original evaluates both sets of scores against the original automatic-E failure labels. updated_human evaluates each mapping against its own probe endpoints; its difference combines score and endpoint changes. The two policies must be interpreted separately. Failure AP uses lower support to predict reference-recovery failure. One-class samples are excluded from AP discrimination comparisons (all-positive AP would be a constant 1). Unadjusted percentile intervals use the original cluster units (DDXPlus pattern_id; TRACE inferred subject_group), seed 261006 and the recorded number of paired bootstrap draws. Excluded one-class draws are counted; intervals based on remaining draws are conditional. This explicit estimability rule applies to this review subset only.

TRACE results are conditional on existing F. Known source-F errors and unresolved source facts remain present; changing E alone cannot repair them. An absence from E does not eliminate an effect through full-input support or feature weights. Raw review errors are not extrapolated to an overall error rate.

## Manuscript mapping

These tables supplement RQ2 natural-explanation measurement reliability and the mapping-review appendix (tab:mapping-flips, tab:ddx-audit-encoder, tab:ddx-audit-lightgbm, tab:trace-audit-encoder, tab:trace-audit-lightgbm). Source-F exposure concerns the TRACE data-processing appendix, tab:trace-flow and the interpretation of RQ1–RQ3. They do not replace formal RQ1–RQ3 main tables. Cross-sentence rule changes and ambiguity remain in the source review metadata; human EXACT sensitivity is not a final adjudication or a pure unchanged-rule correction estimate.

Reviewers manually checked AI-assisted suggestions according to the supplied handoff. Reviewer identifiers were not supplied in this package. No new independent clinical adjudication is claimed. The entire returned package is controlled because source_F_impact.csv and case files retain original IDs.
'''
    findings=['\n## Computed event summary\n']
    for r in trans:
        if r['row_type']=='probe' and r['analysis_scope']=='formal_RQ2_scope' and r['proposal_stratum']=='correct' and r['probe']=='encoder':
            findings.append(f"- {r['dataset']} / {r['audit_group']} / {r['human_subset']}: {r['common_explanations']} common explanations in {r['common_patients']} patients/admissions; failures {r['automatic_failures']} -> {r['human_failures']}; success-to-failure {r['success_to_failure']}, failure-to-success {r['failure_to_success']}; {r['source_F_error_patients']} reviewed patients/admissions have confirmed source-F errors.")
    (a.out/'README_results.md').write_text(readme+'\n'.join(findings)+'\n',encoding='utf-8')
    dump(a.out/'source_F_repair_scope.json',dict(current_F_unchanged=True,
        if_only_confirmed_test_facts_corrected='Keep fitted weights and split; rebuild affected F/E bindings, recompute affected test full-input and subset predictions, attribution/coverage, eligibility and all affected RQ tables/intervals. Regeneration is a separate decision if citing changed facts.',
        if_shared_rules_change_training_or_development='Audit rule scope first. Any actual changes to fitted/selection inputs require a new version, dictionary/train statistics/class SHAP and affected fitted sources or model selection, then full dependent evaluation. Do not refit on reviewed test labels.',
        current_sensitivity='conditional on existing F, not a corrected-F result'))
    withheld={ds:read(a.out/ds/'baseline_check.json').get('quarantined_methods',[]) for ds in ['DDXPlus','TRACE']}
    return dict(status='complete_with_quarantined_control' if any(withheld.values()) else 'complete',quarantined_methods=withheld,
        review_census=census,case_rows=len(allrecords),sensitivity_rows=len(stats),models_trained=False,F_modified=False,main_tables_modified=False,bootstrap=a.bootstrap)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/human_return_20261016.json')
    p.add_argument('--review',type=Path,default=ROOT/'inputs/human_review_20261016')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/human_return_v1')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--threads',type=int,default=4);p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--dataset',choices=['DDXPlus','TRACE']);p.add_argument('--shard',type=int);p.add_argument('--shards',type=int)
    p.add_argument('--summary-only',action='store_true')
    p.add_argument('--resume-from',type=Path,help='Reuse hash-bound source-F audit and completed score shards into a NEW output directory')
    p.add_argument('--quarantine-voting-drift',action='store_true',help='Withhold all DDXPlus Voting scores in both mappings if its baseline replay drifts; other checks remain strict')
    a=p.parse_args()
    a.out=a.out.resolve();a.config=a.config.resolve();a.review=a.review.resolve()
    if a.resume_from:
        a.resume_from=a.resume_from.resolve()
        if a.resume_from==a.out or a.summary_only:raise ValueError('Resume requires a distinct new output directory and cannot be combined with summary-only')
    if a.shard is not None:
        c=configs(a.config);w=dataset_args(a,c['datasets'][a.dataset],a.dataset,a.out/a.dataset)
        if a.dataset=='TRACE':
            os.environ['CODE_TRACE_TABR_QUERY_MODE']='single'
            from run_trace2400_rq2 import worker
        else:from run_code10k_rq2 import worker
        worker(w)
    else:
        if len(set(a.gpus.split(',')))!=len(a.gpus.split(',')):raise ValueError('Repeated GPU')
        a.out.mkdir(parents=True,exist_ok=a.summary_only);state={};started=time.time()
        try:state=run(a)
        except Exception:state=dict(status='failed',error=traceback.format_exc());raise
        finally:
            state.update(seconds=time.time()-started,output_sha256={f.relative_to(a.out).as_posix():sha(f) for f in a.out.rglob('*') if f.is_file() and f.name!='run_manifest.json'},
                analysis_code_sha256={n:sha(ROOT/n) for n in ['scripts/run_human_return_sensitivity.py','scripts/audit_human_source_F.py','src/code_eval/human_return.py']});dump(a.out/'run_manifest.json',state)
            archive=Path(str(a.out)+'_reports_PRIVATE.tar.gz')
            with tarfile.open(archive,'w:gz') as t:
                for f in a.out.rglob('*'):
                    if f.is_file() and not any(s.startswith('shard_') for s in f.relative_to(a.out).parts):t.add(f,arcname=f.relative_to(a.out).as_posix())
            print('Return controlled report:',archive,flush=True)
