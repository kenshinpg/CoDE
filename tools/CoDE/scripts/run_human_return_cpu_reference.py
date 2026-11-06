"""Complete review sensitivity with a separately reported CPU numerical reference.

The accepted formal automatic scores are retained and compared separately.
This is not a replacement for formal tables or a relaxed automatic replay gate.
"""
import argparse,sys,os,time,traceback,subprocess,tarfile,shutil,copy
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from code_eval.human_return import read,rows,sha,dump,emit,table,cook,score_drift_audit,summarize
from code_eval.trace2400.cpu_reference import SOURCES,METHODS,POLICY,TOL,coverage_reference,probability_delta
from run_human_return_sensitivity import configs,verify_files,annotate_human_impact
from diagnose_human_return_trace import verified_saved

def worker(a,c):
    import numpy as np,torch
    from run_trace2400_evaluation import inputs
    from code_eval.trace2400.cpu_reference import CPUReference
    cfg=c['datasets']['TRACE'];arg=SimpleNamespace(view=Path(cfg['view']),models=Path(cfg['models']),partition='test')
    train,labels,enc,paths,split,query=inputs(arg)
    model=CPUReference(paths,read(Path(cfg['view'])/'evidence_dictionary.json'),labels,train,list(query.values()),{p:s['subject_group'] for p,s in split.items()})
    patients={r['patient_id']:r for r in rows(Path(cfg['view'])/'patients_model_input_PRIVATE.jsonl')}
    saved=rows(a.out/'TRACE_input_scores_PRIVATE.jsonl');ledger=rows(a.source/'TRACE/explanation_ledger.jsonl')
    by={}
    for e in ledger:by.setdefault(e['patient_id'],[]).append(e)
    ids=sorted(by)[a.shard::a.workers];result=[];weights=[];gate=dict(probability_max_delta=0.,coverage_max_delta=0.,cache_max_delta=0.,availability_changes=0,patients=0,queries=0,score_checks=0)
    for pos,pid in enumerate(ids,1):
        r=query[pid];full=tuple(sorted(r['tokens']));groups=enc.groups(r['tokens']);minus={g:tuple(sorted(set(full)-set(ts))) for g,ts in groups.items()}
        facts={f['fact_id']:f['feature_id']+'_@_'+f['state'] for f in patients[pid]['facts']}
        if set(full)!=set(facts.values()):raise ValueError('Frozen F/token mismatch')
        cited={e['explanation_id']:tuple(sorted({facts[f] for f in e['cited_fact_ids']})) for e in by[pid]}
        keys=sorted({full,*minus.values(),*cited.values()})
        first={k:model.predict(list(k),pid) for k in keys}
        second={k:model.predict(list(k),pid) for k in reversed(keys)}
        for k in keys:
            gate['probability_max_delta']=max(gate['probability_max_delta'],max(probability_delta(first[k][m],second[k][m]) for m in SOURCES))
        gate['cache_max_delta']=max(gate['cache_max_delta'],model.check_cache(list(full),pid))
        caches=[{m:{k:v[m] for k,v in collection.items()} for m in SOURCES} for collection in [first,second]]
        produced={};mass_by_id={}
        for e in by[pid]:
            scores={};masses={}
            for m in SOURCES:
                if not e['target_supported']:
                    values={s:None for s in ['coverage','absolute','retained']};mass=None
                else:
                    d=labels.index(e['diagnosis']);values,mass=coverage_reference(caches[0][m],full,minus,groups,cited[e['explanation_id']],d)
                    check,_=coverage_reference(caches[1][m],full,minus,groups,cited[e['explanation_id']],d)
                    for name,value in values.items():
                        if (value is None)!=(check[name] is None):gate['availability_changes']+=1
                        elif value is not None:gate['coverage_max_delta']=max(gate['coverage_max_delta'],abs(value-check[name]))
                        gate['score_checks']+=1
                scores.update({m+'_'+s:v for s,v in values.items()});masses[m]=mass
            produced[e['explanation_id']]=scores
            mass_by_id[e['explanation_id']]=masses
            weights.append(dict(patient_id=pid,explanation_id=e['explanation_id'],positive_weight_mass=masses))
        for r in saved:
            if r['patient_id']!=pid:continue
            new=copy.deepcopy(r);new['saved_gpu_scores_before_reference']={m:r['scores'][m] for m in METHODS}
            new['scores'].update(produced[r['explanation_id']])
            for m in SOURCES:
                key=m+'_coverage'
                mass=mass_by_id[r['explanation_id']][m]
                new.setdefault('coverage_unavailability',{})[key]=None if new['scores'][key] is not None else ('unsupported_target' if not new['target_supported'] else 'source_unavailable' if mass is None else 'zero_positive_mass')
            new['numerical_reference']=dict(policy=POLICY,methods=METHODS,weights_unchanged=True,original_formal_table_replaced=False)
            result.append(new)
        gate['patients']+=1;gate['queries']+=len(keys)
        if gate['probability_max_delta']>TOL or gate['coverage_max_delta']>TOL or gate['availability_changes']:
            dump(a.out/f'cpu_gate_{a.shard}.json',gate);raise ValueError('CPU numerical reference repeatability failed; no sensitivity accepted')
        print('CPU worker',a.shard,'patients',pos,'/',len(ids),flush=True)
    gate.update(status='passed',tolerance=TOL,torch=torch.__version__,threads=torch.get_num_threads(),dtype='float64',device='cpu')
    dump(a.out/f'cpu_gate_{a.shard}.json',gate);emit(a.out/f'cpu_scores_{a.shard}_PRIVATE.jsonl',result);emit(a.out/f'cpu_weight_mass_{a.shard}_PRIVATE.jsonl',weights)

def prepare(a,c):
    provenance=verify_files(a,c)
    if read(a.source/'input_binding.json')!=provenance:raise ValueError('Frozen input binding changed')
    manifest=read(a.source/'run_manifest.json');dump(a.out/'input_binding.json',provenance)
    for name in ['source_F_impact.csv','source_F_population_impact.csv']:
        shutil.copyfile(verified_saved(a.source,name,manifest),a.out/name)
    for name in manifest['output_sha256']:
        if name.startswith('source_F/'):
            target=a.out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(verified_saved(a.source,name,manifest),target)
    for name in ['explanation_ledger.jsonl','class_shap.json','evaluation_lock.json']:verified_saved(a.source,'TRACE/'+name,manifest)
    ddx=rows(verified_saved(a.source,'DDXPlus/case_results_PRIVATE.jsonl',manifest))
    check=read(verified_saved(a.source,'DDXPlus/baseline_check.json',manifest))
    if set(check.get('quarantined_methods',[]))!={'voting_coverage','voting_absolute','voting_retained'}:raise ValueError('Unexpected DDXPlus replay status')
    emit(a.out/'DDXPlus_case_results_PRIVATE.jsonl',ddx);dump(a.out/'DDXPlus_baseline_check.json',check)
    saved=[]
    for folder in sorted((a.source/'TRACE').glob('shard_*')):
        if read(folder/'run_manifest.json')['status']!='complete':raise ValueError('Saved TRACE shard incomplete')
        saved+=rows(verified_saved(a.source,folder.relative_to(a.source).as_posix()+'/scores.jsonl',manifest))
    ledger=rows(a.source/'TRACE/explanation_ledger.jsonl');expected={(r['explanation_id'],p) for r in ledger for p in ['encoder','lightgbm']}
    if len(saved)!=len(expected) or {(r['explanation_id'],r['probe']) for r in saved}!=expected:raise ValueError('Saved TRACE census mismatch')
    emit(a.out/'TRACE_input_scores_PRIVATE.jsonl',saved)
    dump(a.out/'analysis_protocol.json',dict(policy=POLICY,methods=METHODS,affected_dataset='TRACE',
        rationale='Same-GPU same-input replay drift in frozen GPU implementation; numerical change separated from human-E change.',
        primary_sensitivity='CPU-reference automatic E vs CPU-reference human EXACT E on common samples; other source scores and probes retained after strict original replay checks',
        numerical_shift='Original formal GPU automatic scores vs CPU-reference automatic scores, same E and same endpoints',
        source=str(a.source),source_manifest_sha256=sha(a.source/'run_manifest.json'),bootstrap=a.bootstrap,seed=261006,
        formal_results_frozen=True,training=False,clinical_adjudication=False))

def reuse_cpu_workers(a):
    if not a.resume_cpu_from:return set()
    source=a.resume_cpu_from;manifest=read(source/'run_manifest.json')
    if manifest['workers']!=a.workers:raise ValueError('Keep the same worker count when resuming')
    for name in ['input_binding.json','TRACE_input_scores_PRIVATE.jsonl']:
        saved=verified_saved(source,name,manifest)
        if sha(saved)!=sha(a.out/name):raise ValueError('CPU resume input changed: '+name)
    old=read(verified_saved(source,'analysis_protocol.json',manifest));new=read(a.out/'analysis_protocol.json')
    if old!=new:raise ValueError('CPU resume analysis protocol changed')
    ledger=rows(a.source/'TRACE/explanation_ledger.jsonl');patients=sorted({r['patient_id'] for r in ledger})
    reused=[]
    for i in range(a.workers):
        gate_name=f'cpu_gate_{i}.json'
        if gate_name not in manifest['output_sha256']:continue
        gate=read(verified_saved(source,gate_name,manifest))
        if gate.get('status')!='passed':continue
        names=[gate_name,f'cpu_scores_{i}_PRIVATE.jsonl',f'cpu_weight_mass_{i}_PRIVATE.jsonl']
        for name in names:verified_saved(source,name,manifest)
        rs=rows(source/names[1]);selected=set(patients[i::a.workers])
        expected={(r['explanation_id'],p) for r in ledger if r['patient_id'] in selected for p in ['encoder','lightgbm']}
        if len(rs)!=len(expected) or {(r['explanation_id'],r['probe']) for r in rs}!=expected:raise ValueError('Saved CPU worker census mismatch')
        for name in names:shutil.copyfile(source/name,a.out/name)
        reused.append(i)
    dump(a.out/'cpu_resume_provenance.json',dict(source=str(source),source_manifest_sha256=sha(source/'run_manifest.json'),
        reused_workers=reused,recomputed_workers=[i for i in range(a.workers) if i not in reused],
        reason='OOV handling fixed; already completed v1 workers contained no GRASP-OOV queries, so their arithmetic is unchanged'))
    print('Reused verified CPU workers:',reused,flush=True)
    return set(reused)

def finish(a,c):
    cfg=c['datasets']['TRACE'];reviews=rows(a.review/'human_return_evidence_sets_PRIVATE.jsonl');lookup={}
    for r in reviews:
        if r['dataset']=='TRACE':
            for v in ['automatic','human_exact']:lookup[r['explanation_id']+'::human_return::'+v]=(r,v)
    scored=[];gates=[]
    for i in range(a.workers):
        gate=read(a.out/f'cpu_gate_{i}.json')
        if gate['status']!='passed':raise ValueError('CPU numerical gate incomplete')
        gates.append(gate);scored+=rows(a.out/f'cpu_scores_{i}_PRIVATE.jsonl')
    original={(r['explanation_id'],r['probe']):r for r in rows(Path(cfg['rq2_scores']))}
    drift=score_drift_audit(scored,lookup,original)
    dump(a.out/'TRACE_numerical_reference_vs_formal_PRIVATE.json',drift)
    table(a.out/'TRACE_numerical_reference_vs_formal.csv',[dict(method=m,**v) for m,v in drift['methods'].items()])
    labels=read(a.source/'TRACE/class_shap.json')['labels'];flags=read(a.out/'source_F/source_flags_PRIVATE.json')
    trace,check=cook(scored,lookup,original,labels,flags,numerical_reference_methods=METHODS)
    dump(a.out/'TRACE_reference_check.json',check)
    records=rows(a.out/'DDXPlus_case_results_PRIVATE.jsonl')+trace
    meta=read(a.review/'human_rule_metadata.json')
    for r in records:r['reported_rule_change_types']=meta.get(r['dataset']+'|'+r['explanation_id'],[])
    emit(a.out/'human_review_case_results_PRIVATE.jsonl',records);annotate_human_impact(a,records)
    print('Summarizing paired human-E effects and separate numerical-reference effects',flush=True)
    stats,trans=summarize(records,a.bootstrap,261006)
    for r in stats+trans:r['score_reference_policy']=POLICY+' for GRASP/TabR/No-retrieval; accepted replay for other methods' if r['dataset']=='TRACE' else 'accepted replay; Voting unavailable'
    table(a.out/'human_review_subset_sensitivity.csv',stats);table(a.out/'endpoint_transitions.csv',trans)
    numerical=[]
    for r in trace:
        if r['version']!='automatic':continue
        before=copy.deepcopy(r);after=copy.deepcopy(r)
        before['scores']=copy.deepcopy(original[r['explanation_id'],r['probe']]['scores'])
        after['version']='human_exact'
        # Identical automatic citations and endpoints; only score calculation changes.
        for v in [before,after]:v['added_fact_ids']=[];v['removed_fact_ids']=[]
        numerical.extend([before,after])
    shifts,_=summarize(numerical,a.bootstrap,261006)
    for r in shifts:r.update(contrast='numerical_reference_only_same_automatic_E',automatic_label='accepted_GPU_formal',human_label='CPU_reference_automatic_E')
    table(a.out/'numerical_reference_shift.csv',shifts)
    dump(a.out/'cpu_stability_gate.json',dict(status='passed',workers=gates,patients=sum(x['patients'] for x in gates),
        probability_max_delta=max(x['probability_max_delta'] for x in gates),score_max_delta=max(x['coverage_max_delta'] for x in gates)))
    (a.out/'README_results.md').write_text('''# Human explicit-citation sensitivity with a numerical reference

This supplementary analysis keeps the original patient facts F, fitted parameters, split, original mappings and formal manuscript tables frozen. It uses human_return_exact_E. adjudicated_E remains null. Results are not final adjudication.

TRACE GRASP, TabR and No-retrieval scores use the SAME CPU float64 single-thread implementation for automatic and human EXACT sets. The unchanged checkpoint parameters are promoted exactly from their stored precision. The native forward architecture, exclusions, weights and scoring definitions are retained. Constant eval-mode TabR bank encodings are cached, with cached/uncached parity checks. Every query is evaluated twice in reverse order; probability and score repeatability must satisfy 1e-10. This is a separate numerical-reference sensitivity analysis, not reproduction or replacement of formal GPU values. CPU stability does not establish that historical GPU scores were correct.

Other TRACE scores and both diagnostic probes are retained from the saved run only after comparison against accepted formal automatic-E scores and endpoints. DDXPlus uses previously verified outputs with the entire Voting family unavailable in both mapping versions. No tolerance was enlarged.

An input containing a token outside the GRASP vocabulary is unavailable to that source, matching the frozen implementation. Its facts remain in F/E and other sources continue. Direct A can remain available for a supported cited subset even when full-input support, retained R and coverage are unavailable. Source unavailability and zero positive weight mass are recorded separately; unavailable values are never replaced with zero.

## Files and interpretation

- source_F_impact.csv and source_F_population_impact.csv: frozen-F exposure in formal populations. source_F contains the original item and candidate audit. No fact was repaired or added.
- human_review_subset_sensitivity.csv: automatic vs human EXACT on common availability; fixed_original uses the original probe endpoint for both sets, updated_human uses each set's own probe endpoint. TRACE's three affected sources use the numerical reference on BOTH sides. The estimand remains a human-set effect under that specified numerical policy.
- endpoint_transitions.csv: frozen diagnostic probe predictions/endpoint flips, failure patients, set changes and score availability. Changing the source-score arithmetic does not itself change these probe outcomes.
- numerical_reference_shift.csv: SAME automatic E, SAME endpoints; original formal GPU scores vs CPU reference. Legacy output columns named human_* in this file mean CPU-reference automatic scores; explicit label columns identify this. This file quantifies numerical-policy shift, not annotation change. Do not add its AP differences to those from the human-set analysis; common score intersections can differ.
- TRACE_numerical_reference_vs_formal.csv and its PRIVATE JSON: numeric/availability discrepancies from the formal automatic baseline, before human-E changes.
- human_review_case_results_PRIVATE.jsonl: original IDs, both sets, raw saved scores, reference scores, source-F flags, probe probabilities and endpoints.
- cpu_stability_gate.json and analysis_protocol.json: numerical policy and repeatability evidence.

Random and targeted memberships are analysed separately; the overlapping DDXPlus explanation is deduplicated within each stratum. COMPLETE requires no unresolved IDs, no incomplete flag and reported COMPLETE. All-reviewed uses the human confirmed EXACT set even when remaining items are unresolved. This is not an estimate of the error rate of all generated explanations.

review_scope covers the reviewed records; formal_RQ2_scope retains original full-input dual-probe correctness. Correct/incorrect/all proposals remain separate. Failure AP uses low score to predict reference-recovery failure. Common availability is required across mapping versions and compared methods. Bootstrap uses 1000 draws by default, seed 261006, DDXPlus pattern_id and TRACE subject_group clusters; one-class comparisons are undefined and one-class draws are counted. Intervals are unadjusted. Sparse targeted failures cannot support population-level prevalence claims.

TRACE findings remain conditional on existing F, including confirmed errors and unresolved facts. Human EXACT mapping does not correct F. Source audit exposure is not causal attribution of errors to failures. Rule-expansion and ambiguity labels remain attached to individual records; the human-set contrast is not restricted to unchanged-rule corrections.

These results supplement RQ2 mapping reliability and its audit appendix. They do not replace full-test RQ1/RQ2/RQ3 or intervention tables. Numerical-reference findings must be presented with the separate formal-to-reference audit. No manuscript files are modified.
''',encoding='utf-8')
    return dict(status='complete_separate_numerical_reference',case_rows=len(records),sensitivity_rows=len(stats),models_refit=False,F_changed=False,formal_tables_replaced=False)

def run(a):
    import torch
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_cpu_reference.py','-v'],cwd=ROOT,check=True)
    c=configs(a.config);prepare(a,c);jobs=[];active_ids=[];reused=reuse_cpu_workers(a)
    try:
        for i in range(a.workers):
            if i in reused:continue
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--shard',str(i),'--workers',str(a.workers)]
            for key in ['source','out','config','review']:cmd+=['--'+key,str(getattr(a,key))]
            log=(a.out/f'cpu_worker_{i}.log').open('w',encoding='utf-8')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            jobs.append((subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
            active_ids.append(i)
        while any(p.poll() is None for p,l in jobs):
            print('CPU reference workers',[(i,p.poll()) for i,(p,l) in zip(active_ids,jobs)],flush=True);time.sleep(10)
        if any(p.returncode for p,l in jobs):raise RuntimeError('CPU reference gate failed; inspect worker logs')
    finally:
        for p,l in jobs:
            if p.poll() is None:p.terminate();p.wait()
            l.close()
    return finish(a,c)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key,value in [('source','artifacts/human_return_v2'),('out','artifacts/human_return_cpu_reference_v1'),('config','configs/human_return_20261016.json'),('review','inputs/human_review_20261016')]:p.add_argument('--'+key,type=Path,default=ROOT/value)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--shard',type=int);p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--summary-only',action='store_true')
    p.add_argument('--resume-cpu-from',type=Path,help='Reuse verified completed CPU workers from a prior run into a new output directory')
    a=p.parse_args()
    for key in ['source','out','config','review']:setattr(a,key,getattr(a,key).resolve())
    if a.resume_cpu_from:
        a.resume_cpu_from=a.resume_cpu_from.resolve()
        if a.resume_cpu_from==a.out or a.summary_only:raise ValueError('CPU resume needs a distinct new output and cannot combine with summary-only')
    if a.workers<1:raise ValueError('workers must be positive')
    if a.shard is not None:worker(a,configs(a.config))
    else:
        a.out.mkdir(parents=True,exist_ok=a.summary_only);state={};start=time.time()
        try:
            if a.summary_only:
                c=configs(a.config)
                if verify_files(a,c)!=read(a.out/'input_binding.json'):raise ValueError('Summary frozen inputs changed')
                lock=read(a.out/'run_manifest.json')
                for i in range(a.workers):
                    for n in [f'cpu_gate_{i}.json',f'cpu_scores_{i}_PRIVATE.jsonl']:verified_saved(a.out,n,lock)
                state=finish(a,c)
            else:state=run(a)
        except Exception:state=dict(status='failed_no_acceptance',error=traceback.format_exc());raise
        finally:
            state.update(seconds=time.time()-start,workers=a.workers,bootstrap=a.bootstrap,output_sha256={f.relative_to(a.out).as_posix():sha(f) for f in a.out.rglob('*') if f.is_file() and f.name!='run_manifest.json'})
            dump(a.out/'run_manifest.json',state);archive=Path(str(a.out)+'_reports_PRIVATE.tar.gz')
            with tarfile.open(archive,'w:gz') as t:
                for f in a.out.rglob('*'):
                    if f.is_file():t.add(f,arcname=f.relative_to(a.out).as_posix())
            print('Return numerical-reference sensitivity:',archive,flush=True)
