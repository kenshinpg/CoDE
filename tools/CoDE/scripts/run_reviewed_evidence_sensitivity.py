"""Re-score independently reviewed citation sets; never alter full-cohort results."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys,time,tarfile,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prepare_lantern_cohort import read,rows,sha,dump
from run_coverage_controls import coverage
from code_eval.proxy_metrics import average_precision

VERSIONS={'automatic':'automatic_fact_ids','reviewed_strict':'reviewed_cited_fact_ids',
          'reviewed_preserve_ambiguous':'counterfactual_fact_ids_preserve_ambiguous'}

def checked_sets(review,record):
    full={f['fact_id'] for f in record['facts']}
    sets={v:set(review[k]) for v,k in VERSIONS.items()}
    if any(not fs<=full for fs in sets.values()):raise ValueError('Nonpatient review fact')
    if sets['automatic']!=set(record['provisional_cited_fact_ids']):raise ValueError('Automatic citation drift')
    if sets['reviewed_preserve_ambiguous']!=sets['reviewed_strict']|(sets['automatic']&set(review['ambiguous_fact_ids'])):
        raise ValueError('Ambiguous-preserving policy drift')
    return sets

def compatible(old,new,tol=1e-6):
    for k in ['patient_id','diagnosis','reference_diagnosis','both_full_correct','full_correct','probe_available','failure','cited_fact_count']:
        if old[k]!=new[k]:raise ValueError('Automatic baseline endpoint drift: '+k)
    for k,v in new['scores'].items():
        if k.startswith('case_'):continue
        w=old['scores'][k]
        if (v is None)!=(w is None) or (v is not None and abs(v-w)>tol):raise ValueError(f'Automatic score drift {k}: {v}, {w}')

def summarize(records):
    by={(r['audit_id'],r['probe'],r['version']):r for r in records}
    output={}
    for flag in ['random_sample','targeted_primary_failure']:
        for probe in ['encoder','lightgbm']:
            for filt in ['all_reviewed','primary_full_correct_and_correct_proposal']:
                base=[r for r in records if r['version']=='automatic' and r[flag] and r['probe']==probe
                      and (filt=='all_reviewed' or r['both_full_correct'] and r['diagnosis_correct'])]
                for version in ['reviewed_strict','reviewed_preserve_ambiguous']:
                    pairs=[(r,by[r['audit_id'],probe,version]) for r in base]
                    valid=[(a,b) for a,b in pairs if a['probe_available'] and b['probe_available']]
                    transitions=Counter(f"{'failure' if a['failure'] else 'success'}_to_{'failure' if b['failure'] else 'success'}" for a,b in valid)
                    methods={}
                    for m in (base[0]['scores'] if base else []):
                        pp=[(a,b) for a,b in valid if a['scores'][m] is not None and b['scores'][m] is not None]
                        delta=[b['scores'][m]-a['scores'][m] for a,b in pp]
                        v=dict(common_records=len(pp),common_patients=len({a['patient_id'] for a,b in pp}),
                               mean_change=float(np.mean(delta)) if delta else None,
                               median_change=float(np.median(delta)) if delta else None,
                               before_unavailable=sum(a['scores'][m] is None for a,b in pairs),
                               after_unavailable=sum(b['scores'][m] is None for a,b in pairs))
                        # Failure-enriched sample is not a valid population AP comparison.
                        if flag=='random_sample':
                            y0=[a['failure'] for a,b in pp];y1=[b['failure'] for a,b in pp]
                            s0=[-a['scores'][m] for a,b in pp];s1=[-b['scores'][m] for a,b in pp]
                            v.update(ap_before=average_precision(y0,s0),
                                     ap_new_score_old_endpoint=average_precision(y0,s1),
                                     ap_new_score_new_endpoint=average_precision(y1,s1),
                                     failures_before=sum(y0),failures_after=sum(y1))
                        methods[m]=v
                    output[f'{flag}|{probe}|{filt}|{version}']=dict(records=len(base),patients=len({r['patient_id'] for r in base}),
                        paired_probe_available=len(valid),transitions=dict(transitions),methods=methods,
                        changed_citation_sets=sum(set(a['fact_ids'])!=set(b['fact_ids']) for a,b in pairs),
                        changed_prediction=sum(a['prediction']!=b['prediction'] for a,b in valid),
                        mean_fact_count_change=float(np.mean([len(b['fact_ids'])-len(a['fact_ids']) for a,b in pairs])) if pairs else None)
    return output

def run(a):
    from code_eval.frozen_bank import FrozenBank,PatientCache
    from code_eval.tree_subset import Encoding,checked_contributions,pattern
    from code_eval.case_retrieval import CaseRetrieval
    from code_eval.cohort_stats import identity
    package=read(a.review/'input_manifest.json')
    for name,h in package['sha256'].items():
        if sha(a.review/name)!=h:raise ValueError('Review package drift: '+name)
    reviews=list(rows(a.review/'reviewed_sets_for_sensitivity.jsonl'))
    tasks={r['audit_id']:r for r in rows(a.review/'tasks.jsonl')}
    assert len(reviews)==96 and len({r['explanation_id'] for r in reviews})==96
    rm=read(a.reserved/'run_manifest.json');bm=read(a.bank/'run_manifest.json');bankdata=read(a.bank/'frozen_source_bank.json')
    lock=read(a.bank/'recipe_lock.json');tm=read(a.source/'tree/run_manifest.json')
    if any(m['status']!='complete' for m in [rm,bm,tm]):raise ValueError('Incomplete upstream')
    if sha(a.bank/'frozen_source_bank.json')!=rm['source_bank_sha256'] or sha(a.bank/'frozen_source_bank.json')!=bm['output_sha256']['frozen_source_bank.json']:raise ValueError('Frozen bank drift')
    if sha(a.bank/'recipe_lock.json')!=bankdata['recipe_sha256']:raise ValueError('Recipe drift')
    if sha(a.source/'tree/run_manifest.json')!=lock['tree_manifest_sha256']:raise ValueError('Tree drift')
    if sha(a.reserved/'scores.jsonl')!=rm['output_sha256']['scores.jsonl']:raise ValueError('Saved score drift')
    cohort=a.reserved/'cohort'
    if sha(cohort/'evaluation_records.jsonl')!=rm['record_sha256']:raise ValueError('Record drift')
    if sha(cohort/'excluded_fit_patterns.txt')!=lock['exclusion_sha256']:raise ValueError('Exclusion drift')
    dictionary=cohort/'evidence_dictionary.json'
    if [h for p,h in tm['source_sha256'].items() if Path(p).name==dictionary.name]!=[sha(dictionary)]:raise ValueError('Dictionary drift')
    cm=read(a.controls/'run_manifest.json')
    if cm['status']!='complete' or cm['source_tree_sha256']!=lock['tree_manifest_sha256'] or sha(a.controls/'class_shap.json')!=cm['output_sha256']['class_shap.json']:raise ValueError('Class SHAP drift')
    paths={k:Path(v['path']) for k,v in bankdata['sources'].items()}
    for k,p in paths.items():
        if sha(p)!=bankdata['sources'][k]['sha256']:raise ValueError('Model drift: '+k)
    wanted={r['explanation_id'] for r in reviews}
    records={r['explanation_id']:r for r in rows(cohort/'evaluation_records.jsonl') if r['explanation_id'] in wanted}
    old={(r['explanation_id'],r['probe']):r for r in rows(a.reserved/'scores.jsonl') if r['kind']=='natural' and r['explanation_id'] in wanted}
    assert set(records)==wanted and len(old)==192
    labels=tm['labels'];enc=Encoding(read(dictionary));bank=FrozenBank(paths,enc,labels,a.threads)
    cls=read(a.controls/'class_shap.json');assert cls['columns']==enc.columns and cls['labels']==labels
    means=np.asarray(cls['mean_absolute']);gain=bank.tree.feature_importance(importance_type='gain')
    fitpath=a.source/'tree/teacher_fit_selected_private.jsonl'
    if sha(fitpath)!=lock['fit_sha256']:raise ValueError('Fitting data drift')
    fit=list(rows(fitpath));excluded=set((cohort/'excluded_fit_patterns.txt').read_text().splitlines())
    if any(r['pattern_id'] in excluded for r in fit):raise ValueError('Fit/research overlap')
    retrieval=CaseRetrieval(fit,labels,5);fit_ids={r['input_group_id'] for r in fit}
    dump(a.out/'protocol_lock.json',dict(input_manifest=package,review_manifest_sha256=sha(a.review/'input_manifest.json'),
        reserved_scores_sha256=sha(a.reserved/'scores.jsonl'),bank_sha256=sha(a.bank/'frozen_source_bank.json'),
        versions=VERSIONS,scope='96 completed reviews only; not a replacement of full-cohort tables',
        summary='random and targeted overlap reported separately; targeted AP omitted',models_trained=0,api_calls=0))
    grouped=defaultdict(list)
    for review in reviews:grouped[review['patient_id']].append(review)
    output=[]
    print('Recompute',len(reviews),'explanations,',len(grouped),'patients, 3 sets each',flush=True)
    for n,(pid,rr) in enumerate(grouped.items(),1):
        r=records[rr[0]['explanation_id']];tokens={f['fact_id']:f['provenance']['source_token'] for f in r['facts']};full=set(tokens);groups=r['groups'];ctx=r['demographics']
        assert pattern(list(tokens.values()))==r['pattern_id']
        fid=identity(float(ctx['age']),ctx['sex'],list(tokens.values()));assert fid not in fit_ids
        sets={}
        for review in rr:
            record=records[review['explanation_id']];task=tasks[review['audit_id']]
            assert review['patient_id']==record['patient_id']==pid
            assert record['text']==task['explanation'] and record['facts']==task['patient_facts']
            assert all(record[k]==r[k] for k in ['facts','groups','demographics','pattern_id'])
            sets[review['audit_id']]=checked_sets(review,record)
        minus={g:full-set(fs) for g,fs in groups.items()}
        pc=PatientCache(bank,tokens,ctx);pc.ensure([full]+list(minus.values())+[fs for vs in sets.values() for fs in vs.values()])
        fp={m:pc.get(m,full) for m in pc.cache};ready={m:fp[m] is not None and all(pc.get(m,fs) is not None for fs in minus.values()) for m in pc.cache}
        rc={}
        def case(fs):
            k=tuple(sorted(fs))
            if k not in rc:rc[k]=retrieval.query([tokens[f] for f in k],ctx,excluded_pattern=r['pattern_id'],excluded_identity=fid)
            return rc[k]['probabilities'] if rc[k]['available'] else None
        cf=case(full);cminus={g:case(fs) for g,fs in minus.items()}
        cr=cf is not None and all(v is not None for v in cminus.values())
        shap=checked_contributions(bank.tree,enc.encode(list(tokens.values()),ctx)[None,:],len(labels))[0]
        cols={g:sorted({enc.token(tokens[f])[0] for f in fs}) for g,fs in groups.items()}
        for review in rr:
            eid=review['explanation_id'];record=records[eid];diagnosis=record['ranked_diagnoses'][0];d=labels.index(diagnosis)
            ref=old[eid,'encoder']['reference_diagnosis']
            weights={m:{g:max(0.,float(fp[m][d]-pc.get(m,fs)[d])) for g,fs in minus.items()} if ready[m] else None for m in pc.cache}
            weights.update(tree_gain={g:float(gain[cs].sum()) for g,cs in cols.items()},
                tree_positive_shap={g:max(0.,float(shap[d,cs].sum())) for g,cs in cols.items()},
                tree_absolute_shap={g:float(np.abs(shap[d,cs]).sum()) for g,cs in cols.items()},
                tree_class_shap={g:float(means[d,cs].sum()) for g,cs in cols.items()})
            cw={g:max(0.,cf[d]-v[d]) for g,v in cminus.items()} if cr else None
            both=all(fp[m] is not None and labels[int(fp[m].argmax())]==ref for m in ['encoder','lightgbm'])
            for version,E in sets[review['audit_id']].items():
                scores=dict(fact_coverage=len(E)/len(full),group_coverage=coverage({g:1. for g in groups},groups,E))
                for m in pc.cache:
                    pred=pc.get(m,E)
                    scores[m+'_coverage']=coverage(weights[m],groups,E) if weights[m] is not None else None
                    scores[m+'_absolute']=float(pred[d]) if pred is not None else None
                    scores[m+'_retained']=float(pred[d]-fp[m][d]) if pred is not None and fp[m] is not None else None
                for m in ['tree_gain','tree_positive_shap','tree_absolute_shap','tree_class_shap']:scores[m]=coverage(weights[m],groups,E)
                cp=case(E);scores.update(case_absolute=cp[d] if cp is not None else None,
                    case_retained=cp[d]-cf[d] if cp is not None and cf is not None else None,
                    case_coverage=coverage(cw,groups,E) if cw is not None else None)
                for probe in ['encoder','lightgbm']:
                    pred=pc.get(probe,E);avail=pred is not None and fp[probe] is not None
                    row=dict(audit_id=review['audit_id'],explanation_id=eid,patient_id=pid,pattern_id=r['pattern_id'],
                        random_sample=review['random_sample'],targeted_primary_failure=review['targeted_primary_failure'],
                        version=version,probe=probe,diagnosis=diagnosis,diagnosis_correct=diagnosis==ref,reference_diagnosis=ref,
                        both_full_correct=both,full_correct=bool(avail and labels[int(fp[probe].argmax())]==ref),probe_available=avail,
                        failure=bool(avail and labels[int(pred.argmax())]!=ref),prediction=labels[int(pred.argmax())] if avail else None,
                        proposed_support=float(pred[d]) if avail else None,reference_support=float(pred[labels.index(ref)]) if avail else None,
                        fact_ids=sorted(E),cited_fact_count=len(E),scores=scores)
                    if version=='automatic':compatible(old[eid,probe],row)
                    output.append(row)
        if n%10==0 or n==len(grouped):print('Completed patients',n,'/',len(grouped),flush=True)
    with (a.out/'scores.jsonl').open('w',encoding='utf8') as f:
        for r in output:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    dump(a.out/'sensitivity_summary.json',summarize(output))
    return dict(status='complete',explanations=len(reviews),patients=len(grouped),rows=len(output),automatic_baseline_checks=192,
        source_workbook_sha256=package['source_workbook_sha256'],models_trained=0,api_calls=0,
        note='subset sensitivity only; single-reviewer mapping reference; two pending reviews excluded')

def main():
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,default=ROOT/'inputs/blind_review_20261007')
    p.add_argument('--reserved',type=Path,default=Path('artifacts/reserved_results_v1'));p.add_argument('--bank',type=Path,default=Path('artifacts/formal_sources_v1'))
    p.add_argument('--source',type=Path,default=Path('artifacts/lantern_development_v1'));p.add_argument('--controls',type=Path,default=Path('artifacts/coverage_controls_v1'))
    p.add_argument('--out',type=Path,default=Path('artifacts/reviewed_evidence_sensitivity_v1'));p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);start=time.time();state={}
    try:state=run(a)
    except Exception:state=dict(status='failed',error=traceback.format_exc());raise
    finally:
        state['elapsed_seconds']=time.time()-start
        state['code_sha256']={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'src/code_eval/frozen_bank.py',ROOT/'src/code_eval/case_retrieval.py']}
        state['output_sha256']={f.name:sha(f) for f in a.out.iterdir() if f.is_file()};dump(a.out/'run_manifest.json',state)
        archive=a.out.with_name(a.out.name+'_reports.tar.gz')
        with tarfile.open(archive,'w:gz') as t:
            for f in a.out.iterdir():
                if f.is_file():t.add(f,arcname=f.name)
        print('Return report:',archive,flush=True)
if __name__=='__main__':main()
