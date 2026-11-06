import sys,unittest,tempfile,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from code_eval.human_return import validate_reviews,complete,make_ledger,ap_draws,weights,summarize,cook,score_drift_audit,sha,dump
from audit_human_source_F import base_impact,panel_records

def review(eid='e'):
    return dict(dataset='DDXPlus',explanation_id=eid,patient_id='p',audit_id='n',sample_membership={'random':True,'targeted':True},
        automatic_E=['f1'],human_return_exact_E=['f2'],unresolved_fact_ids=[],reported_completeness='COMPLETE',has_unresolved_or_incomplete=False,adjudicated_E=None)

class Contracts(unittest.TestCase):
    def score_fixture(self):
        r=review();lookup={};records=[];old={}
        base=dict(patient_id='p',reference_diagnosis='a',diagnosis='a',probe_available=True,both_full_correct=True,
            reference_failure=False,proposal_failure=False,cited_fact_count=1,cited_probabilities=[.8,.2],
            scores={'voting_coverage':.2,'voting_absolute':.4,'voting_retained':-.2,'tree_positive_shap':.8})
        for probe in ['encoder','lightgbm']:old['e',probe]=dict(base)
        for v in ['automatic','human_exact']:
            eid='e::human_return::'+v;lookup[eid]=(r,v)
            for probe in ['encoder','lightgbm']:
                records.append(dict(base,explanation_id=eid,probe=probe,scores=dict(base['scores'],voting_coverage=.7)))
        return records,lookup,old

    def test_complete_drift_audit_and_explicit_whole_family_quarantine(self):
        rs,lookup,old=self.score_fixture();a=score_drift_audit(rs,lookup,old)
        self.assertEqual(a['mismatching_methods'],['voting_coverage']);self.assertEqual(len(a['details']),2)
        with self.assertRaisesRegex(ValueError,'Automatic score drift'):cook(rs,lookup,old,['a','b'])
        q={'voting_coverage','voting_absolute','voting_retained'}
        out,check=cook(rs,lookup,old,['a','b'],quarantined_methods=q)
        self.assertEqual(check['automatic_rows'],2)
        for x in out:
            self.assertTrue(all(x['scores'][m] is None for m in q))
            self.assertEqual(x['quarantined_raw_scores']['voting_coverage'],.7)
            self.assertEqual(x['scores']['tree_positive_shap'],.8)
        self.assertEqual(rs[0]['scores']['voting_coverage'],.7)

    def test_nonvoting_drift_still_fails_with_quarantine(self):
        rs,lookup,old=self.score_fixture();rs[0]['scores']['tree_positive_shap']=.1
        self.assertEqual(len(score_drift_audit(rs,lookup,old)['mismatching_methods']),2)
        with self.assertRaisesRegex(ValueError,'tree_positive_shap'):
            cook(rs,lookup,old,['a','b'],quarantined_methods={'voting_coverage','voting_absolute','voting_retained'})
        with self.assertRaisesRegex(ValueError,'Only the explicit'):
            cook(rs,lookup,old,['a','b'],quarantined_methods={'tree_positive_shap'})

    def test_resume_copy_rejects_modified_cache(self):
        from run_human_return_sensitivity import verified_copy
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';dst=root/'dest';src.mkdir();dst.mkdir()
            f=src/'scores.jsonl';f.write_text('original',encoding='utf-8')
            lock={'output_sha256':{'scores.jsonl':sha(f)}}
            verified_copy(src,dst,'scores.jsonl',lock);self.assertEqual((dst/'scores.jsonl').read_text(),'original')
            f.write_text('changed',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'identity changed'):verified_copy(src,dst,'scores.jsonl',lock)
            with self.assertRaisesRegex(ValueError,'Unsafe'):verified_copy(src,dst,'../scores.jsonl',lock)

    def test_resume_reuses_verified_audit_and_shards(self):
        from types import SimpleNamespace
        from run_human_return_sensitivity import reuse_source_audit,reuse_dataset_shards
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'old';dst=Path(d)/'new';src.mkdir();dst.mkdir()
            (src/'source_F').mkdir();(src/'DDXPlus/shard_0').mkdir(parents=True);(dst/'DDXPlus').mkdir()
            binding={'frozen':'identity'};dump(src/'input_binding.json',binding)
            dump(src/'source_F/source_flags_PRIVATE.json',{});dump(src/'source_F/summary.json',{'status':'complete'})
            for name in ['source_F_impact.csv','source_F_population_impact.csv']:(src/name).write_text('column\nvalue\n')
            for name in ['explanation_ledger.jsonl','class_shap.json','evaluation_lock.json']:
                (src/'DDXPlus'/name).write_text('{}');(dst/'DDXPlus'/name).write_text('{}')
            (src/'DDXPlus/shard_0/scores.jsonl').write_text('{}\n')
            dump(src/'DDXPlus/shard_0/run_manifest.json',{'status':'complete'})
            dump(src/'run_manifest.json',{'output_sha256':{f.relative_to(src).as_posix():sha(f) for f in src.rglob('*') if f.is_file() and f.name!='run_manifest.json'}})
            a=SimpleNamespace(resume_from=src,out=dst)
            self.assertEqual(reuse_source_audit(a,binding),{})
            self.assertTrue(reuse_dataset_shards(a,'DDXPlus'));self.assertFalse(reuse_dataset_shards(a,'TRACE'))
            self.assertEqual((dst/'DDXPlus/shard_0/scores.jsonl').read_text(),'{}\n')
            (src/'DDXPlus/shard_0/scores.jsonl').write_text('modified')
            with self.assertRaisesRegex(ValueError,'identity changed'):reuse_dataset_shards(a,'DDXPlus')

    def test_frozen_voting_order_risk_on_synthetic_tie(self):
        import code_eval.case_retrieval as cr
        from unittest.mock import patch
        original=cr.by_feature
        tokens=lambda cs:[f'f{i}_@_v{j}' for i,n in enumerate(cs) for j in range(n)]
        query=[f'f{i}_@_v0' for i in range(6)]
        bank=[]
        for i,ts in enumerate([tokens([3,7,10,5,9,11]),tokens([3,7,10,9,11,5])]+[query+[f'extra{i}'] for i in range(4)]):
            bank.append(dict(input_group_id=str(i),pattern_id=cr.pattern(ts),tokens=ts,label='B' if i==1 else 'A',context={'age':0,'sex':'F'}))
        model=cr.CaseRetrieval(bank,['A','B'],k=5);results=[]
        for rev in [False,True]:
            with patch.object(cr,'by_feature',lambda ts,rev=rev:dict(sorted(original(ts).items(),reverse=rev))):
                results.append(model.query(query,{'age':100,'sex':'M'},excluded_pattern='',excluded_identity=''))
        self.assertNotEqual(results[0]['probabilities'],results[1]['probabilities'])
        self.assertTrue(all(r['boundary_tie'] for r in results))

    def test_explicit_not_adjudicated_and_overlap(self):
        r=review();c=validate_reviews([r]);self.assertEqual(c['DDXPlus']['overlap'],1)
        r['adjudicated_E']=[]
        with self.assertRaises(ValueError):validate_reviews([r])
        r['adjudicated_E']=None;r['human_return_exact_E']=None
        with self.assertRaises(ValueError):validate_reviews([r])

    def test_complete_requires_all_three_flags(self):
        r=review();self.assertTrue(complete(r));r['unresolved_fact_ids']=['f3'];self.assertFalse(complete(r))
        r['unresolved_fact_ids']=[];r['has_unresolved_or_incomplete']=True;self.assertFalse(complete(r))

    def test_ledger_retains_original_and_rejects_nonpatient(self):
        r=review();e=dict(explanation_id='e',patient_id='p',partition='test',reviewed_available=True,cited_fact_ids=['f1'])
        ledger,_=make_ledger([r],{'e':e},{'p':{'f1','f2'}})
        self.assertEqual(e['cited_fact_ids'],['f1']);self.assertEqual(ledger[1]['cited_fact_ids'],['f2'])
        with self.assertRaises(ValueError):make_ledger([r],{'e':e},{'p':{'f1'}})

    def test_AP_ties_weighted_draws_and_undefined(self):
        x=ap_draws([1,0,1],[.9,.5,.5],np.array([[1.,1,1],[2.,1,0],[0.,1,0]]))
        self.assertAlmostEqual(x[0],5/6);self.assertAlmostEqual(x[1],1.);self.assertTrue(np.isnan(x[2]))
        self.assertTrue(np.isnan(ap_draws([1],[1],np.ones((2,1)))).all())
        self.assertTrue(np.isnan(ap_draws([],[],np.empty((2,0)))).all())

    def test_cluster_resampling_keeps_same_patient_group(self):
        pp=[({'pattern_id':'g1'},{}),({'pattern_id':'g1'},{}),({'pattern_id':'g2'},{})]
        w,n=weights(pp,30,261006,'DDXPlus');self.assertEqual(n,2);np.testing.assert_array_equal(w[:,0],w[:,1])

    def test_source_errors_not_propagated_to_same_feature(self):
        pats=[dict(patient_id=p,facts=[dict(fact_id='f'+p,feature_id='HP_X',state='absent')]) for p in ['a','b']]
        split={'a':{'partition':'test'},'b':{'partition':'train'}}
        src=[dict(record_type='FROZEN_FACT',judgment='ERROR_CONFIRMED',patient_id='a',fact_id='fa',record_id='s',inherited_issue_type='CONCEPT')]
        x,flags=base_impact(src,pats,split,[])
        self.assertFalse(x[0]['in_training_input']);self.assertEqual(x[0]['same_feature_state_train_patients'],1);self.assertNotIn('b',flags)

    def test_panel_uses_actual_common_scores_and_scan_policy(self):
        base=dict(probe='encoder',mapping_available=True,probe_available=True,both_full_correct=True,target_supported=True,
                  diagnosis_correct=True,scan_status='COMPLETE',scores={'group_coverage':.5,'tree_positive_shap':None,'lightgbm_absolute':.7})
        partial=dict(base,scan_status='PARTIAL')
        p=dict(probe='encoder',family='direct_support',stratum='correct',methods=['group_coverage','lightgbm_absolute'])
        self.assertEqual(len(panel_records([base,partial],p)),1)
        p['methods']=['group_coverage','tree_positive_shap'];self.assertEqual(panel_records([base],p),[])

    def test_auto_endpoint_drift_stops_cooking(self):
        r=review();key='e::human_return::automatic'
        original=dict(patient_id='p',reference_diagnosis='a',diagnosis='a',probe_available=True,both_full_correct=True,
                      reference_failure=False,proposal_failure=False,cited_fact_count=1)
        changed=dict(original,explanation_id=key,probe='encoder',reference_failure=True)
        with self.assertRaisesRegex(ValueError,'Automatic endpoint drift'):
            cook([changed],{key:(r,'automatic')},{('e','encoder'):original},['a','b'])

    def test_fixed_and_updated_endpoint_and_overlap(self):
        records=[]
        for j in range(3):
            for version in ['automatic','human_exact']:
                records.append(dict(dataset='DDXPlus',patient_id='p'+str(j),explanation_id='e'+str(j),pattern_id='g'+str(j),probe='encoder',version=version,
                    sample_membership={'random':True,'targeted':j==0},human_complete=j!=2,both_full_correct=True,diagnosis_correct=True,target_supported=True,
                    mapping_available=True,probe_available=True,reference_failure=(j==0 if version=='automatic' else j==1),
                    prediction=('a' if j==0 else 'b') if version=='automatic' else 'c',fact_ids=['a'] if version=='automatic' else ['b'],
                    added_fact_ids=['b'],removed_fact_ids=['a'],source_F_error_fact_ids=[],scores={'group_coverage':(.1 if j==0 else .8) if version=='automatic' else (.1 if j==1 else .8)}))
        stats,trans=summarize(records,20)
        chosen=[r for r in stats if r['audit_group']=='random' and r['human_subset']=='all_reviewed' and r['analysis_scope']=='review_scope' and r['proposal_stratum']=='all']
        fixed=next(r for r in chosen if r['endpoint_policy']=='fixed_original');updated=next(r for r in chosen if r['endpoint_policy']=='updated_human')
        self.assertEqual(fixed['common_explanations'],3);self.assertLess(fixed['human_estimate'],updated['human_estimate'])
        t=next(r for r in trans if r['row_type']=='probe' and r['audit_group']=='random' and r['human_subset']=='all_reviewed' and r['analysis_scope']=='review_scope' and r['proposal_stratum']=='all' and r['probe']=='encoder')
        self.assertEqual((t['success_to_failure'],t['failure_to_success']),(1,1))

if __name__=='__main__':unittest.main()
