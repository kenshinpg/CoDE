import sys,unittest,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from code_eval.round1_followup import APContrast,PairContrast,resample,intervals,availability,absolute_endpoints
from code_eval.proxy_metrics import average_precision
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def record(pid,y,x,z,**kw):
    r=dict(patient_id=pid,pattern_id=pid,subject_group=pid,explanation_id=str(pid)+str(y)+str(x),
        diagnosis='a',reference_diagnosis='a',diagnosis_correct=True,reference_failure=bool(y),
        cited_fact_count=2,probe='encoder',mapping_available=True,target_supported=True,
        both_full_correct=True,probe_available=True,full_correct=True,
        scores=dict(x=x,z=z,fact_coverage=.5,group_coverage=.5),
        full_probabilities=[.8,.2],cited_probabilities=[.2,.8] if y else [.8,.2])
    r.update(kw);return r

class FollowupContracts(unittest.TestCase):
    def test_weighted_ap_equals_expanded_tied_cluster_bootstrap(self):
        rs=[record('a',1,.2,.7),record('a',0,.2,.5),record('b',1,.6,.3),record('c',0,.9,.5)]
        c=APContrast(rs,'x','z','pattern_id');rng=np.random.default_rng(13)
        for _ in range(40):
            counts=rng.integers(0,5,3);w=counts[[0,0,1,2]];ix=np.repeat(np.arange(4),w)
            a=average_precision([rs[i]['reference_failure'] for i in ix],[-rs[i]['scores']['x'] for i in ix])
            b=average_precision([rs[i]['reference_failure'] for i in ix],[-rs[i]['scores']['z'] for i in ix])
            if a is None:self.assertTrue(np.isnan(c.evaluate(w)))
            else:self.assertAlmostEqual(c.evaluate(w),a-b,places=12)
    def test_family_shared_draws_and_missingness_no_intersection(self):
        a=[record('a',1,.2,.7),record('b',0,.8,.1),record('c',0,.6,.7)]
        c=APContrast(a,'x','z','pattern_id');d=APContrast(a[:2],'x','z','pattern_id')
        draws=resample([c,c,d],100,9)
        np.testing.assert_equal(draws[:,0],draws[:,1]);self.assertEqual(c.n,3);self.assertEqual(d.n,2)
        report=intervals([c,c,d],draws);self.assertGreater(report[0]['undefined_replicates'],0)
        self.assertEqual(report[0]['simultaneous_status'],'conditional_on_defined_replicates')
    def test_pair_patient_weighting_and_fixed_diagnosis_count(self):
        rs=[record('a',0,.9,.1),record('a',1,.1,.9),record('a',1,.2,.8),
            record('b',0,.1,.9),record('b',1,.9,.1),record('b',1,.1,.9,cited_fact_count=3)]
        c=PairContrast(rs,'x','z','patient_id')
        self.assertEqual(c.pairs,3);self.assertEqual(c.n,2);self.assertEqual(c.evaluate(np.ones(2)),0)
    def test_availability_direct_not_removed_by_undefined_coverage(self):
        rs=[record('a',0,.8,.5),record('b',1,None,.4)]
        p=dict(probe='encoder',stratum='correct',endpoint='reference',family='direct_support',methods={'z':{}},common=2,patients=2)
        report=availability(rs,[p],'DDX','RQ2')
        self.assertEqual(next(r for r in report if r['kind']=='funnel' and r['stage']=='common_scores')['records'],2)
        p.update(family='coverage',methods={'x':{}},common=1,patients=1)
        report=availability(rs,[p],'DDX','RQ2')
        self.assertEqual(next(r for r in report if r['kind']=='score_missing_reason')['stage'],'reason_not_recorded')
    def test_absolute_patient_mean_and_missing_background(self):
        rs=[record('a',0,.2,.5),record('a',1,.2,.5),record('b',1,.1,.2)]
        output=absolute_endpoints(rs,['a','b'],'DDX','RQ2',{})
        r=next(x for x in output if x['input']=='H+E' and x['population']=='all_mapped' and x['probe']=='encoder')
        self.assertAlmostEqual(r['accuracy_record_weighted'],1/3);self.assertEqual(r['accuracy_patient_mean'],.25)
        bg=next(x for x in output if x['input']=='H' and x['probe']=='encoder')
        self.assertIsNone(bg['accuracy_record_weighted']);self.assertEqual(bg['unavailable'],2)
    def test_trace_excluded_profile_generation_denominators(self):
        from run_round1_followup import trace_profile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);test=[];patients=[];prov=[];generation=[];ledger=[];scores=[]
            for i in range(359):
                pid=str(i);tokens=['f_@_present']+(['u_@_present'] if i<37 else [])
                test.append(dict(patient_id=pid,tokens=tokens,label='a'))
                facts=[]
                for j,t in enumerate(tokens):
                    fid=pid+'_'+str(j);feature,state=t.split('_@_');facts.append(dict(fact_id=fid,feature_id=feature,state=state))
                    prov.append(dict(fact_id=fid,sources=['history']))
                patients.append(dict(patient_id=pid,facts=facts))
                for j in range(4):
                    eid=pid+'|'+str(j);valid=j<3
                    generation.append(dict(task_id=eid,patient_id=pid,status='complete' if valid else 'failed',parsed_diagnoses=['a','b','c'] if valid else None))
                    ledger.append(dict(explanation_id=eid,repair_status='COMPLETE' if valid else 'UNAVAILABLE'))
                    for probe in ['encoder','lightgbm']:
                        scores.append(dict(patient_id=pid,explanation_id=eid,probe=probe,full_probabilities=None if i<37 and probe=='encoder' else [.9,.1],full_correct=not(i<37 and probe=='encoder')))
            def emit(name,rs):
                p=root/name;p.write_text(''.join(json.dumps(r)+'\n' for r in rs));return str(p)
            emit('test_records_PRIVATE.jsonl',test);emit('train_records_PRIVATE.jsonl',[dict(tokens=['f_@_present'])])
            emit('patients_model_input_PRIVATE.jsonl',patients)
            cfg=dict(view=str(root),fact_provenance=emit('provenance.jsonl',prov),generations=emit('generation.jsonl',generation),ledger=emit('ledger.jsonl',ledger))
            results,diseases,sources=trace_profile(cfg,scores)
            r=next(x for x in results if x['group']=='excluded_37' and 'generation_attempts' in x)
            self.assertEqual(r['generation_attempts'],148);self.assertEqual(r['mapping_complete_rate_all'],.75)
            self.assertEqual(r['mapping_complete_rate_valid'],1);self.assertEqual(r['llm_top1_all'],.75)
            self.assertEqual(r['unknown_state_occurrences'],37)

if __name__=='__main__':unittest.main()
