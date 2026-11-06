import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from code_eval.adjudication_sensitivity import *
from code_eval.round1_followup import resample
from run_adjudication_sensitivity import summarize,emit,read

def record(i,version='automatic'):
    y=i%3==0
    return dict(dataset='DDXPlus',explanation_id='e'+str(i),audit_id='N'+str(i),patient_id='p'+str(i),pattern_id='cluster'+str(i//2),
        subject_group='s'+str(i//2),probe='encoder',version=version,random_sample=True,targeted_sample=i%2==0,reference_status='complete',
        diagnosis='D',reference_diagnosis='D',diagnosis_correct=True,target_supported=True,mapping_available=True,
        probe_available=True,both_full_correct=True,full_correct=True,reference_failure=y,proposal_failure=y,
        full_probabilities=[.9,.1],cited_probabilities=[.2,.8] if y else [.8,.2],prediction='X' if y else 'D',cited_fact_count=1,
        fact_ids=['f'],scores=dict(cohd_coverage=.12+i*.08,group_coverage=.2+(i%2)*.2,tree_positive_shap=.7-i*.06,lightgbm_absolute=.4+i*.04,fact_coverage=.3))

def sample():
    rr=[]
    for i in range(8):
        a=record(i);b=copy.deepcopy(a);b['version']='adjudicated';b['scores']['tree_positive_shap']+=.15 if i%2 else -.1
        if i==1:b.update(reference_failure=True,proposal_failure=True,prediction='X',cited_probabilities=[.1,.9],fact_ids=[])
        rr.extend([a,b])
    return rr

class Contracts(unittest.TestCase):
    def test_same_common_population_and_unresolved_exclusion(self):
        rr=sample();rr=[r for r in rr if not(r['explanation_id']=='e0' and r['version']=='adjudicated')]
        rr[3]['scores']['tree_positive_shap']=None
        panels,flow=make_panels(rr);p=panels[0]
        for c in p['comparisons']:
            self.assertTrue(all(a['explanation_id']==b['explanation_id'] for a,b in c['pairs']))
            self.assertFalse(any(a['explanation_id']=='e0' for a,b in c['pairs']))
            for a,b in c['pairs']:
                for r in [a,b]:self.assertTrue(all(r['scores'][m] is not None for m in c['needed']))
        self.assertGreater(len(p['comparisons'][0]['pairs']),len(p['comparisons'][1]['pairs']))

    def test_replay_requires_same_availability_scores_and_endpoints(self):
        a=record(1);b=copy.deepcopy(a)
        self.assertEqual(automatic_gate(a,b),0)
        b['scores']['tree_positive_shap']=None
        with self.assertRaises(ValueError):automatic_gate(a,b)
        b=copy.deepcopy(a);b['scores']['cohd_coverage']+=.01
        with self.assertRaises(ValueError):automatic_gate(a,b)
        b=copy.deepcopy(a);b['reference_failure']=not b['reference_failure']
        with self.assertRaises(ValueError):automatic_gate(a,b)

    def test_frozen_reference_and_ledger(self):
        r=dict(dataset='DDXPlus',explanation_id='e',audit_id='N',patient_id='p',random_sample=True,targeted_sample=False,
            status='unresolved',final_fact_ids=['f'],unresolved_fact_ids=['g'],automatic_fact_ids=['g'],common_omissions_checked=True,completed_by=['R01','R02'])
        validate_reference([r]);base={'e':dict(patient_id='p',partition='test',reviewed_available=True,cited_fact_ids=['g'])}
        ledger,lookup=make_ledger([r],base,{'p':{'f','g'}})
        self.assertEqual(len(ledger),1)
        r.update(status='complete',unresolved_fact_ids=[]);ledger,lookup=make_ledger([r],base,{'p':{'f','g'}})
        self.assertEqual(len(ledger),2)
        r['final_fact_ids']=['outside']
        with self.assertRaises(ValueError):make_ledger([r],base,{'p':{'f','g'}})

    def test_cluster_block_matches_original_serial_bootstrap_and_ties(self):
        p=make_panels(sample())[0][0];cs=contrasts(p);keys=sorted({k for c in cs for k in c.cluster});rng=np.random.default_rng(261022)
        counts=np.array([np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys)) for _ in range(40)])
        pid,start,out=bootstrap_block((p,0,counts,keys))
        np.testing.assert_array_equal(out,resample(cs,40,261022))
        c=cs[0];w=counts[0];ix={k:i for i,k in enumerate(keys)};pairs=p['comparisons'][0]['pairs']
        duplicate=[a for a,b in pairs for _ in range(w[ix[a['pattern_id']]])]
        expected=average_precision([r['reference_failure'] for r in duplicate],[-r['scores']['cohd_coverage'] for r in duplicate])-average_precision([r['reference_failure'] for r in duplicate],[-r['scores']['group_coverage'] for r in duplicate])
        self.assertAlmostEqual(out[0,0],expected)

    def test_endpoint_flip_and_no_failure_reporting(self):
        p=make_panels(sample())[0][0];draws=resample(contrasts(p),30,3);rr,tt=finish_panel(p,draws,30,3)
        self.assertTrue(all(t['success_to_failure']==1 for t in tt))
        self.assertTrue(all(r['adjudicated_failures']==r['automatic_failures']+1 for r in rr))
        empty=make_panels([])[0][0];rr,tt=finish_panel(empty,np.full((30,9),np.nan),30,3)
        self.assertTrue(all(r['automatic_difference'] is None and r['automatic_undefined_replicates']==30 for r in rr))

    def test_cook_recomputes_prediction_and_preserves_unavailable(self):
        review=dict(dataset='DDXPlus',explanation_id='e1',audit_id='N1',patient_id='p1',status='complete',
            random_sample=True,targeted_sample=False,automatic_fact_ids=['f'],final_fact_ids=[])
        lookup={v:(review,v) for v in ['automatic','adjudicated']};scored=[];old={}
        for probe in PROBES:
            a=record(1);a['probe']=probe;old['e1',probe]=copy.deepcopy(a)
            for version in lookup:
                r=copy.deepcopy(a);r['explanation_id']=version
                if version=='adjudicated':r.update(reference_failure=True,proposal_failure=True,cited_probabilities=[.1,.9])
                scored.append(r)
        cooked,check=cook(scored,lookup,old,['D','X'])
        self.assertEqual(check['automatic_rows'],2)
        self.assertTrue(all(r['prediction']=='X' for r in cooked if r['version']=='adjudicated'))
        for r in scored+list(old.values()):r.update(probe_available=False,full_correct=False,both_full_correct=False,full_probabilities=None,cited_probabilities=None,reference_failure=None,proposal_failure=None)
        cooked,check=cook(scored,lookup,old,['D','X'])
        self.assertTrue(all(r['prediction'] is None for r in cooked))

    def test_parallel_summary_resume_is_identical(self):
        with tempfile.TemporaryDirectory() as d:
            a=SimpleNamespace(out=Path(d),bootstrap=501,seed=261022,summary_workers=2)
            rr=sample();emit(a.out/'scored_records_PRIVATE.jsonl',rr);summarize(a,rr)
            first=read(a.out/'statistics.json');summarize(a,rr);second=read(a.out/'statistics.json')
            self.assertEqual(first['results'],second['results']);self.assertEqual(first['transitions'],second['transitions'])
            blocks=[np.load(a.out/f'bootstrap/panel_00_block_{i:05d}.npz')['draws'] for i in [0,250,500]]
            np.testing.assert_array_equal(np.vstack(blocks),resample(contrasts(make_panels(rr)[0][0]),501,261022))
            (a.out/'scored_records_PRIVATE.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):summarize(a,rr)

if __name__=='__main__':unittest.main()
