import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_trace27b_review32 as m


def example(i=0,version='automatic',failure=False):
    return dict(explanation_id='e'+str(i),patient_id=str(i),subject_group='s'+str(i),
        audit_id='QC'+str(i),probe='encoder',version=version,generator='Qwen3.8-27B',
        mapping_available=True,scan_status='COMPLETE',repair_status='COMPLETE',
        target_supported=True,probe_available=True,both_full_correct=True,full_correct=True,
        diagnosis_correct=True,diagnosis='appendicitis',reference_diagnosis='appendicitis',
        reference_failure=failure,proposal_failure=failure,cited_fact_count=1,
        full_probabilities=[.9,.05,.03,.02],cited_probabilities=[.1,.8,.05,.05] if failure else [.9,.05,.03,.02],
        prediction='cholecystitis' if failure else 'appendicitis',
        scores=dict(fact_coverage=.5,group_coverage=.5,cohd_coverage=.2+i*.1,
            tree_positive_shap=.3+i*.1,lightgbm_absolute=.9-i*.1))


class ReplayContracts(unittest.TestCase):
    @unittest.skipUnless((ROOT/'inputs/trace27b_review32_R02_20261027/resolved27_PRIVATE.json').exists(), 'Requires separately authorized audit input')
    def test_resolved27_excluded5_identity_and_sets(self):
        folder=ROOT/'inputs/trace27b_review32_R02_20261027'
        resolved=m.read(folder/'resolved27_PRIVATE.json'); excluded=m.read(folder/'excluded5_PRIVATE.json')
        m.validate_reviews(resolved,excluded)
        self.assertEqual({r['case'] for r in excluded},{'QC04','QC07','QC09','QC13','QC21'})
        self.assertEqual(sum(len(set(r['auto_fact_ids'])-set(r['reviewed_fact_ids'])) for r in resolved),38)
        self.assertEqual(sum(len(set(r['reviewed_fact_ids'])-set(r['auto_fact_ids'])) for r in resolved),13)
        bad=copy.deepcopy(resolved);bad[0]['fully_resolved']=False
        with self.assertRaises(ValueError):m.validate_reviews(bad,excluded)

    def test_common_population_and_fixed_updated_endpoints(self):
        records=[]
        for i in range(3):
            a=example(i,failure=i==0);b=example(i,'reviewed',failure=i==1)
            if i==2:b['scores']['tree_positive_shap']=None
            records.extend([a,b])
        p=m.make_panels(records)[0]
        self.assertEqual(p['common_records'],2)
        self.assertEqual(p['failures_to_success'],1);self.assertEqual(p['success_to_failure'],1)
        self.assertEqual([r['reference_failure'] for r in p['versions']['reviewed_fixed_endpoint']],[True,False])
        self.assertEqual([r['reference_failure'] for r in p['versions']['reviewed_updated_endpoint']],[False,True])

    def test_jobs_reuse_identity_and_reject_nonpatient_facts(self):
        r=dict(case='QC01',patient_id='0',task_id='e0',auto_fact_ids=['f1'],reviewed_fact_ids=['f2'])
        e=dict(explanation_id='e0',patient_id='0',subject_group='s0',reviewed_available=True,
            repair_status='COMPLETE',cited_fact_ids=['f1'],diagnosis='appendicitis')
        patients={'0':dict(patient_id='0',facts=[dict(fact_id='f1'),dict(fact_id='f2')])}
        baseline={('e0',p):dict(example(),probe=p) for p in ['encoder','lightgbm']}
        jobs,lookup=m.make_jobs([r],{'e0':e},patients,{'0':dict(label='appendicitis')},baseline)
        self.assertEqual(len(jobs),1);self.assertEqual(len(lookup),2)
        self.assertEqual([x['cited_fact_ids'] for x in jobs[0][2]],[['f1'],['f2']])
        self.assertEqual(e['cited_fact_ids'],['f1'])
        bad=dict(r,reviewed_fact_ids=['missing'])
        with self.assertRaises(ValueError):m.make_jobs([bad],{'e0':e},patients,{'0':{}},baseline)

    def test_gate_rejects_drift_and_freezes_full_input(self):
        a=example();b=example(version='reviewed',failure=True)
        old=copy.deepcopy(a);old['probe']='encoder'
        review=dict(case='QC01',task_id='e0',patient_id='0')
        baseline={('e0','encoder'):old,('e0','lightgbm'):dict(old,probe='lightgbm')}
        lookup={'a':(review,'automatic'),'b':(review,'reviewed')}
        scored=[dict(r,explanation_id=eid,probe=p) for r,eid in [(a,'a'),(b,'b')] for p in ['encoder','lightgbm']]
        out,gate=m.cook(scored,lookup,baseline)
        self.assertEqual(gate['automatic_rows'],2)
        self.assertTrue(next(r for r in out if r['version']=='reviewed')['reference_failure'])
        bad=copy.deepcopy(scored);bad[0]['reference_failure']=True
        with self.assertRaises(ValueError):m.cook(bad,lookup,baseline)
        bad=copy.deepcopy(scored);bad[2]['full_probabilities']=[.8,.1,.05,.05]
        with self.assertRaises(ValueError):m.cook(bad,lookup,baseline)

    def test_weighted_ap_matches_explicit_cluster_duplicates(self):
        rs=[example(i,failure=i in [0,2]) for i in range(4)]
        rs[1]['subject_group']=rs[0]['subject_group']
        c=m.APContrast(rs,'cohd_coverage','group_coverage','subject_group')
        weights=np.array([2,2,0,1]);expanded=[r for r,n in zip(rs,weights) for _ in range(n)]
        y=[r['reference_failure'] for r in expanded]
        expected=m.average_precision(y,[-r['scores']['cohd_coverage'] for r in expanded])-m.average_precision(y,[-r['scores']['group_coverage'] for r in expanded])
        self.assertAlmostEqual(c.evaluate(weights),expected)
        self.assertTrue(np.isnan(c.evaluate(np.zeros(4))))

    def test_bootstrap_reproducible_and_empty_ap_undefined(self):
        p=m.make_panels([example(0),example(0,'reviewed')])[0];p['seed_index']=0
        first=m.bootstrap_block((p,0,25))[2];second=m.bootstrap_block((p,0,25))[2]
        np.testing.assert_array_equal(first,second)
        self.assertTrue(np.isnan(first).all())
        stats=m.intervals(m.contrasts_for(p)[:3],first[:,:3])
        self.assertIsNone(stats[0]['difference']);self.assertEqual(stats[0]['undefined_replicates'],25)

    def test_parallel_summary_and_resume(self):
        records=[]
        for i in range(4):
            records.extend([example(i,failure=i==0),example(i,'reviewed',failure=i==1)])
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(out=Path(tmp),bootstrap=250,summary_workers=2)
            m.summarize(records,a)
            before=m.sha(a.out/'statistics.json')
            m.summarize(records,a)
            self.assertEqual(before,m.sha(a.out/'statistics.json'))
            stats=m.read(a.out/'statistics.json')
            self.assertEqual(len(stats['comparisons']),54)
            self.assertEqual(len(stats['changes']),36)


if __name__=='__main__':unittest.main()
