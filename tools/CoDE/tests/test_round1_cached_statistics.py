import importlib.util, json, tempfile, unittest
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'scripts/run_round1_cached_statistics.py'
spec=importlib.util.spec_from_file_location('cached_stats',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class CachedStatistics(unittest.TestCase):
    def test_ap_ties_missingness_and_deletion_replay(self):
        self.assertAlmostEqual(m.ap([1,0,1],[0,0,0]),2/3)
        self.assertIsNone(m.ap([0,0],[1,2]))
        rows=[]
        for i,(fail,disease,score,complete) in enumerate([(True,'A',.1,True),(False,'B',.9,True),(False,'A',.5,True),(True,'B',.2,False)]):
            rows.append(dict(explanation_id=str(i),patient_id=str(i),subject_group=str(i),probe='encoder',mapping_available=True,scan_status='COMPLETE' if complete else 'PARTIAL',target_supported=True,probe_available=True,both_full_correct=True,diagnosis_correct=True,reference_failure=fail,proposal_failure=fail,reference_diagnosis=disease,scores={'group_coverage':score,'tree_positive_shap':score}))
        m.DATA={'TRACE':rows}
        p=dict(mapping_policy='complete_only',probe='encoder',stratum='correct',endpoint='reference',family='tree_rules',common=3,patients=3,failures=1,methods={k:{'ap':1.} for k in rows[0]['scores']})
        out,contrasts,audit=m.panel_job(('TRACE',p))
        self.assertEqual(audit['replayed_records'],3)
        removed_a=[r for r in out if r['removed_disease']=='A']
        self.assertTrue(all(r['ap'] is None for r in removed_a))
        self.assertEqual(len(contrasts),2)
        bad=dict(p,common=4)
        with self.assertRaises(AssertionError):m.panel_job(('TRACE',bad))

    def test_patient_weighting_and_numerical_sign(self):
        # Input is already averaged within patient; explanation counts must not reweight it.
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);panel='local_shap|support|add|correct|encoder'
            rs=[dict(panel=panel,patient_id=str(i),explanations=n,metrics={'high_minus_random_d':x,'high_minus_random_y':x}) for i,(n,x) in enumerate([(9,.1),(1,-.1),(3,1e-11)])]
            mean=float(np.mean([.1,-.1,1e-11]));ref={panel:dict(patients=3,analyzed_explanations=13,metrics={k:{'mean':mean} for k in ['high_minus_random_d','high_minus_random_y']})}
            for name,obj in [('rs.json',rs),('ref.json',ref)]: (folder/name).write_text(json.dumps(obj))
            out=m.intervention_tables(folder/'rs.json',folder/'ref.json')[0]
            self.assertAlmostEqual(out['mean_pp'],mean*100)
            self.assertEqual((out['positive_patients'],out['zero_patients'],out['negative_patients']),(1,1,1))
if __name__=='__main__':unittest.main()
