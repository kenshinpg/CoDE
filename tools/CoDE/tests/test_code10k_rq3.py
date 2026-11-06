import sys,unittest,copy
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from run_code10k_rq3 import sampled_sets,controlled_panel,validate_records,support_scores
from test_code10k_rq2 import row

class ControlledContracts(unittest.TestCase):
    def test_float32_support_subtraction_promotes_before_arithmetic(self):
        full=np.float32(.91)
        probabilities=np.array([.001,.1,.4,.8,.95],dtype=np.float32)
        old_offsets=[float(a)-float(a-full) for a in probabilities]
        self.assertGreater(max(old_offsets)-min(old_offsets),1e-12)
        offsets=[]
        for p in probabilities:
            a,r=support_scores(p,full);offsets.append(a-r)
        self.assertLess(max(offsets)-min(offsets),1e-12)
        self.assertTrue(all(abs(x-float(full))<1e-12 for x in offsets))
        records=[]
        for fraction,p in zip([.25,.5,.75],probabilities):
            for probe in ['encoder','lightgbm']:
                a,r=support_scores(p,full)
                records.append(row(str(fraction),fraction=fraction,probe=probe,diagnosis_correct=True,
                    cited_tokens=['x'],cited_fact_count=1,anchor_fact_count=1,unique_sets=1,
                    scores=dict(lightgbm_absolute=a,lightgbm_retained=r)))
        self.assertEqual(validate_records(records,{'p'})['unique_sets'],3)

    def test_group_integrity_budget_and_reproducibility(self):
        groups={'a':['a1','a2'],'b':['b'],'c':['c'],'d':['d1','d2','d3']}
        for fraction in [.25,.5,.75]:
            count,sets=sampled_sets(groups,261006,'p',fraction,10)
            self.assertEqual((count,sets),sampled_sets(groups,261006,'p',fraction,10))
            self.assertEqual(len(sets),len(set(sets)))
            for fs in sets:
                self.assertEqual(len(fs),count)
                for group in groups.values():self.assertIn(len(set(group)&set(fs)),[0,len(group)])

    def test_budget_separation_and_ar_order(self):
        records=[]
        for fraction in [.25,.5,.75]:
            for i,fail in enumerate([False,True]):
                a=.8 if not fail else .2
                r=row(f'{fraction}-{i}',fraction=fraction,diagnosis_correct=True,reference_failure=fail,
                    scores=dict(fact_coverage=.5,group_coverage=.5,tree_positive_shap=.4 if not fail else .3,lightgbm_absolute=a,lightgbm_retained=a-.9))
                records.append(r)
        out=controlled_panel(records,'encoder',.25,'paired_tree_positive_shap_vs_lightgbm_absolute',20)
        self.assertEqual(out['common'],2);self.assertEqual(out['discordant_pairs'],1)
        self.assertEqual(out['information_audit']['paired_concordance_differences']['tree_positive_shap minus lightgbm_absolute']['difference'],0)

    def test_sampling_audit_rejects_drift(self):
        records=[]
        for fraction in [.25,.5,.75]:
            for probe in ['encoder','lightgbm']:
                records.append(row(str(fraction),fraction=fraction,probe=probe,diagnosis_correct=True,
                    cited_tokens=['x'],cited_fact_count=1,anchor_fact_count=1,unique_sets=1,
                    scores=dict(lightgbm_absolute=.7,lightgbm_retained=-.2)))
        self.assertEqual(validate_records(records,{'p'})['unique_sets'],3)
        bad=copy.deepcopy(records);bad[0]['scores']['lightgbm_retained']=0
        with self.assertRaisesRegex(ValueError,'constant shift'):validate_records(bad,{'p'})
        bad=copy.deepcopy(records);bad[0]['anchor_fact_count']=2
        with self.assertRaisesRegex(ValueError,'count mismatch'):validate_records(bad,{'p'})
        with self.assertRaises(ValueError):validate_records(records,{'p','missing'})

if __name__=='__main__':unittest.main()
