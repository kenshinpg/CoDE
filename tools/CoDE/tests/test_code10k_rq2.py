import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from code_eval.rq2_summary import panel,add_pairwise_families
from run_coverage_controls import coverage

def row(eid,**kw):
    r=dict(explanation_id=eid,patient_id='p',pattern_id='pattern',probe='encoder',mapping_available=True,target_supported=True,diagnosis='d',reference_diagnosis='y',component='coverage',diagnosis_correct=False,both_full_correct=True,probe_available=True,cited_fact_count=1,reference_failure=True,proposal_failure=False,scores=dict(fact_coverage=.5,group_coverage=.5,cohd_coverage=.9,tree_positive_shap=.1))
    r.update(kw);return r

class RQ2Contracts(unittest.TestCase):
    def test_patient_counts_concentration_and_pair_difference(self):
        add_pairwise_families()
        family='paired_cohd_coverage_vs_tree_positive_shap'
        rs=[row('a',diagnosis_correct=True),row('b',diagnosis_correct=True),
            row('c',diagnosis_correct=True,reference_failure=False,scores=dict(fact_coverage=.5,group_coverage=.5,cohd_coverage=.1,tree_positive_shap=.9)),
            row('d',patient_id='p2',pattern_id='p2',reference_diagnosis='other',diagnosis_correct=True,reference_failure=False),
            row('missing',patient_id='p3',mapping_available=False)]
        result=panel(rs,'encoder','correct','reference',family,20)
        audit=result['information_audit']
        self.assertEqual(audit['failure'],dict(explanations=2,patients=1))
        self.assertEqual(audit['sequential_flow'][0]['patients'],3)
        self.assertEqual(audit['sequential_flow'][-1]['patients'],2)
        self.assertEqual(audit['pair_funnel']['equal_fact_count_discordant'],dict(pairs=2,patients=1))
        self.assertEqual(audit['paired_concordance_differences']['cohd_coverage minus tree_positive_shap']['difference'],-1.)
        self.assertEqual(len(audit['leave_one_disease']),2)
        removed=next(r for r in audit['leave_one_disease'] if r['removed_disease']=='y')
        self.assertIsNone(removed['ap']['cohd_coverage'])

    def test_pair_funnel_preserves_strict_count_control(self):
        rs=[row('a'),row('b',reference_failure=False,cited_fact_count=2)]
        result=panel(rs,'encoder','all','reference','quantity_only',10)
        audit=result['information_audit']
        self.assertEqual(audit['pair_funnel']['discordant']['pairs'],1)
        self.assertEqual(result['discordant_pairs'],0)
        self.assertEqual(audit['pair_funnel']['equal_fact_count_discordant']['patients'],0)

    def test_target_alignment_never_relabels_reference(self):
        add_pairwise_families()
        rs=[row('a'),row('b',reference_failure=False,proposal_failure=True,scores=dict(fact_coverage=.5,group_coverage=.5,cohd_coverage=.1,tree_positive_shap=.9))]
        family='paired_cohd_coverage_vs_tree_positive_shap'
        ref=panel(rs,'encoder','incorrect','reference',family,10)
        aligned=panel(rs,'encoder','incorrect','proposal',family,10)
        self.assertEqual(ref['failures'],1);self.assertEqual(aligned['failures'],1)
        self.assertTrue(rs[0]['reference_failure']);self.assertFalse(rs[0]['proposal_failure'])
        self.assertEqual(ref['pair_patients'],1)
        self.assertEqual(ref['methods']['cohd_coverage']['ap'],.5)
        self.assertEqual(aligned['methods']['cohd_coverage']['ap'],1.)
    def test_unavailable_not_empty_and_unsupported_retained(self):
        rs=[row('ok'),row('partial',mapping_available=False),row('outside',target_supported=False),row('full_wrong',both_full_correct=False)]
        q=panel(rs,'encoder','all','reference','quantity_only',10)
        self.assertEqual(q['input_mapped'],3);self.assertEqual(q['common'],2)
        add_pairwise_families();r=panel(rs,'encoder','all','reference','paired_cohd_coverage_vs_tree_positive_shap',10)
        self.assertEqual(r['common'],1)
        self.assertIn('cohd_coverage minus tree_positive_shap',r['paired_ap_differences'])
    def test_missing_coverage_does_not_remove_direct_or_quantity(self):
        r=row('empty',scores=dict(fact_coverage=0.,group_coverage=0.,cohd_coverage=None,tree_positive_shap=None))
        self.assertEqual(panel([r],'encoder','all','reference','quantity_only',10)['common'],1)
        self.assertIsNone(coverage({'g':0.},{'g':['x']},set()))
        self.assertEqual(coverage({'g':1.},{'g':['x']},set()),0.)

if __name__=='__main__':unittest.main()
