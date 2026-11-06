import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from code_eval.frozen_bank import PatientCache
from run_controlled_sets import sampled_sets
from code_eval.proxy_metrics import summarize


class FakeBank:
    def __init__(self):self.calls=[]
    def predict(self,inputs,context):
        self.calls.append(inputs)
        return {n:[np.array([.8,.2]) if 'good' in ts else np.array([.1,.9]) for ts in inputs] for n in ['lightgbm','encoder','grasp','cohd']}


class ReservedContracts(unittest.TestCase):
    def test_cache_exact_sets_dedup_context_and_no_unknown_facts(self):
        bank=FakeBank();cache=PatientCache(bank,{'f1':'good','f2':'other'},{'age':30,'sex':'F'})
        cache.ensure([['f1','f2'],['f2','f1'],[],['f2']]);self.assertEqual(len(bank.calls[0]),3)
        cache.ensure([['f1','f2']]);self.assertEqual(len(bank.calls),1)
        self.assertEqual(cache.get('encoder',['f2']).argmax(),1)
        self.assertEqual(cache.get('cohd',['f2','f1']).argmax(),0)
        with self.assertRaises(ValueError):cache.ensure([['not_a_fact']])

    def test_random_sets_have_no_label_or_model_input(self):
        groups={'g1':['a','b'],'g2':['c'],'g3':['d']}
        n,sets=sampled_sets(groups,261006,'p',.25,10)
        self.assertTrue(all(len(fs)==n for fs in sets))
        self.assertEqual((n,sets),sampled_sets(groups,261006,'p',.25,10))

    def test_common_missingness_and_pair_definition(self):
        base=dict(patient_id='p',pattern_id='pattern',diagnosis='d',reference_diagnosis='d',cited_fact_count=1,probe_available=True,full_correct=True)
        rs=[dict(base,failure=False,scores={'fact_coverage':.2,'group_coverage':.3,'source':.8}),
            dict(base,failure=True,scores={'fact_coverage':.2,'group_coverage':.3,'source':.1}),
            dict(base,failure=False,scores={'fact_coverage':.2,'group_coverage':.3,'source':None})]
        result=summarize(rs,['fact_coverage','group_coverage','source'],10,5,baseline_methods=['fact_coverage','group_coverage'])
        self.assertEqual(result['eligible'],3);self.assertEqual(result['common'],2)
        self.assertEqual(result['methods']['source']['unavailable_rate'],1/3)
        self.assertEqual(result['methods']['source']['pair_concordance'],1)
        self.assertEqual(result['methods']['fact_coverage']['pair_concordance'],.5)
        self.assertIn('source minus group_coverage',result['paired_ap_differences'])


if __name__=='__main__':unittest.main()
