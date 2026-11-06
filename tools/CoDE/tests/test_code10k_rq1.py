import itertools,random,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from code_eval.rq1_summary import FAMILIES,BUDGETS,summarize,self_source
from code_eval.frozen_bank import FrozenBank
from evaluate_saved_subsets import matched_groups

class RQ1Contracts(unittest.TestCase):
    def test_common_population_and_targeted_missingness(self):
        records=[]
        methods=sorted(set(itertools.chain.from_iterable(FAMILIES.values())))
        for pid,m,frac,probe in itertools.product(['a','b'],methods,BUDGETS,['encoder','lightgbm']):
            valid=not(pid=='b' and m=='voting')
            records.append(dict(patient_id=pid,selector=m,fraction=frac,probe=probe,available=valid,both_full_correct=pid=='a',correct=1 if valid else None,random_accuracy=.5 if valid else None,selected_tokens=['x'],fact_count=1))
        out=summarize(records,bootstrap=20)
        for p in out['panels']:
            expected=1 if p['population']=='both_full_correct' or p['family']=='retrieval' else 2
            self.assertEqual(p['patients'],expected);self.assertEqual(p['gain']['mean'],.5)
        with self.assertRaisesRegex(ValueError,'Duplicate'):summarize(records+[records[0]],bootstrap=2)
    def test_random_draw_matches_facts_preserves_groups(self):
        groups={'a':['a1','a2'],'b':['b'],'c':['c']}
        for seed in range(20):
            fs=set(matched_groups(groups,2,random.Random(seed)))
            self.assertEqual(len(fs),2)
            self.assertIn(fs,[{'a1','a2'},{'b','c'}])
        self.assertTrue(self_source('tree_local_abs_shap','lightgbm'))
        self.assertFalse(self_source('tree_local_abs_shap','encoder'))
    def test_probe_only_api_skips_association(self):
        bank=FrozenBank.__new__(FrozenBank);bank.torch=None;bank.neural={}
        class Encoding:
            def encode(self,ts,ctx):return np.array([len(ts)])
        class Tree:
            def predict(self,x,num_threads):return np.tile([.7,.3],(len(x),1))
        class Counts:
            def distribution(self,*args):raise AssertionError('Unused source evaluated')
        bank.encoding=Encoding();bank.tree=Tree();bank.co=Counts();bank.labels=['a','b']
        self.assertEqual(set(bank.predict([['x']],{},sources=['lightgbm'])),{'lightgbm'})
        with self.assertRaises(ValueError):bank.predict([[]],{},sources=['unknown'])

if __name__=='__main__':unittest.main()
