import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from code_eval.round1_extra_baselines import frequency_weights,greedy_order,first_sis,anchor_mask
from run_round1_extra_baselines import summarize_panel

class BaselineContracts(unittest.TestCase):
    def test_vendor_manifest_portable_paths_and_hash_guard(self):
        import tempfile,json,hashlib
        from pathlib import PurePosixPath
        from run_round1_extra_baselines import vendor_parts,verify_vendor
        self.assertEqual(str(PurePosixPath('/vendor').joinpath(*vendor_parts(r'sis\sis.py'))),'/vendor/sis/sis.py')
        for bad in ['../escape','/absolute',r'C:\absolute']:
            with self.assertRaises(ValueError):vendor_parts(bad)
        verify_vendor(ROOT)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);v=root/'vendor';(v/'sis').mkdir(parents=True)
            (v/'sis/sis.py').write_bytes(b'original')
            (v/'round1_baseline_sources.json').write_text(json.dumps([dict(path=r'sis\sis.py',sha256=hashlib.sha256(b'original').hexdigest())]))
            verify_vendor(root)
            (v/'sis/sis.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Vendor content changed'):verify_vendor(root)

    def test_worker_exports_original_natural_schema_and_group_sets(self):
        import tempfile,json
        from types import SimpleNamespace
        from unittest.mock import patch
        import run_round1_extra_baselines as module
        import run_code10k_rq1
        class Bank:
            def predict(self,sets,context,sources):
                pp=[np.array([1-(.01+.98*('a' in s and 'b' in s)),.01+.98*('a' in s and 'b' in s)]) for s in sets]
                return {name:pp for name in sources}
        class Encoding:
            def groups(self,tokens):return {t:[t] for t in tokens}
        query={'p':dict(patient_id='p',tokens=['a','b','c'],context={},label='yes')}
        ledger=[dict(patient_id='p',explanation_id='e')]
        scores=[dict(patient_id='p',explanation_id='e',probe=p,mapping_available=True,cited_tokens=['a'],scores={'group_coverage':1/3}) for p in ['encoder','lightgbm']]
        old=[dict(patient_id='p',selector='tree_local_abs_shap',fraction=.25,probe=p,both_full_correct=True,available=True,selected_tokens=['a'],correct=0) for p in ['encoder','lightgbm']]
        with tempfile.TemporaryDirectory() as d:
            a=SimpleNamespace(dataset='ddx',out=Path(d),shard=0,shards=1,anchor_queries=200000)
            x=SimpleNamespace(source=Path(d),rq2_ledger=Path('ledger'),rq2_scores=Path('scores'))
            with patch.object(module,'args_for',return_value=x),patch.object(module,'setup',return_value=(Bank(),Encoding(),['no','yes'],query)),patch.object(run_code10k_rq1,'verify',return_value=({},[{'tokens':['a','b']}],{})),patch.object(module,'read_baseline',return_value=([],old)),patch.object(module,'rows',side_effect=lambda p:ledger if p==Path('ledger') else scores):
                module.worker(a)
            rr=[json.loads(l) for l in (Path(d)/'ddx/shard_0/selections_PRIVATE.jsonl').read_text().splitlines()]
            self.assertEqual(len(rr),22)
            self.assertTrue(all(set(r['selected_tokens'] or [])<={'a','b','c'} for r in rr))
            natural=[json.loads(l) for l in (Path(d)/'ddx/shard_0/natural_PRIVATE.jsonl').read_text().splitlines()]
            self.assertEqual(len(natural),2);self.assertEqual(natural[0]['scores']['frequency'],.5)
    def test_training_frequency_counts_patients_and_unseen_tokens(self):
        w=frequency_weights([{'tokens':['a','a','b']},{'tokens':['a']}],{'g':['a'],'h':['b','c']})
        self.assertEqual(w['frequency']['g'],1.)
        self.assertEqual(w['frequency']['h'],.25)
        self.assertGreater(w['idf']['h'],w['idf']['g'])
    def test_greedy_and_upstream_sis_preserve_interaction(self):
        score=lambda masks: .01+.98*(masks[:,0]*masks[:,1])
        self.assertEqual(greedy_order(score,3,2),[0,1])
        self.assertEqual(first_sis(score,3)['indices'],[0,1])
        self.assertEqual(first_sis(lambda m:np.ones(len(m))*.5,3)['status'],'full_below_threshold')
        self.assertEqual(first_sis(lambda m:np.ones(len(m))*.99,3)['status'],'empty_sufficient')
    def test_upstream_anchor_uses_class_agreement_and_censors_limit(self):
        classify=lambda m:(m[:,0]*m[:,1]).astype(int)
        result=anchor_mask(classify,3,1,261023)
        self.assertEqual(result['status'],'complete');self.assertEqual(result['indices'],[0,1])
        self.assertEqual(result['verification_success'],2000)
        self.assertEqual(anchor_mask(classify,3,1,261023,limit=1)['status'],'query_limit')
    def test_cluster_comparison_and_no_event_ap(self):
        rows=[dict(cluster=str(i),correct=1,random_accuracy=.5) for i in range(4)]
        self.assertEqual(summarize_panel(('selection','x',rows,None,5))['gain_ci95'],[.5,.5])
        rows=[dict(cluster=str(i),reference_failure=False,scores={'a':.2,'b':.3}) for i in range(3)]
        r=summarize_panel(('natural','x',rows,['a','b'],5))
        self.assertIsNone(r['difference']);self.assertEqual(r['valid_bootstrap'],0)

if __name__=='__main__':unittest.main()
