import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_round1_temperature as m
from round1_temperature_data import validate_cache,partition_check

class TemperatureContracts(unittest.TestCase):
    def test_development_fit_preserves_argmax_and_improves_nll(self):
        p=np.array([[.99,.005,.005],[.99,.005,.005],[.005,.99,.005],[.005,.005,.99]])
        y=np.array([0,1,1,2]);r=m.fit_temperature([(p,y)])
        self.assertGreater(r['temperature'],1)
        self.assertLess(r['dev_nll_after'],r['dev_nll_before'])
        self.assertTrue(np.array_equal(m.transform(p,1),p))
        for t in [.05,1,20]:self.assertTrue(np.array_equal(m.transform(p,t).argmax(1),p.argmax(1)))
        with self.assertRaises(ValueError):m.transform(p,0)

    def test_cache_axes_and_partition_barrier(self):
        p=np.array([[.8,.2],[np.nan,np.nan]])
        self.assertEqual(validate_cache(p,['d1','d2'],[0,1],['d1','d2'],[0,1],2).tolist(),[True,False])
        with self.assertRaises(ValueError):validate_cache(p,['d2','d1'],[0,1],['d1','d2'],[0,1],2)
        with self.assertRaises(ValueError):validate_cache(p,['d1','d2'],[1,0],['d1','d2'],[0,1],2)
        with self.assertRaises(ValueError):partition_check(['d1'],['d1'],['train'])
        partition_check(['dev'],['test'],['train'])

    def records(self):
        rs=[]
        for i,(p,y) in enumerate([([.8,.15,.05],'a'),([.3,.4,.3],'a'),([.7,.2,.1],'a'),([.2,.7,.1],'a')]):
            for probe in ['encoder','lightgbm']:
                rs.append(dict(explanation_id=f'e{i}',patient_id='p'+str(i//2),pattern_id='p'+str(i//2),diagnosis='a',reference_diagnosis=y,probe=probe,probe_available=True,mapping_available=True,both_full_correct=True,diagnosis_correct=True,target_supported=True,reference_failure=bool(np.argmax(p)!=0),cited_probabilities=p,cited_fact_count=2,scores=dict(cohd_coverage=.8-i*.1,group_coverage=.7-i*.04,tree_positive_shap=.6-i*.08,lightgbm_absolute=p[0])))
        return rs

    def test_vector_join_changes_only_A(self):
        rs=self.records();out,n=m.calibrate_scores(rs,['a','b','c'],2.)
        self.assertEqual(n,4)
        for a,b in zip(rs,out):
            for k in a:
                if k!='scores':self.assertEqual(a[k],b[k])
            for k in a['scores']:
                if k!='lightgbm_absolute':self.assertEqual(a['scores'][k],b['scores'][k])
        self.assertNotEqual(rs[0]['scores']['lightgbm_absolute'],out[0]['scores']['lightgbm_absolute'])
        rs[0]['scores']['lightgbm_absolute']=0
        with self.assertRaises(ValueError):m.calibrate_scores(rs,['a','b','c'],2.)

    def test_core_replay_and_parallel_blocks_are_paired(self):
        rs=self.records();after,_=m.calibrate_scores(rs,['a','b','c'],2.);aa=[r for r in rs if r['probe']=='encoder'];panels=[]
        for left,right in m.AP_COMPARISONS:
            c=m.APContrast(aa,left,right,'pattern_id');diff=c.evaluate(np.ones(c.n))
            panels.append(dict(probe='encoder',endpoint='reference',stratum='correct',family='paired_'+left+'_vs_'+right,methods={left:{},right:{}},common=4,patients=2,failures=2,paired_ap_differences={left+' minus '+right:{'difference':diff,'ci95':[diff,diff]}}))
        f=m.make_family('DDX','rq2',rs,after,panels);m.init_workers([f])
        _,_,b,a=m.bootstrap_block((0,0,12));_,_,b1,a1=m.bootstrap_block((0,0,5));_,_,b2,a2=m.bootstrap_block((0,5,7))
        np.testing.assert_equal(b,np.vstack([b1,b2]));np.testing.assert_equal(a,np.vstack([a1,a2]))
        np.testing.assert_equal(a[:,:2],b[:,:2])
        summary=m.core_summary([f],[(b,a)])
        self.assertEqual(len(summary),9)
        for r in summary:
            if r['variant']=='calibrated_minus_original' and r['right']=='group_coverage':self.assertEqual(r['difference'],0.)
        panels[0]['common']=5
        with self.assertRaises(ValueError):m.make_family('DDX','rq2',rs,after,panels)

    def test_calibration_denominators_do_not_change(self):
        rs=self.records()
        for r in rs:r['full_probabilities']=[.8,.15,.05]
        fits={('DDX',p,b):2. for p in ['encoder','lightgbm'] for b in ['pooled','1']}
        result,_=m.calibration_panels('DDX','rq2',rs,['a','b','c'],fits)
        self.assertTrue(all(r['prediction_changes']==0 for r in result))
        self.assertEqual({r['n'] for r in result if r['view']=='full'},{2})
        self.assertEqual({r['n'] for r in result if r['view']=='cited'},{4})

    def test_controlled_family_replays_pairs_and_runs_in_processes(self):
        rs=[]
        for frac in [.25,.5,.75]:
            rs.extend(dict(r,fraction=frac,explanation_id=r['explanation_id']+str(frac)) for r in self.records())
        after,_=m.calibrate_scores(rs,['a','b','c'],2.);panels=[]
        for frac in [.25,.5,.75]:
            a=[r for r in rs if r['probe']=='encoder' and r['fraction']==frac]
            c=m.PairContrast(a,'lightgbm_absolute','tree_positive_shap','patient_id');v=c.evaluate(np.ones(c.n))
            panels.append(dict(probe='encoder',endpoint='reference',stratum='all',fraction=frac,family='paired_tree_positive_shap_vs_lightgbm_absolute',methods={'lightgbm_absolute':{},'tree_positive_shap':{}},common=4,patients=2,failures=2,pair_patients=c.n,discordant_pairs=c.pairs,information_audit={'paired_concordance_differences':{'lightgbm_absolute minus tree_positive_shap':{'difference':v,'ci95':[v,v]}}}))
        f=m.make_family('DDX','rq3',rs,after,panels);m.init_workers([f]);expected=m.bootstrap_block((0,0,12))
        ctx=m.multiprocessing.get_context('spawn')
        with m.ProcessPoolExecutor(2,mp_context=ctx,initializer=m.init_workers,initargs=([f],)) as pool:
            a=pool.submit(m.bootstrap_block,(0,0,6));b=pool.submit(m.bootstrap_block,(0,6,6));a,b=a.result(),b.result()
        np.testing.assert_equal(expected[2],np.vstack([a[2],b[2]]));np.testing.assert_equal(expected[3],np.vstack([a[3],b[3]]))

if __name__=='__main__':unittest.main()
