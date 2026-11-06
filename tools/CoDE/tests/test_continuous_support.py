import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_continuous_support import make_pairs,summarize,METHODS

def row(pid,token,q):
 return dict(patient_id=pid,diagnosis='d',reference_diagnosis='d',cited_fact_count=1,cited_tokens=[token],mapping_available=True,target_supported=True,probe_available=True,both_full_correct=True,reference_failure=False,cited_probabilities=[q,1-q],scores={m:(q if m!='fact_coverage' else .5) for m in METHODS})
class ContinuousTests(unittest.TestCase):
 def test_identity_dedup_common_and_empty(self):
  a=row('p','a',.6);b=row('p','b',.8);pairs,audit=make_pairs([a,a,b],['d','y'])
  self.assertEqual(len(pairs),1);self.assertEqual(audit['counts']['duplicate_sets_removed'],1)
  r=summarize(pairs,'q',.01,'both_correct',20)
  self.assertEqual(r['methods']['lightgbm_absolute']['mean'],1)
  self.assertEqual(r['methods']['fact_coverage']['mean'],.5)
  self.assertIsNone(r['methods']['lightgbm_absolute']['ci95'])
  b['scores']['cohd_absolute']=None;pairs,_=make_pairs([a,b],['d','y']);r=summarize(pairs,'q',.01,'all',20)
  self.assertEqual(r['endpoint_eligible_pairs'],1);self.assertEqual(r['common_pairs'],0);self.assertIsNone(r['methods']['group_coverage']['mean'])
 def test_patient_equal_weight_not_pair_weight(self):
  base=dict(common=True,both_correct=True,q=.1,margin=.2,qy=.1,reference_recovery=0)
  correct=dict(base,patient_id='one',scores=[0]+[.1]*6)
  incorrect=dict(base,patient_id='two',scores=[0]+[-.1]*6)
  r=summarize([correct]*9+[incorrect],'q',.01,'all',20)
  self.assertEqual(r['patients'],2);self.assertEqual(r['methods']['cohd_coverage']['mean'],.5)
 def test_tiny_endpoint_not_eligible(self):
  a=row('p','a',.6);b=row('p','b',.600000001);pairs,_=make_pairs([a,b],['d','y'])
  self.assertEqual(summarize(pairs,'q',.01,'all',0)['patients'],0)
  self.assertEqual(summarize(pairs,'q',1e-10,'all',0)['patients'],1)
if __name__=='__main__':unittest.main()
