import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_round1_budget_calibration import matched_job,calibration
class Contracts(unittest.TestCase):
 def test_equal_facts_changes_population_without_changing_predictions(self):
  a={'1':dict(available=True,both_full_correct=True,fact_count=2,correct=1),'2':dict(available=True,both_full_correct=True,fact_count=3,correct=0)}
  b={'1':dict(available=True,both_full_correct=True,fact_count=2,correct=0),'2':dict(available=True,both_full_correct=True,fact_count=4,correct=1)}
  r=matched_job((dict(population='both_full_correct'),a,b));self.assertEqual(r[0]['difference'],0);self.assertEqual(r[1]['difference'],1);self.assertEqual(r[1]['matched'],1);self.assertEqual(r[1]['match_fraction'],.5);self.assertIsNone(r[1]['ci_low'])
 def test_top_label_ece_and_multiclass_brier(self):
  r,b=calibration([[.8,.2],[.4,.6]],[0,0]);self.assertAlmostEqual(r['accuracy'],.5);self.assertAlmostEqual(r['ece'],.4);self.assertAlmostEqual(r['brier'],.4);self.assertEqual(sum(x['n'] for x in b),2)
  with self.assertRaises(AssertionError):calibration([[1.2,-.2]],[0])
if __name__=='__main__':unittest.main()
