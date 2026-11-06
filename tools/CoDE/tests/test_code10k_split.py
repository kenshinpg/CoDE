import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_code10k import grouped_split

class GroupedSplitTests(unittest.TestCase):
    def test_disjoint_groups_and_order_invariance(self):
        data=[dict(patient_id=str(i),pattern_id=str(i//2),disease=str((i//2)%4),component=['coverage','challenge'][i%2]) for i in range(200)]
        a=grouped_split(data,9);b=grouped_split(list(reversed(data)),9)
        self.assertEqual(a,b)
        for i in range(0,200,2):self.assertEqual(a[str(i)],a[str(i+1)])
        self.assertEqual(set(a.values()),{'train','development','test'})
        self.assertEqual({r['disease'] for r in data if a[r['patient_id']]=='train'},{r['disease'] for r in data})

    def test_extraction_and_generated_outcome_do_not_affect_split(self):
        data=[dict(patient_id=str(i),pattern_id=str(i),disease=str(i%3),component='coverage') for i in range(60)]
        altered=[dict(r,reviewed_available=False,diagnosis_correct=False) for r in data]
        self.assertEqual(grouped_split(data,7),grouped_split(altered,7))

    def test_conflicting_labels_same_pattern_cannot_cross(self):
        data=[dict(patient_id='a',pattern_id='one',disease='A',component='coverage'),dict(patient_id='b',pattern_id='one',disease='B',component='challenge')]
        split=grouped_split(data,5);self.assertEqual(split['a'],split['b'])

if __name__=='__main__':unittest.main()
