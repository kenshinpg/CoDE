import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from code_eval.subset_selectors import (EvidenceSupport, leave_group_out,
    sis_backward_path, evidence_support_record, ranked_subset)


class SubsetMechanisms(unittest.TestCase):
    def test_sis_traverses_nonmonotone_path(self):
        # All one-deletion subsets fail, but each one-fact subset is sufficient.
        # Stopping at the first failure would miss the smaller sufficient set.
        p = {0: .1, 1: .8, 2: .6, 3: .9}
        s = EvidenceSupport(['a','b','c'], {'a':['a'],'b':['b'],'c':['c']},
                            ['D','X'], lambda fs: {'D':p[len(fs)], 'X':1-p[len(fs)]}, allow_empty=True)
        result = sis_backward_path(s, .7)
        self.assertEqual(result['selected']['group_count'], 1)
        self.assertEqual(len(result['deletion_path']), 3)
        self.assertEqual(result['target'], 'D')

    def test_redundancy_and_signed_opposition(self):
        def q(fs):
            p = .1 + .7*bool(fs & {'a','b'}) - .05*('c' in fs)
            return {'D':p,'X':1-p}
        s = EvidenceSupport(['a','b','c'], {'a':['a'],'b':['b'],'c':['c']}, ['D','X'], q, allow_empty=True)
        r = leave_group_out(s)
        self.assertAlmostEqual(r['signed_effects']['a'], 0)
        self.assertAlmostEqual(r['signed_effects']['b'], 0)
        self.assertLess(r['signed_effects']['c'], 0)
        self.assertEqual(sis_backward_path(s, .7)['selected']['group_count'], 1)

    def test_equal_low_support_is_not_accepted_as_sufficient(self):
        s = EvidenceSupport(['a','b'], {'a':['a'],'b':['b']}, ['D','X'],
                            lambda fs:{'D':.01,'X':.99})
        r = evidence_support_record(s, ['a'], 'D', eligibility_threshold=.5)
        self.assertEqual(r['support_loss'], 0)
        self.assertEqual(r['eligibility_status'], 'LOW_FULL_SUPPORT')
        self.assertIsNone(r['clinical_omission_label'])
        self.assertEqual(s.source_target(), 'X')

    def test_schema_empty_and_distribution_barriers(self):
        s = EvidenceSupport(['a'], {'g':['a']}, ['D','X'], lambda fs:{'D':.9,'X':.1})
        with self.assertRaisesRegex(ValueError, 'validated'):
            sis_backward_path(s,.8)
        with self.assertRaisesRegex(ValueError, 'Unknown patient'):
            s.distribution(['invented'])
        with self.assertRaises(ValueError):
            ranked_subset(s,['g','g'],1)
        bad = EvidenceSupport(['a'], {'g':['a']}, ['D','X'], lambda fs:{'D':.9,'X':.9})
        with self.assertRaisesRegex(ValueError, 'normalized'):
            bad.distribution(['a'])


if __name__ == '__main__':
    unittest.main()
