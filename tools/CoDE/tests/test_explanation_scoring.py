import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
import unittest
from code_eval.explanation_scoring import (score_coverage, validate_mapping,
    plan_single_group_deletions, permuted_coverage, AssociationSupport)
from code_eval.cohort_stats import Counts


class ExplanationScoring(unittest.TestCase):
    def test_interaction_and_negative_evidence(self):
        # Removing opposing evidence increases support. Do not clip R to zero.
        values = {frozenset('ab'): .6, frozenset('a'): .9,
                  frozenset('b'): .1, frozenset(): .2}
        score = score_coverage('ab', 'a', ['a', 'b'], 'D', lambda f,d: values[f])
        self.assertAlmostEqual(score['retained_support'], .3)
        self.assertEqual(score['weighted_coverage'], 1)
        self.assertLess(score['signed_contributions'][1], 0)

    def test_redundant_support_is_not_positive_removal_weight(self):
        score = score_coverage('ab', 'a', ['a', 'b'], 'D', lambda f,d: .9)
        self.assertIsNone(score['weighted_coverage'])
        self.assertEqual(score['retained_support'], 0)
        self.assertEqual(score['weighted_na_reason'], 'no_positive_support_weight')

    def test_group_fraction_and_empty_policy(self):
        calls = []
        def q(f,d):
            calls.append(f)
            return .2 + .1*len(f)
        score = score_coverage('abc', 'a', ['ab', 'c'], 'D', q)
        self.assertAlmostEqual(score['weighted_coverage'], 1/3)
        empty = score_coverage('ab', [], ['ab'], 'D', q)
        self.assertIsNone(empty['retained_support'])
        self.assertNotIn(frozenset(), calls)

    def test_mapping_and_leakage_boundaries(self):
        for facts,cited,groups in [('ab','x',['a','b']),('ab','a',['ab','b']),
                                   ('ab','aa',['a','b'])]:
            with self.assertRaises(ValueError): validate_mapping(facts,cited,groups)
        with self.assertRaises(ValueError):
            score_coverage('a','a',['a'],'D',lambda f,d: float('nan'))

    def test_edit_plan_keeps_dependency_group_together(self):
        plans = plan_single_group_deletions('abc','abc',['ab','c'])
        self.assertEqual(plans[0]['expected_cited_fact_ids'], ['c'])
        self.assertEqual(plans[0]['remove_fact_ids'], ['a','b'])
        self.assertEqual(len(plans), 2)

    def test_permutation_shared_and_repetition_invariant(self):
        kwargs = dict(patient_id='p', diagnosis='d', donor_id='cohd', seed=42)
        self.assertEqual(permuted_coverage([1,2,3],[1,0,0],**kwargs),
                         permuted_coverage([1,2,3],[1,0,0],**kwargs))
        self.assertEqual(permuted_coverage([1,2,3],[1,1,1],**kwargs), 1)

    def test_real_association_adapter(self):
        counts = Counts(['A','B'])
        counts.add(['x'], 'A'); counts.add(['y'], 'B')
        q = AssociationSupport(counts, {'a':'x','b':'y'})
        self.assertGreater(q(frozenset('a'),'A'), q(frozenset('b'),'A'))
        score = score_coverage('ab','a',['a','b'],'A',q,empty_supported=True)
        self.assertGreater(score['retained_support'], 0)
        with self.assertRaises(ValueError): q(frozenset('a'),'gold_unknown')


if __name__ == '__main__': unittest.main()
