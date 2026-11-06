import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from copy import deepcopy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extend_rq1_random_controls import controls, replay_patient, preserve_incomplete_panels
from intervention_control_mc import calculate
from prepare_adjudication_reference import validate


class FakeEncoding:
    def groups(self, tokens):
        return {"a": ["a1", "a2"], "b": ["b"], "c": ["c"]}


class FakeBank:
    def predict(self, inputs, context, sources):
        return {p: [np.array([.8, .2]) if "a1" in ts else np.array([.2, .8]) for ts in inputs]
                for p in sources}


class RevisionContracts(unittest.TestCase):
    def test_original_prefix_group_matching_and_replay(self):
        enc = FakeEncoding()
        groups = enc.groups(None)
        old = controls(groups, "p", .25, 2, 10, 261006)
        new = controls(groups, "p", .25, 2, 50, 261006)
        self.assertEqual(old, new[:10])
        self.assertTrue(all(len(x) == 2 and (("a1" in x) == ("a2" in x)) for x in new))
        r = dict(patient_id="p", selector="tree_local_abs_shap", probe="encoder", fraction=.25,
                 available=True, selected_tokens=["a1", "a2"], fact_count=2,
                 controls=old, random_correct=[int("a1" in x) for x in old], correct=1)
        r["random_accuracy"] = float(np.mean(r["random_correct"]))
        result = replay_patient(FakeBank(), enc, ["y", "n"],
                                dict(patient_id="p", tokens=["a1", "a2", "b", "c"], context={}, label="y"),
                                [r], 50, 261006)[0]
        self.assertEqual(result["random_correct"][:10], r["random_correct"])
        self.assertEqual(result["correct"], r["correct"])
        self.assertEqual(len(r["controls"]), 10)
        self.assertEqual(len(result["controls"]), 50)
        corrupted = dict(r, correct=0)
        with self.assertRaisesRegex(ValueError, "predictions changed"):
            replay_patient(FakeBank(), enc, ["y", "n"],
                           dict(patient_id="p", tokens=[], context={}, label="y"),
                           [corrupted], 50, 261006)

    def test_unseen_additional_draw_retains_secondary_and_rejects_primary(self):
        class SparseBank(FakeBank):
            neural = {"encoder": (None, {"a1": 1, "a2": 2})}
            def predict(self, inputs, context, sources):
                result = super().predict(inputs, context, sources)
                result["encoder"] = [None if "unknown" in ts else v
                                     for ts, v in zip(inputs, result["encoder"])]
                return result
        old = [["a1", "a2"] for _ in range(10)]
        extended = old + [["a1", "unknown"] for _ in range(40)]
        record = dict(patient_id="p", selector="tree_local_abs_shap", probe="encoder", fraction=.25,
                      available=True, both_full_correct=False, selected_tokens=["a1", "a2"],
                      fact_count=2, controls=old, random_correct=[1]*10, correct=1, random_accuracy=1.)
        patient = dict(patient_id="p", tokens=["a1", "a2", "unknown"], context={}, label="y")
        with patch("extend_rq1_random_controls.controls", side_effect=lambda *args: old if args[-2] == 10 else extended):
            row = replay_patient(SparseBank(), FakeEncoding(), ["y", "n"], patient,
                                 [record], 50, 261013, allow_secondary_unavailable=True)[0]
            self.assertEqual(row["random_correct"], [1]*10)
            self.assertTrue(row["available"])
            self.assertEqual(row["unscorable_draw_indices"], list(range(10,50)))
            self.assertEqual(row["unseen_control_tokens"], ["unknown"])
            self.assertEqual(row["attempted_random_correct_50"], [1]*10 + [None]*40)
            self.assertNotIn("extension_status", record)
            with self.assertRaisesRegex(ValueError, "protected comparison"):
                replay_patient(SparseBank(), FakeEncoding(), ["y", "n"], patient,
                               [dict(record, both_full_correct=True)], 50, 261013,
                               allow_secondary_unavailable=True)

    def test_family_fallback_preserves_denominators_and_all_method_contrasts(self):
        fields = dict(probe="encoder", fraction=.25, family="test")
        before = dict(panels=[], paired_method_differences=[])
        records = []
        for population in ["both_full_correct", "all_test"]:
            n = 1 if population == "both_full_correct" else 2
            for method in ["a", "b"]:
                before["panels"].append(dict(fields, population=population, selector=method,
                                              patients=n, gain="old", accuracy=.5))
            before["paired_method_differences"].append(dict(fields,population=population,
                left="a",right="b",patients=n,selected_accuracy_difference=0.,matched_gain_difference="old"))
        for pid, primary in [("p",True),("q",False)]:
            for method in ["a", "b"]:
                row = dict(fields,patient_id=pid,both_full_correct=primary,available=True,selector=method)
                if pid == "q" and method == "a":row["extension_status"]="unseen_encoder_state"
                records.append(row)
        after = deepcopy(before)
        for r in after["panels"]:r["gain"]="new"
        for r in after["paired_method_differences"]:r["matched_gain_difference"]="new"
        blocked,reasons=preserve_incomplete_panels(before,after,records,{"test":["a","b"]})
        self.assertEqual(blocked,{("encoder",.25,"all_test","test")})
        self.assertEqual(reasons[0]["affected_patients"],1)
        for old,new in zip(before["panels"],after["panels"]):
            self.assertEqual(old["patients"],new["patients"])
            if old["population"] == "all_test":self.assertEqual(old,new)
            else:self.assertEqual(new["gain"],"new")
        self.assertEqual(after["paired_method_differences"][1],before["paired_method_differences"][1])
        self.assertEqual(after["paired_method_differences"][0]["matched_gain_difference"],"new")

    def test_mc_keeps_patient_weighting_and_zero_variance(self):
        tasks = [dict(patient_id="a", high=1., controls=[.5]*10, unique=1),
                 dict(patient_id="a", high=1., controls=[.5]*10, unique=1),
                 dict(patient_id="b", high=.5, controls=[.5]*10, unique=1)]
        result = calculate(tasks, draws=20)
        self.assertEqual(result["patients"], 2)
        self.assertAlmostEqual(result["gain"], .25)
        self.assertEqual(result["estimated_control_mc_se"], 0.)
        np.testing.assert_allclose(result["empirical_control_resampling_range95"], [.25, .25])

    def test_adjudication_requires_actual_confirmation_and_decisions(self):
        base = dict(dataset="TRACE", explanation_id="e", audit_id="a", patient_id="p",
                    random_sample=True, targeted_sample=False, automatic_fact_ids=["f"],
                    original_reviewer_sets={"R01": {"explicit": ["f"]}, "R02": {"explicit": []}})
        final = dict(base, status="complete", final_fact_ids=["f"], common_omissions_checked=True,
                     decisions=[dict(fact_id="f", decision="include", basis="same assertion",
                                     quote="test finding", resolution="discussion")],
                     unresolved_fact_ids=[], completed_by=["R01", "R02"], adjudication_date="2026-10-20")
        policy = dict(confirmed_before_adjudication=True, confirmed_at="2026-10-20",
                      confirmed_by=["R01", "R02"], mapping_policy_text="explicit and semantic matches",
                      third_reviewer_trigger="unresolved disagreement")
        self.assertEqual(validate([base], [final], policy)["complete"], 1)
        with self.assertRaises(ValueError):
            validate([base], [final], dict(policy, confirmed_before_adjudication=False))
        with self.assertRaises(ValueError):
            validate([base], [dict(final, decisions=[])], policy)


if __name__ == "__main__":
    unittest.main()
