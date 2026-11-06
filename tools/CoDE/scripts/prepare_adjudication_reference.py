"""Prepare/validate finite-audit adjudication transfer records. Never infer decisions."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_trace2400_sources import read, rows, dump


def templates(root):
    output = []
    first = rows(root / "review102_policies_20261007/review_sets.jsonl")
    second = {r["explanation_id"]: r for r in rows(root / "review102_R02_policies_20261007/review_sets.jsonl")}
    if len(first) != 102 or len(second) != 102 or {r["explanation_id"] for r in first} != set(second):
        raise ValueError("DDXPlus reviewer census differs")
    for r in first:
        s = second[r["explanation_id"]]
        for k in ["audit_id", "patient_id", "random_sample", "targeted_failure", "automatic_fact_ids"]:
            if r[k] != s[k]:
                raise ValueError("DDXPlus reviewer join differs: "+k)
        output.append(dict(dataset="DDXPlus", audit_id=r["audit_id"], explanation_id=r["explanation_id"],
            patient_id=r["patient_id"], random_sample=r["random_sample"], targeted_sample=r["targeted_failure"],
            automatic_fact_ids=r["automatic_fact_ids"],
            original_reviewer_sets={who: {k: item[k] for k in ["conservative_fact_ids", "inclusive_fact_ids"]}
                                    for who, item in [("R01", r), ("R02", s)]}))
    trace = rows(root / "trace2400_review100_20261015/review_sets_PRIVATE.jsonl")
    if len(trace) != 100 or len({r["explanation_id"] for r in trace}) != 100:
        raise ValueError("TRACE reviewer census differs")
    for r in trace:
        output.append(dict(dataset="TRACE", audit_id=r["audit_id"], explanation_id=r["explanation_id"],
            patient_id=r["patient_id"], subject_group=r["subject_group"],
            random_sample=r["random_sample"], targeted_sample=r["targeted_sample"],
            automatic_fact_ids=r["automatic_fact_ids"],
            original_reviewer_sets={who: {policy: r[who+"_"+policy] for policy in ["explicit", "compatible"]}
                                    for who in ["R01", "R02"]}))
    for r in output:
        r.update(status="pending", final_fact_ids=None, common_omissions_checked=False,
                 decisions=[], unresolved_fact_ids=[], completed_by=[], adjudication_date=None)
    return output


def validate(original, final, policy):
    if not policy.get("confirmed_before_adjudication") or not policy.get("confirmed_at") or not policy.get("confirmed_by"):
        raise ValueError("Human confirmation of the policy is missing")
    if not policy.get("mapping_policy_text") or not policy.get("third_reviewer_trigger"):
        raise ValueError("Mapping rules and third-reviewer trigger must be specified")
    policy_date = date.fromisoformat(policy["confirmed_at"][:10])
    if len(set(policy["confirmed_by"])) < 2:
        raise ValueError("Both reviewers must confirm the adjudication policy")
    key = lambda r: (r["dataset"], r["explanation_id"])
    ix = {key(r): r for r in final}
    if len(ix) != len(final) or set(ix) != {key(r) for r in original}:
        raise ValueError("Frozen audit sample changed")
    count = {"complete": 0, "unresolved": 0, "pending": 0}
    for base in original:
        r = ix[key(base)]
        for k in ["audit_id", "patient_id", "random_sample", "targeted_sample",
                  "automatic_fact_ids", "original_reviewer_sets"]:
            if r[k] != base[k]:
                raise ValueError("Original review/sample information changed: "+k)
        if "subject_group" in base and r["subject_group"] != base["subject_group"]:
            raise ValueError("Cluster changed")
        if r["status"] not in count:
            raise ValueError("Unknown adjudication status")
        count[r["status"]] += 1
        if r["status"] == "pending":
            if r["final_fact_ids"] is not None:
                raise ValueError("Pending record cannot carry an accepted final set")
            continue
        if not r["common_omissions_checked"] or not r["completed_by"] or not r["adjudication_date"]:
            raise ValueError("Adjudication process is incomplete")
        if len(set(r["completed_by"])) < 2 or date.fromisoformat(r["adjudication_date"][:10]) < policy_date:
            raise ValueError("Two-reviewer completion/date inconsistent with policy confirmation")
        fs = r["final_fact_ids"]
        if not isinstance(fs, list) or len(fs) != len(set(fs)):
            raise ValueError("Final facts must be a unique list (empty only after completed review)")
        if set(fs) & set(r["unresolved_fact_ids"]):
            raise ValueError("Unresolved facts cannot enter the final reference")
        if r["status"] == "complete" and r["unresolved_fact_ids"]:
            raise ValueError("Complete record has unresolved facts")
        candidates = set(base["automatic_fact_ids"]) | set(fs) | set(r["unresolved_fact_ids"])
        for reviewer in base["original_reviewer_sets"].values():
            for ids in reviewer.values():
                candidates.update(ids)
        decisions = {d["fact_id"]: d for d in r["decisions"]}
        if len(decisions) != len(r["decisions"]) or not candidates <= decisions.keys():
            raise ValueError("Per-fact decisions do not cover the reviewed candidate union")
        for fid, d in decisions.items():
            if d.get("decision") not in ["include", "exclude", "unresolved"] or not d.get("basis"):
                raise ValueError("Missing decision/basis")
            if (d["decision"] == "include") != (fid in fs):
                raise ValueError("Decision and final reference differ")
            if (d["decision"] == "unresolved") != (fid in r["unresolved_fact_ids"]):
                raise ValueError("Unresolved decision list differs")
            if d.get("resolution") not in ["agreement", "discussion", "third_reviewer"]:
                raise ValueError("Resolution route required")
            if d["decision"] == "include" and not d.get("quote"):
                raise ValueError("Included fact needs a supporting original-text quote")
            if d["resolution"] == "third_reviewer" and not d.get("reviewer_id"):
                raise ValueError("Third-reviewer identity is missing")
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    p.add_argument("--out", type=Path)
    p.add_argument("--validate", type=Path)
    p.add_argument("--policy", type=Path)
    a = p.parse_args()
    original = templates(a.inputs)
    if a.validate:
        if not a.policy:
            p.error("--policy required with --validate")
        result = validate(original, rows(a.validate), read(a.policy))
        print(json.dumps(dict(status="structure_checked", **result,
                              note="Does not verify semantic decisions, quotes or patient-fact membership; those require source records.")))
    else:
        if not a.out:
            p.error("--out required")
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out / "adjudication_reference_PENDING_PRIVATE.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in original), encoding="utf8")
        dump(a.out / "policy_PENDING.json", dict(
            confirmed_before_adjudication=False, confirmed_at=None, confirmed_by=[],
            mapping_policy_text=None, third_reviewer_trigger=None,
            scope="Same reviewed samples; original F and automatic E; no outcome-driven changes"))
        print("Prepared", len(original), "pending records; no decisions inferred.")
