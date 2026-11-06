"""Finish random50 statistics from completed shards; no model loading or scoring."""
import argparse
import json
from pathlib import Path
import tarfile
import time
import traceback
from extend_rq1_random_controls import read_baseline, read, rows, dump, sha, summary


def run(a):
    lock = read(a.scored / "input_lock.json")
    files, old = read_baseline(a.source, a.dataset)
    if lock["dataset"] != a.dataset or lock["repeats"] != 50:
        raise ValueError("Scoring dataset/repeats differ")
    if lock["selection_files"] != {str(p): sha(p) for p in files}:
        raise ValueError("Original selections differ from scoring input lock")
    if lock["baseline_manifest"] != sha(a.source / "run_manifest.json"):
        raise ValueError("Original baseline manifest changed")
    folders = sorted(a.scored.glob("shard_*"))
    if not folders:
        raise ValueError("No scoring shards")
    new = []
    for folder in folders:
        manifest = read(folder / "run_manifest.json")
        if manifest["status"] != "complete":
            raise ValueError("Scoring shard incomplete: " + str(folder))
        rs = list(rows(folder / "selections_PRIVATE.jsonl"))
        if len({r["patient_id"] for r in rs}) != manifest["patients"]:
            raise ValueError("Scoring shard patient count mismatch")
        new.extend(rs)
    key = lambda r: tuple(r[k] for k in ["patient_id", "selector", "probe", "fraction"])
    baseline = {key(r): r for r in old}
    if len(new) != len(old) or len({key(r) for r in new}) != len(new) or {key(r) for r in new} != set(baseline):
        raise ValueError("Missing/duplicate scoring records")
    for r in new:
        b = baseline[key(r)]
        for field in ["selected_tokens", "available", "both_full_correct", "correct", "fact_count", "subject_group"]:
            if r.get(field) != b.get(field):
                raise ValueError("Immutable scoring field changed: " + field)
        for field in ["controls", "random_correct"]:
            actual = r[field]
            if (actual[:10] if actual is not None else None) != b[field]:
                raise ValueError("Original ten-draw prefix changed")
        if r["fraction"] != .25 and r != b:
            raise ValueError("Unextended budget changed")
    print(f"Validated {len(folders)} complete scoring shards, {len(new)} records; loading zero models", flush=True)
    dump(a.out / "input_lock.json", dict(lock, completed_scoring_source=str(a.scored),
         scored_files={str(f / "selections_PRIVATE.jsonl"): sha(f / "selections_PRIVATE.jsonl") for f in folders}))
    summary(a, old, new)
    return dict(status="complete", dataset=a.dataset, records=len(new), summary_only=True,
                scoring_reused=True, models_trained=False, models_loaded=False, api_calls=0,
                summary_workers=a.summary_workers, baseline_predictions_replayed_in_scoring=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["ddx", "trace"], default="trace")
    p.add_argument("--scored", type=Path, default=Path("artifacts/trace_rq1_random50_v2"))
    p.add_argument("--source", type=Path)
    p.add_argument("--out", type=Path, default=Path("artifacts/trace_rq1_random50_summary_v3"))
    p.add_argument("--summary-workers", type=int, default=64)
    a = p.parse_args()
    a.source = (a.source or Path("artifacts/trace2400_bce_audit_v1" if a.dataset == "trace"
                               else "artifacts/code10k_rq1_v1_retry1")).resolve()
    a.scored = a.scored.resolve(); a.out = a.out.resolve(); a.repeats = 50
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); state = {}
    try:
        state = run(a)
    except Exception:
        state = dict(status="failed", error=traceback.format_exc())
        raise
    finally:
        state["seconds"] = time.monotonic() - started
        dump(a.out / "run_manifest.json", state)
        archive = Path(str(a.out) + "_reports.tar.gz")
        with tarfile.open(archive, "w:gz") as t:
            for f in sorted(a.out.glob("*.json")):
                t.add(f, arcname=f.name)
        print("Return report:", archive, flush=True)
