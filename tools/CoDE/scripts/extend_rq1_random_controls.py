"""Extend saved 25% RQ1 controls to 50; retain selections, fits and first 10 draws."""
import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_trace2400_sources import read, rows, dump, sha
from evaluate_saved_subsets import matched_groups


def controls(groups, pid, fraction, count, repeats, seed):
    return [matched_groups(groups, count, random.Random(
        f"{seed}|{pid}|{fraction}|{count}|{j}")) for j in range(repeats)]


def read_baseline(source, dataset):
    folder = source if dataset == "ddx" else source / "rq1"
    manifest = read(source / "run_manifest.json")
    required_status = "complete" if dataset == "ddx" else "complete_gates_passed"
    if manifest["status"] != required_status:
        raise ValueError("Baseline run incomplete")
    files = ([folder / "selections_PRIVATE.jsonl"] if dataset == "trace"
             else sorted(folder.glob("shard_*/selections.jsonl")))
    if not files:
        raise FileNotFoundError(f"No saved selection shards in {folder}")
    records = []
    for f in files:
        if dataset == "ddx" and read(f.parent / "run_manifest.json")["status"] != "complete":
            raise ValueError("Baseline shard incomplete")
        expected = manifest.get("output_sha256", {}).get(f.relative_to(source).as_posix())
        if expected and sha(f) != expected:
            raise ValueError("Baseline selections changed")
        records.extend(rows(f))
    keys = {(r["patient_id"], r["selector"], r["probe"], r["fraction"]) for r in records}
    expected_count = (1506 * 10 if dataset == "ddx" else 359 * 9) * 3 * 2
    if len(keys) != len(records) or len(records) != expected_count:
        raise ValueError("Incomplete/duplicate baseline selection census")
    return files, records


def setup(a):
    import torch
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original_lock = read(a.source / "evaluation_lock.json")
    if sha(a.models / "fitted_sources.json") != original_lock["fitted_manifest_sha256"]:
        raise ValueError("Models differ from the original RQ1 evaluation")
    if a.dataset == "ddx":
        from run_code10k_rq1 import verify
        from run_code10k_sources import structural_record
        from prepare_code10k import unique
        from code_eval.tree_subset import Encoding
        from code_eval.frozen_bank import FrozenBank
        fm, _, dictionary = verify(a)
        enc = Encoding(dictionary)
        paths = {n: a.models / p for n, p in {
            "lightgbm": "source/tree/mask_augmented.txt", "encoder": "encoder/encoder.pt",
            "grasp": "formal/grasp.pt", "cohd": "formal/cohd_counts.json"}.items()}
        bank = FrozenBank(paths, enc, fm["labels"], a.threads)
        split = unique(a.protocol / "patient_split_PRIVATE.jsonl", "patient_id")
        patients = unique(a.sample / "data/patients_public.jsonl", "patient_id")
        query = {pid: structural_record(patients[pid], s)
                 for pid, s in split.items() if s["partition"] == "test"}
        return bank, enc, fm["labels"], query
    from run_trace2400_evaluation import inputs
    from code_eval.trace2400.bank import TraceBank
    if sha(a.view / "test_records_PRIVATE.jsonl") != original_lock["query_sha256"]:
        raise ValueError("TRACE facts differ from the original RQ1 evaluation")
    train, labels, enc, paths, split, query = inputs(a)
    bank = TraceBank(paths, read(a.view / "evidence_dictionary.json"), labels,
                     train, list(query.values()), {p: s["subject_group"] for p, s in split.items()},
                     a.threads)
    return bank, enc, labels, query


def replay_patient(bank, enc, labels, patient, records, repeats, seed,
                   allow_secondary_unavailable=False):
    """Use legacy batching for replay, then score only unseen additional subsets."""
    pid = patient["patient_id"]
    groups = enc.groups(patient["tokens"])
    keys = set()
    for r in records:
        if r["selected_tokens"] is None:
            continue
        old = controls(groups, pid, r["fraction"], r["fact_count"], 10, seed)
        if old != r["controls"]:
            raise ValueError("Original control draws do not replay")
        keys.add(tuple(r["selected_tokens"]))
        keys.update(map(tuple, old))
    ordered = sorted(keys)
    if not ordered:
        return records
    values = bank.predict([list(k) for k in ordered], patient["context"],
                          sources=["encoder", "lightgbm"])
    lookup = {p: dict(zip(ordered, v)) for p, v in values.items()}

    def correct(probe, ts):
        v = lookup[probe][tuple(ts)]
        return None if v is None else int(labels[int(v.argmax())] == patient["label"])

    for r in records:
        if r["selected_tokens"] is None:
            continue
        c = correct(r["probe"], r["selected_tokens"])
        rc = [correct(r["probe"], ts) for ts in r["controls"]]
        available = c is not None and all(v is not None for v in rc)
        if available != r["available"]:
            raise ValueError("Original probe availability changed")
        if available and (c != r["correct"] or rc != r["random_correct"]):
            raise ValueError("Original probe predictions changed; preserve original math policy")
        if available and not np.isclose(np.mean(rc), r["random_accuracy"], atol=1e-12, rtol=0):
            raise ValueError("Original random accuracy inconsistent")

    extended = {}
    for r in records:
        if r["fraction"] == .25 and r["available"]:
            n = r["fact_count"]
            extended[n] = controls(groups, pid, .25, n, repeats, seed)
    new = sorted({tuple(k) for cs in extended.values() for k in cs} - keys)
    if new:
        values = bank.predict([list(k) for k in new], patient["context"],
                              sources=["encoder", "lightgbm"])
        for p, vs in values.items():
            lookup[p].update(zip(new, vs))
    output = []
    for r in records:
        updated = dict(r)
        if r["fraction"] == .25 and r["available"]:
            cs = extended[r["fact_count"]]
            rc = [correct(r["probe"], ts) for ts in cs]
            if any(v is None for v in rc):
                # TRACE full-input-ineligible patients may have subsets that happen to
                # avoid unseen states in ten draws but encounter them in fifty.
                if (not allow_secondary_unavailable or r["both_full_correct"]
                        or r["probe"] != "encoder"):
                    raise ValueError(f"Additional control unscorable in protected comparison: {pid}")
                vocab = bank.neural["encoder"][1]
                failed = [j for j, v in enumerate(rc) if v is None]
                if any(j < 10 or not any(t not in vocab for t in cs[j]) for j in failed):
                    raise ValueError("Unscorable control is not explained by unseen encoder states")
                updated.update(random_repeats=10, extension_status="unseen_encoder_state",
                               attempted_controls_50=cs, attempted_random_correct_50=rc,
                               unscorable_draw_indices=failed,
                               unseen_control_tokens=sorted({t for j in failed for t in cs[j] if t not in vocab}))
                output.append(updated)
                continue
            if rc[:10] != r["random_correct"] or cs[:10] != r["controls"]:
                raise ValueError("First ten controls changed")
            updated.update(controls=cs, random_correct=rc, random_accuracy=float(np.mean(rc)),
                           random_repeats=repeats, unique_control_sets=len(set(map(tuple, cs))))
        output.append(updated)
    return output


def worker(a):
    files, old = read_baseline(a.source, a.dataset)
    lock = read(a.out / "input_lock.json")
    if {str(p): sha(p) for p in files} != lock["selection_files"]:
        raise ValueError("Baseline selection files changed after lock")
    bank, enc, labels, query = setup(a)
    by = defaultdict(list)
    for r in old:
        by[r["patient_id"]].append(r)
    if set(by) != set(query):
        raise ValueError("Selection/query population mismatch")
    folder = a.out / f"shard_{a.shard}"
    folder.mkdir()
    selected = sorted(query)[a.shard::a.shards]
    with (folder / "selections_PRIVATE.jsonl").open("w", encoding="utf8") as out:
        for i, pid in enumerate(selected):
            if a.dataset == "trace" and any(r["subject_group"] != query[pid]["subject_group"] for r in by[pid]):
                raise ValueError("Subject cluster changed")
            rs = replay_patient(bank, enc, labels, query[pid], by[pid], a.repeats,
                                261006 if a.dataset == "ddx" else 261013,
                                allow_secondary_unavailable=a.dataset == "trace")
            for r in rs:
                out.write(json.dumps(r, allow_nan=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"Shard {a.shard}: {i+1}/{len(selected)} patients", flush=True)
    dump(folder / "run_manifest.json", dict(status="complete", patients=len(selected)))



def preserve_incomplete_panels(before, after, records, families):
    """Use ten draws for an entire common-family comparison if any member lacks 50.

    All patients and selectors remain present. Never pool ten- and fifty-draw
    means in a published panel, and never remove an unscorable draw.
    """
    fields = ("probe", "fraction", "population", "family")
    key = lambda r: tuple(r[f] for f in fields)
    blocked = set()
    reasons = []
    for k in sorted({key(r) for r in before["panels"]}):
        probe, fraction, population, family = k
        if fraction != .25:
            continue
        rs = [r for r in records if r["probe"] == probe and r["fraction"] == fraction
              and r["available"] and (population == "all_test" or r["both_full_correct"])]
        ids = set.intersection(*[{r["patient_id"] for r in rs if r["selector"] == m}
                                for m in families[family]])
        failed = [r for r in rs if r["patient_id"] in ids and r["selector"] in families[family]
                  and r.get("extension_status") == "unseen_encoder_state"]
        if failed:
            if population != "all_test":
                raise ValueError("Primary analysis must retain fifty scorable controls")
            blocked.add(k)
            reasons.append(dict(zip(fields, k), retained_draws=10,
                                affected_patients=len({r["patient_id"] for r in failed}),
                                affected_patient_selector_records=len(failed),
                                reason="Additional draws include unseen encoder states; entire family retains original ten draws."))
    for collection, identifiers in [("panels", ("selector",)),
                                    ("paired_method_differences", ("left", "right"))]:
        rowkey = lambda r: key(r) + tuple(r[f] for f in identifiers)
        original = {rowkey(r): r for r in before[collection]}
        after[collection] = [dict(original[rowkey(r)]) if key(r) in blocked else r
                             for r in after[collection]]
    return blocked, reasons


def summary(a, old, new):
    mod = importlib.import_module("code_eval.rq1_summary" if a.dataset == "ddx"
                                  else "code_eval.trace2400.rq1_summary")
    from code_eval.parallel_rq1_summary import summarize_pair
    before, after = summarize_pair(old, new, a.dataset, workers=getattr(a, "summary_workers", 64))
    from code_eval.trace2400.bce_audit import equal
    # Verify that the saved census reproduces the accepted formal summary.
    equal(before, read(a.source / "rq1_summary.json"), "formal RQ1 replay", 1e-10)
    print("Summary: original summary replay passed; checking availability and Monte Carlo blocks", flush=True)
    blocked, retained = preserve_incomplete_panels(before, after, new, mod.FAMILIES)
    def key(r):
        return tuple(r[k] for k in ["probe", "fraction", "population", "family", "selector"])
    baseline = {key(r): r for r in before["panels"]}
    panels = []
    for r in after["panels"]:
        b = baseline[key(r)]
        if r["patients"] != b["patients"] or r["accuracy"] != b["accuracy"]:
            raise ValueError("Selected accuracy or population changed")
        if r["fraction"] != .25:
            if r != b:
                raise ValueError("Unextended budget changed")
            continue
        if tuple(r[k] for k in ["probe", "fraction", "population", "family"]) in blocked:
            panels.append(dict(**{k: r[k] for k in ["probe", "fraction", "population", "family", "selector", "patients"]},
                               gain_10=b["gain"], gain_50=None, gain_change=None,
                               status="retained_original_ten", ten_draw_block_gains=[], mean_unique_sets=None))
            continue
        # Ten-repeat blocks show Monte Carlo variation, conditional on fixed patients/models.
        ids = {x["patient_id"] for x in new if x["probe"] == r["probe"] and x["fraction"] == .25
               and x["available"] and (r["population"] == "all_test" or x["both_full_correct"])}
        for m in mod.FAMILIES[r["family"]]:
            ids &= {x["patient_id"] for x in new if x["probe"] == r["probe"] and x["fraction"] == .25
                    and x["selector"] == m and x["available"]
                    and (r["population"] == "all_test" or x["both_full_correct"])}
        chosen = [x for x in new if x["patient_id"] in ids and x["selector"] == r["selector"]
                  and x["probe"] == r["probe"] and x["fraction"] == .25]
        if len(chosen) != r["patients"]:
            raise ValueError("Monte Carlo/common-family population differs")
        if any(len(x["random_correct"]) != a.repeats for x in chosen):
            raise ValueError("Mixed control counts in an expanded comparison")
        blocks = [float(np.mean([x["correct"] - np.mean(x["random_correct"][j:j+10])
                                 for x in chosen])) for j in range(0, a.repeats, 10)] if chosen else []
        panels.append(dict(**{k: r[k] for k in ["probe", "fraction", "population", "family", "selector", "patients"]},
                           gain_10=b["gain"], gain_50=r["gain"],
                           gain_change=None if not chosen else r["gain"]["mean"]-b["gain"]["mean"],
                           ten_draw_block_gains=blocks,
                           mean_unique_sets=float(np.mean([x["unique_control_sets"] for x in chosen])) if chosen else None))
    for b, r in zip(before["paired_method_differences"], after["paired_method_differences"]):
        if b["selected_accuracy_difference"] != r["selected_accuracy_difference"]:
            raise ValueError("Direct method contrast changed")
    after["interval_unit"] = ("inferred-subject cluster" if a.dataset == "trace" else "patient") + \
        "; fixed fitted models and realized controls; 50 draws at 25% except explicitly retained all-test families; 10 at other budgets"
    after["retained_ten_draw_families"] = retained
    dump(a.out / "rq1_summary.json", after)
    failed = [r for r in new if r.get("extension_status") == "unseen_encoder_state"]
    dump(a.out / "extension_availability.json", dict(
        status="complete", primary_comparisons_fully_extended=True,
        affected_patients=len({r["patient_id"] for r in failed}),
        affected_records=len(failed), retained_ten_draw_families=retained,
        patient_census_unchanged=True, unavailable_counted_as_incorrect=False,
        redraw_until_scorable=False,
        publication_rule="Use rq1_summary.json; retained all-test families reproduce original ten-draw aggregates, not mixed-count estimates."))
    dump(a.out / "random_stability.json", dict(panels=panels,
         direct_method_contrasts_unchanged=True, other_budgets_unchanged=True,
         retained_ten_draw_families=retained,
         monte_carlo_scope="Five ten-draw blocks describe random-control variability, not new patient samples.",
         replacement="Original uniform sampler with replacement across draws; unique sets reported separately."))


def main(a):
    for module in ["torch", "lightgbm", "numpy"]:
        importlib.import_module(module)
    files, old = read_baseline(a.source, a.dataset)
    dump(a.out / "input_lock.json", dict(dataset=a.dataset, selection_files={str(p): sha(p) for p in files},
         baseline_manifest=sha(a.source / "run_manifest.json"), repeats=a.repeats, fraction=.25,
         models_refit=False, new_attribution=False, prefix_retained=10,
         unscorable_policy="Primary comparisons require all 50; an affected all-test family retains its full original ten-draw comparison."))
    procs = []
    try:
        for i, gpu in enumerate(a.gpus.split(",")):
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--dataset", a.dataset,
                   "--shard", str(i), "--shards", str(len(a.gpus.split(","))),
                   "--threads", str(a.threads), "--repeats", str(a.repeats)]
            for name in ["source", "models", "protocol", "sample", "config", "view", "out"]:
                cmd.extend(["--"+name, str(getattr(a, name))])
            log = (a.out / f"worker_{i}.log").open("w", encoding="utf8")
            procs.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)), log))
        while any(p.poll() is None for p, _ in procs):
            print("Random-control scoring", [(i, p.poll()) for i, (p, _) in enumerate(procs)], flush=True)
            if any(p.poll() not in (None, 0) for p, _ in procs):
                raise RuntimeError("Worker failed; inspect worker logs")
            time.sleep(10)
        if any(p.returncode for p, _ in procs):
            raise RuntimeError("Worker failed; inspect worker logs")
    finally:
        for p, log in procs:
            if p.poll() is None:
                p.terminate()
                p.wait()
            log.close()
    new = []
    for i in range(len(procs)):
        folder = a.out / f"shard_{i}"
        if read(folder / "run_manifest.json")["status"] != "complete":
            raise ValueError("Incomplete new shard")
        new.extend(rows(folder / "selections_PRIVATE.jsonl"))
    if len(new) != len(old):
        raise ValueError("Selection census changed")
    row_key = lambda r: (r["patient_id"], r["selector"], r["probe"], r["fraction"])
    if len({row_key(r) for r in new}) != len(new) or {row_key(r) for r in new} != {row_key(r) for r in old}:
        raise ValueError("Missing/duplicate selection keys")
    print("Summarizing matched comparisons and Monte Carlo blocks", flush=True)
    summary(a, old, new)
    return dict(status="complete", dataset=a.dataset, records=len(new), models_trained=False,
                api_calls=0, baseline_predictions_replayed=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["ddx", "trace"], required=True)
    defaults = dict(source=None, models=None, protocol="artifacts/code10k_protocol_v1",
                    sample="inputs/ddxplus_sample",
                    config=str(ROOT / "configs/code10k_rq1_20261007.json"),
                    view="artifacts/trace2400_view_v2_names", out=None)
    for k, v in defaults.items():
        p.add_argument("--"+k, type=Path, default=Path(v) if v else None)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--summary-workers", type=int, default=64)
    p.add_argument("--repeats", type=int, default=50, choices=[50])
    p.add_argument("--shard", type=int)
    p.add_argument("--shards", type=int)
    a = p.parse_args()
    a.partition = "test"
    ddx = a.dataset == "ddx"
    a.source = a.source or Path("artifacts/code10k_rq1_v1_retry1" if ddx else "artifacts/trace2400_bce_audit_v1")
    a.models = a.models or Path("artifacts/code10k_sources_v1" if ddx else "artifacts/trace2400_extended_sources_v1")
    a.out = a.out or Path("artifacts/"+a.dataset+"_rq1_random50")
    for k in defaults:
        setattr(a, k, getattr(a, k).resolve())
    if len(set(a.gpus.split(","))) != len(a.gpus.split(",")):
        p.error("Distinct GPU/worker slots required")
    if a.shard is not None:
        worker(a)
    else:
        a.out.mkdir(parents=True, exist_ok=False)
        state = {}
        try:
            state = main(a)
        except Exception:
            state = dict(status="failed", error=traceback.format_exc())
            raise
        finally:
            dump(a.out / "run_manifest.json", state)
            archive = Path(str(a.out)+"_reports.tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                for f in a.out.glob("*"):
                    if f.is_file() and (f.suffix == ".json" or f.name.startswith("worker_")):
                        tar.add(f, arcname=f.name)
            print("Return report:", archive, flush=True)
