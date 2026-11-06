"""Conditional Monte Carlo sensitivity of existing exploratory SHAP edit controls.

Reuses the ten saved alternatives. No new patients, predictions, or random edits
are claimed. Covers the correct-proposal support-direction contrasts in the abstract.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_trace2400_sources import read, rows, dump, sha
from analyze_code10k_intervention_effects import effect


def calculate(tasks, seed=261020, draws=5000):
    """Equal-patient weighting; empirical resampling of controls within each task."""
    by_patient = defaultdict(list)
    for t in tasks:
        by_patient[t["patient_id"]].append(t)
    if not by_patient:
        return dict(patients=0, explanations=0, gain=None)
    ts = [t for pid in sorted(by_patient) for t in by_patient[pid]]
    weights = np.array([1 / len(by_patient) / len(by_patient[t["patient_id"]]) for t in ts])
    high = np.array([t["high"] for t in ts])
    control = np.asarray([t["controls"] for t in ts], dtype=float)
    if control.ndim != 2 or control.shape[1] != 10 or not np.isfinite(control).all():
        raise ValueError("Expected ten finite saved alternatives per explanation")
    gain = float(weights @ (high - control.mean(1)))
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, draws, 64):
        ix = rng.integers(0, 10, (min(64, draws-start), len(ts), 10))
        samples = control[np.arange(len(ts))[None, :, None], ix].mean(2)
        estimates.extend((weights @ high - samples @ weights).tolist())
    # The original random draws are independent across tasks, even within patient.
    se = float(np.sqrt(np.sum(weights**2 * control.var(1, ddof=1) / 10)))
    leave_one = [float(weights @ (high - np.delete(control, j, axis=1).mean(1))) for j in range(10)]
    return dict(patients=len(by_patient), explanations=len(ts), gain=gain,
                estimated_control_mc_se=se,
                empirical_control_resampling_range95=np.quantile(estimates, [.025, .975]).tolist(),
                empirical_control_resampling_draws=draws,
                leave_one_draw_out_gains=leave_one,
                unique_alternatives_per_task=dict(
                    minimum=min(t["unique"] for t in ts),
                    median=float(np.median([t["unique"] for t in ts])),
                    maximum=max(t["unique"] for t in ts)),
                interval_scope="Control resampling conditional on fixed patients, models and ten observed alternatives; not a patient CI or new-draw validation.")


def main(a):
    manifest = read(a.source / "scoring_manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Incomplete intervention scoring")
    for name, expected in manifest["output_sha256"].items():
        if sha(a.source / name) != expected:
            raise ValueError("Original intervention file changed: "+name)
    cfg = read(a.source / "protocol_lock.json")["config"]
    if cfg["random_repeats"] != 10:
        raise ValueError("Expected original ten-draw controls")
    grouped = defaultdict(list)
    for r in rows(a.source / "interventions.jsonl"):
        if r["selector"] == "local_shap" and r["direction"] == "support" and r["diagnosis_correct"]:
            grouped[r["operation"], r["explanation_id"]].append(r)
    panels = {}
    for op in ["add", "delete"]:
        for probe in ["encoder", "lightgbm"]:
            tasks = []
            for (operation, eid), rs in grouped.items():
                if operation != op:
                    continue
                high = [r for r in rs if r["condition"] == "high"]
                randoms = sorted((r for r in rs if r["condition"] == "random"), key=lambda r: r["repeat"])
                if len(high) != 1 or [r["repeat"] for r in randoms] != list(range(10)):
                    raise ValueError("Incomplete high/random alternatives")
                if not all(r["probe"][probe]["available"] for r in rs):
                    continue
                tasks.append(dict(patient_id=high[0]["patient_id"], high=effect(high[0], probe, "d"),
                                  controls=[effect(r, probe, "d") for r in randoms],
                                  unique=len({r["group_id"] for r in randoms})))
            key = f"local_shap|support|{op}|correct|{probe}"
            panels[key] = calculate(tasks, a.seed, a.draws)
            existing = a.source / "effects/g1_all_panels.json"
            if existing.exists():
                old = read(existing)[key]
                result = panels[key]
                if old["patients"] != result["patients"] or old["analyzed_explanations"] != result["explanations"]:
                    raise ValueError("Original analysis population changed")
                if result["gain"] is not None:
                    metric = old["metrics"]["high_minus_random_d"]
                    if not np.isclose(metric["mean"], result["gain"], atol=1e-12, rtol=0):
                        raise ValueError("Original intervention estimate does not replay")
                    result["original_patient_bootstrap_ci95"] = metric["ci95"]
            print("Completed", key, "patients", len({t["patient_id"] for t in tasks}), flush=True)
    dump(a.out / "monte_carlo_sensitivity.json", dict(
        panels=panels, seed=a.seed, new_edits=0, new_model_predictions=0,
        scope="Exploratory Monte Carlo sensitivity of original ten controls. Empirical resampling cannot reveal unobserved alternatives or replace 30 genuinely sampled alternatives.",
        source_manifest=sha(a.source / "scoring_manifest.json")))
    dump(a.out / "run_manifest.json", dict(status="complete", api_calls=0, models_trained=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("artifacts/code10k_interventions_v1"))
    p.add_argument("--out", type=Path, default=Path("artifacts/code10k_intervention_control_mc"))
    p.add_argument("--seed", type=int, default=261020)
    p.add_argument("--draws", type=int, default=5000)
    a = p.parse_args()
    if a.draws < 2:
        p.error("At least two resamples required")
    a.out.mkdir(parents=True, exist_ok=False)
    try:
        main(a)
    except Exception as exc:
        dump(a.out / "run_manifest.json", dict(status="failed", error=str(exc)))
        raise
    finally:
        archive = Path(str(a.out)+"_reports.tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            for f in a.out.glob("*.json"):
                tar.add(f, arcname=f.name)
        print("Return report:", archive, flush=True)
