"""Parallel RQ1 intervals with the exact original draws and estimators."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import importlib
import multiprocessing
import os
import time
import numpy as np


@lru_cache(maxsize=8)
def draw_indices(groups, seed, bootstrap):
    # Preserve the ordering and RNG calls in trace2400.resampling.cluster_indices.
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    return tuple(np.concatenate([by[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
                 for _ in range(bootstrap))


def interval_batch(job):
    groups, seed, bootstrap, metrics = job
    ix = draw_indices(groups, seed, bootstrap)
    results = []
    for index, values in metrics:
        v = np.asarray(values, dtype=float)
        # Same scalar mean per resample as the original implementation.
        bs = [float(v[i].mean()) for i in ix]
        results.append((index, dict(mean=float(v.mean()),
                       ci95=np.quantile(bs, [.025, .975]).tolist(), n=len(v))))
    return results


def summarize_pair(old, new, dataset, workers=64, bootstrap=1000):
    mod = importlib.import_module("code_eval.trace2400.rq1_summary" if dataset == "trace"
                                  else "code_eval.rq1_summary")
    default_seed = 261013 if dataset == "trace" else 261006
    pending = []
    refs = []
    def capture(values, seed=default_seed, bootstrap=bootstrap, groups=None):
        if len(values) == 0:
            return dict(mean=None, ci95=None, n=0)
        groups = tuple(groups) if groups is not None else tuple(range(len(values)))
        if len(groups) != len(values):
            raise ValueError("Cluster/value length mismatch")
        result = {}
        refs.append(result)
        pending.append((groups, seed, bootstrap, len(refs)-1, list(values)))
        return result
    original = mod.interval
    try:
        mod.interval = capture
        print("Summary: collecting original-ten comparisons", flush=True)
        before = mod.summarize(old, bootstrap=bootstrap)
        print("Summary: collecting extended comparisons", flush=True)
        after = mod.summarize(new, bootstrap=bootstrap)
    finally:
        mod.interval = original
    grouped = defaultdict(list)
    for groups, seed, draws, index, values in pending:
        grouped[(groups, seed, draws)].append((index, values))
    jobs = []
    for (groups, seed, draws), metrics in grouped.items():
        for start in range(0, len(metrics), 16):
            jobs.append((groups, seed, draws, metrics[start:start+16]))
    if not jobs:
        return before, after
    workers = max(1, min(int(workers), os.cpu_count() or 1, len(jobs)))
    print(f"Summary: {len(refs)} intervals in {len(jobs)} batches; {workers} CPU processes; {bootstrap} resamples each", flush=True)
    # Each process runs NumPy single-threaded; parallelism is across statistics.
    names = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
    saved = {k: os.environ.get(k) for k in names}
    for k in names:os.environ[k] = "1"
    start = time.monotonic()
    done_count = 0
    last_log = start
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(interval_batch, j) for j in jobs}
            while futures:
                ready, futures = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
                for future in ready:
                    for index, result in future.result():
                        refs[index].update(result)
                        done_count += 1
                now = time.monotonic()
                if not ready or not futures or now - last_log >= 5:
                    print(f"Summary progress: {done_count}/{len(refs)} intervals; {now-start:.1f}s", flush=True)
                    last_log = now
    finally:
        for k, value in saved.items():
            if value is None:os.environ.pop(k, None)
            else:os.environ[k] = value
    if any(not r for r in refs):
        raise ValueError("Missing parallel interval")
    return before, after
