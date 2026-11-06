"""Training-frequency rules and set selectors on intact observed evidence groups.

SIS and AnchorBaseBeam are vendored upstream implementations. Their input space
is explicitly adapted to group-retention masks; no labels enter the selectors.
"""
from collections import Counter
from pathlib import Path
import importlib.util
import math
import numpy as np


def load_vendor(name, filename):
    p = Path(__file__).resolve().parents[2] / 'vendor' / name / filename
    spec = importlib.util.spec_from_file_location('code_vendor_' + name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frequency_weights(train, groups):
    """Each real training patient counts once per exact feature/value token."""
    count = Counter(t for r in train for t in set(r['tokens']))
    n = len(train)
    if not n:
        raise ValueError('Empty training population')
    return weights_from_counts(count, n, groups)


def weights_from_counts(count, n, groups):
    return {
        'frequency': {g: float(np.mean([count[t] / n for t in ts])) for g, ts in groups.items()},
        'idf': {g: float(np.mean([math.log((n + 1) / (count[t] + 1)) + 1 for t in ts])) for g, ts in groups.items()},
    }


def order_weights(w):
    return sorted(w, key=lambda g: (-w[g], g))


def greedy_order(score, n, max_size):
    """Forward maximize q(d | retained groups); score returns full distributions."""
    selected = []
    for _ in range(max_size):
        rest = [j for j in range(n) if j not in selected]
        masks = np.zeros((len(rest), n), dtype=np.int8)
        masks[:, selected] = 1
        masks[np.arange(len(rest)), rest] = 1
        values = score(masks)
        selected.append(rest[int(np.argmax(values))])
    return selected


def first_sis(score, n, threshold=.9):
    module = load_vendor('sis', 'sis.py')
    full = float(score(np.ones((1, n), dtype=np.int8))[0])
    empty = float(score(np.zeros((1, n), dtype=np.int8))[0])
    if full < threshold:
        return dict(status='full_below_threshold', indices=None, full=full, empty=empty)
    if empty >= threshold:
        return dict(status='empty_sufficient', indices=[], full=full, empty=empty)
    result = module.find_sis(score, threshold, np.ones(n), np.ones(n, dtype=bool), np.zeros(n))
    if result is None:
        raise ValueError('Upstream SIS failed despite eligible full input')
    indices = sorted(int(x[0]) for x in result.sis)
    mask = np.zeros((1, n), dtype=np.int8); mask[0, indices] = 1
    if float(score(mask)[0]) < threshold - 1e-10:
        raise ValueError('SIS threshold not preserved')
    return dict(status='complete', indices=indices, full=full, empty=empty)


class SearchLimit(RuntimeError):
    pass


def anchor_mask(classify, n, target, seed, limit=200000):
    """Original Anchors search under independent Bernoulli(.5) group removal.

    Precision means retaining the *source prediction under this distribution*.
    It is neither clinical precision nor population-conditional tabular coverage.
    """
    np.random.seed(seed)
    module = load_vendor('anchor', 'anchor_base.py')
    used = 0
    def sample(present, size, compute_labels=True):
        nonlocal used
        masks = np.random.binomial(1, .5, (size, n)).astype(np.int8)
        masks[:, present] = 1
        if not compute_labels:
            return masks.copy(), masks, np.zeros(0, dtype=int)
        if used + size > limit:
            raise SearchLimit('Anchor prediction-query limit reached')
        used += size
        labels = (classify(masks) == target).astype(int)
        return masks.copy(), masks, labels
    try:
        r = module.AnchorBaseBeam.anchor_beam(sample, desired_confidence=.95,
            delta=.05, epsilon=.1, epsilon_stop=.05, batch_size=16,
            min_samples_start=100, beam_size=1, coverage_samples=10000)
    except SearchLimit:
        return dict(status='query_limit', indices=None, search_queries=used)
    indices = sorted(int(x) for x in r['feature'])
    precision = float(r['precision'][-1]) if indices else float(r['all_precision'])
    # Independent fixed verification sample; reported, not used to choose another anchor.
    rng = np.random.default_rng(seed + 104729)
    masks = rng.binomial(1, .5, (2000, n)).astype(np.int8); masks[:, indices] = 1
    success = int((classify(masks) == target).sum())
    return dict(status='complete' if precision >= .95 else 'precision_not_met',
                indices=indices if precision >= .95 else None,
                candidate_indices=indices, search_precision=precision,
                verification_success=success, verification_n=2000,
                perturbation_coverage=2. ** (-len(indices)), search_queries=used)
