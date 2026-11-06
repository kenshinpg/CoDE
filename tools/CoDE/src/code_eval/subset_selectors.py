"""Source-independent evidence subset selectors; no outcome labels are accepted.

The predictor is bound to one patient's fixed context. It returns the entire
class distribution for the exact supplied fact set. Sparse validity and fitting
isolation must be established by the caller, not inferred from successful calls.
"""
import math
import random


class EvidenceSupport:
    def __init__(self, facts, groups, labels, predict, *, allow_empty=False):
        self.facts = frozenset(facts)
        self.groups = {k: frozenset(v) for k, v in sorted(groups.items())}
        if not self.facts or len(self.facts) != len(facts):
            raise ValueError('Nonempty unique fact IDs required')
        flat = [f for g in self.groups.values() for f in g]
        if not self.groups or any(not g for g in self.groups.values()) or len(flat) != len(set(flat)) or set(flat) != self.facts:
            raise ValueError('Groups must partition all facts')
        if any(not isinstance(f, str) or not f for f in self.facts):
            raise ValueError('Nonempty string fact IDs required')
        if any(not isinstance(g, str) or not g for g in self.groups):
            raise ValueError('Nonempty string group IDs required')
        self.labels = tuple(labels)
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise ValueError('Unique class labels required')
        self.predict = predict
        self.allow_empty = allow_empty
        self.cache = {}

    def distribution(self, facts):
        key = frozenset(facts)
        if not key <= self.facts:
            raise ValueError('Unknown patient facts')
        if not key and not self.allow_empty:
            raise ValueError('Empty subset not validated for this source')
        if key not in self.cache:
            raw = self.predict(key)
            if set(raw) != set(self.labels):
                raise ValueError('Prediction class space mismatch')
            if any(isinstance(v, bool) for v in raw.values()):
                raise ValueError('Boolean is not a probability')
            values = {d: float(raw[d]) for d in self.labels}
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
                raise ValueError('Nonfinite or out-of-range probability')
            if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
                raise ValueError('Expected normalized class distribution')
            self.cache[key] = values
        return dict(self.cache[key])

    def source_target(self):
        full = self.distribution(self.facts)
        # Frozen label order supplies a deterministic tie break.
        return max(self.labels, key=lambda d: full[d])

    def facts_for_groups(self, group_ids):
        if not set(group_ids) <= set(self.groups):
            raise ValueError('Unknown group IDs')
        return frozenset(f for g in group_ids for f in self.groups[g])


def leave_group_out(source):
    """RQ1 target is the full-input prediction, never a supplied gold diagnosis."""
    target = source.source_target()
    full = source.distribution(source.facts)[target]
    effects = {g: full - source.distribution(source.facts - fs)[target]
               for g, fs in source.groups.items()}
    order = sorted(effects, key=lambda g: (-abs(effects[g]), g))
    return dict(target=target, signed_effects=effects, order=order,
                ranking_semantics='absolute_prediction_change',
                clinical_necessity_validated=False)


def ranked_subset(source, order, k):
    if set(order) != set(source.groups) or len(order) != len(source.groups):
        raise ValueError('Ordering must include every group once')
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= len(order):
        raise ValueError('Invalid group budget')
    chosen = list(order[:k])
    return dict(group_ids=chosen, fact_ids=sorted(source.facts_for_groups(chosen)),
                group_count=k, fact_count=len(source.facts_for_groups(chosen)))


def random_order(source, seed):
    order = sorted(source.groups)
    random.Random(seed).shuffle(order)
    return order


def sis_backward_path(source, threshold):
    """Single SIS-style path, not a complete collection or exhaustive minimum.

    Continue deletion through threshold crossings. The restored suffix searches
    the complete path; the empty/fixed-context baseline must be validated.
    """
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('Threshold must be in (0,1]')
    if not source.allow_empty:
        raise ValueError('SIS requires a validated fixed-context baseline')
    target = source.source_target()
    baseline = source.distribution(frozenset())[target]
    full = source.distribution(source.facts)[target]
    result = dict(target=target, threshold=threshold, baseline_support=baseline,
                  full_support=full, clinical_necessity_validated=False)
    if full < threshold:
        return dict(result, status='FULL_INPUT_BELOW_THRESHOLD', selected=None)
    if baseline >= threshold:
        return dict(result, status='BASELINE_ALREADY_SUFFICIENT', selected=None)
    remaining = set(source.groups)
    removed, path = [], []
    while remaining:
        candidates = {g: source.distribution(source.facts_for_groups(remaining-{g}))[target]
                      for g in sorted(remaining)}
        chosen = min(candidates, key=lambda g: (-candidates[g], g))
        remaining.remove(chosen)
        removed.append(chosen)
        path.append(dict(removed_group=chosen, remaining_support=candidates[chosen],
                         remaining_groups=len(remaining)))
    order = list(reversed(removed))
    selected = None
    for k in range(1, len(order)+1):
        candidate = ranked_subset(source, order, k)
        if source.distribution(candidate['fact_ids'])[target] >= threshold:
            selected = candidate
            break
    return dict(result, status='PATH_SUBSET_FOUND', selected=selected,
                order=order, deletion_path=path,
                minimality='minimum_size_on_this_path_not_global',
                unique_prediction_queries=len(source.cache))


def evidence_support_record(source, cited, diagnosis, *, eligibility_threshold=None):
    """Export separate support components without inventing a clinical reward."""
    if diagnosis not in source.labels:
        raise ValueError('Unknown proposed diagnosis')
    if eligibility_threshold is not None and (isinstance(eligibility_threshold, bool)
            or not math.isfinite(eligibility_threshold) or not 0 <= eligibility_threshold <= 1):
        raise ValueError('Invalid development-frozen eligibility threshold')
    full = source.distribution(source.facts)[diagnosis]
    retained = source.distribution(cited)
    value = retained[diagnosis]
    status = ('THRESHOLD_NOT_FROZEN' if eligibility_threshold is None else
              'LOW_FULL_SUPPORT' if full < eligibility_threshold else 'ELIGIBLE_BY_SOURCE_THRESHOLD')
    return dict(proposed_diagnosis=diagnosis, full_support=full, cited_support=value,
                support_loss=full-value, retained_support=value-full,
                cited_margin=value-max(v for d, v in retained.items() if d != diagnosis),
                eligibility_status=status, clinical_omission_label=None)
