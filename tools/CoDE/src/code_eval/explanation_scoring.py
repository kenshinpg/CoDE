"""CoDE scoring primitives. No text extraction or clinical reference labels.

Inputs are fact IDs already aligned by the shared data producer. A support
callback accepts (frozenset[fact_id], proposed_diagnosis) and returns [0, 1].
Unknown/sparse input handling belongs to the source adapter, not this layer.
"""
import hashlib
import math
import random


def validate_mapping(facts, cited, groups):
    facts, cited = list(facts), list(cited)
    if not facts or any(not isinstance(x, str) or not x for x in facts):
        raise ValueError('Nonempty patient fact IDs required')
    if len(set(facts)) != len(facts) or len(set(cited)) != len(cited):
        raise ValueError('Duplicate IDs; normalize upstream without counting repetition')
    if not set(cited) <= set(facts):
        raise ValueError('Unsupported citations cannot count as patient coverage')
    if not groups or any(not g for g in groups):
        raise ValueError('Nonempty evidence groups required')
    flat = [x for g in groups for x in g]
    if len(flat) != len(set(flat)) or set(flat) != set(facts):
        raise ValueError('Groups must be a disjoint complete partition')
    return frozenset(facts), frozenset(cited), tuple(frozenset(g) for g in groups)


def checked_support(support, facts, diagnosis):
    value = support(facts, diagnosis)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Adapter must return a numeric support')
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Expected finite normalized support in [0, 1]')
    return float(value)


def score_coverage(facts, cited, groups, diagnosis, support, *, empty_supported=False):
    """Compute manuscript C and R. NA is explicit; negative effects are retained.

    Empty callback inputs are forbidden unless adapter-specific validation has
    authorized them. Missing any removal weight makes weighted coverage NA.
    This function does not certify nonempty sparse inputs as in-distribution.
    """
    if not isinstance(diagnosis, str) or not diagnosis:
        raise ValueError('Proposed diagnosis required, never substitute gold')
    facts, cited, groups = validate_mapping(facts, cited, groups)
    cache = {}
    def q(subset):
        if not subset and not empty_supported:
            return None
        if subset not in cache:
            cache[subset] = checked_support(support, subset, diagnosis)
        return cache[subset]
    full, retained = q(facts), q(cited)
    effects, weights, fractions = [], [], []
    for group in groups:
        removed = q(facts - group)
        effect = None if removed is None else full - removed
        effects.append(effect)
        weights.append(None if effect is None else max(effect, 0.0))
        fractions.append(len(cited & group) / len(group))
    reason = None
    weighted = None
    if any(w is None for w in weights):
        reason = 'unvalidated_empty_removal'
    elif sum(weights) == 0:
        reason = 'no_positive_support_weight'
    else:
        weighted = sum(w*c for w, c in zip(weights, fractions)) / sum(weights)
    return dict(uniform_coverage=len(cited)/len(facts), weighted_coverage=weighted,
                weighted_na_reason=reason, support_full=full, support_cited=retained,
                retained_support=None if retained is None else retained-full,
                retention_na_reason='unvalidated_empty_citations' if retained is None else None,
                signed_contributions=effects, positive_weights=weights,
                group_coverage=fractions, unique_support_calls=len(cache))


def permuted_coverage(weights, fractions, *, patient_id, diagnosis, donor_id, seed):
    """Same source weights and permutation for all variants of a patient/diagnosis.

    Control definition is explicit but the primary donor is not yet frozen.
    """
    if len(weights) != len(fractions) or not weights:
        raise ValueError('Aligned nonempty weights and coverage required')
    if any(w is None for w in weights):
        return None
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('Nonnegative finite donor weights required')
    if any(not math.isfinite(c) or not 0 <= c <= 1 for c in fractions):
        raise ValueError('Coverage must be in [0, 1]')
    shuffled = list(weights)
    import json
    key = json.dumps([patient_id, diagnosis, donor_id, seed], ensure_ascii=False)
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))
    rng.shuffle(shuffled)
    return None if sum(shuffled) == 0 else sum(w*c for w,c in zip(shuffled, fractions))/sum(shuffled)


def plan_single_group_deletions(facts, cited, groups):
    """Private edit plans only: no generated text and no clinical importance label.

    Caller supplies dependency-safe groups. Every cited group is eligible;
    no teacher weight or correctness label is consulted.
    """
    facts, cited, groups = validate_mapping(facts, cited, groups)
    return [dict(group_index=i, remove_fact_ids=sorted(cited & group),
                 expected_cited_fact_ids=sorted(cited - group),
                 status='planned_requires_text_edit_and_independent_remapping')
            for i, group in enumerate(groups) if cited & group]


class AssociationSupport:
    """Wrap existing COHD-inspired Counts; fact/token mapping must be lossless.

    Actual cohort provenance, split isolation and sparse behavior must be audited
    separately. This adapter alone does not approve a fitted artifact for use.
    """
    def __init__(self, counts, fact_tokens):
        self.counts = counts
        self.fact_tokens = dict(fact_tokens)
        if any(not isinstance(t, str) or not t for t in self.fact_tokens.values()):
            raise ValueError('Each fact must map to an existing nonempty input token')
        if len(set(self.fact_tokens.values())) != len(self.fact_tokens):
            raise ValueError('Distinct scoring facts must not alias a single token')

    def __call__(self, facts, diagnosis):
        if diagnosis not in self.counts.labels:
            raise ValueError('Diagnosis outside source label space')
        tokens = [self.fact_tokens[f] for f in facts]
        if any(self.counts.feature[t] == 0 for t in tokens):
            raise ValueError('Unseen input token: audit rather than silently use prior')
        result = self.counts.score(tokens)
        return next(row['support'] for row in result['scores'] if row['diagnosis'] == diagnosis)
