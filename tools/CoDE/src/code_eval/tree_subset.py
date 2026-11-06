"""Explicit observed-value encoding and native multiclass TreeSHAP checks."""
import hashlib
import json
import math
import random

import numpy as np


def pattern(tokens):
    return hashlib.sha256(json.dumps(sorted(set(tokens)), ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()


class Encoding:
    def __init__(self, dictionary):
        # Shared LANTERN dictionaries also contain demographic pseudo-features;
        # those already occupy the two fixed-context columns.
        self.features = {k: v for k, v in dictionary['features'].items()
                         if k not in ('DEM_AGE', 'DEM_SEX')}
        self.columns = ['age', 'sex']
        self.categorical = [1]
        self.locations = {}
        self.value_maps = {}
        for code, meta in sorted(self.features.items()):
            kind = meta['type']
            if kind not in ('B', 'C', 'M'):
                raise ValueError('Unsupported feature type: ' + code)
            values = sorted(str(v) for v in meta.get('values', {}))
            if kind == 'M':
                for value in values:
                    self.locations[(code, value)] = len(self.columns)
                    self.columns.append(code + ':' + value)
            else:
                self.locations[(code, None)] = len(self.columns)
                if kind == 'C':
                    self.categorical.append(len(self.columns))
                    self.value_maps[code] = {v: i for i, v in enumerate(values)}
                self.columns.append(code)

    def token(self, token):
        parts = token.split('_@_', 1)
        code = parts[0]
        meta = self.features[code]  # Unknown features must fail, never disappear.
        kind = meta['type']
        value = parts[1] if len(parts) == 2 else 'true'
        if kind == 'B':
            if value not in ('true', 'false'):
                raise ValueError('Invalid binary value: ' + token)
            return self.locations[(code, None)], float(value == 'true')
        if kind == 'C':
            return self.locations[(code, None)], float(self.value_maps[code][value])
        return self.locations[(code, value)], 1.0

    def encode(self, tokens, context):
        # NaN means unobserved. It never encodes an explicit negative.
        x = np.full(len(self.columns), np.nan, dtype=np.float32)
        age = float(context['age'])
        if not math.isfinite(age) or age < 0 or context['sex'] not in ('M', 'F'):
            raise ValueError('Invalid fixed context')
        x[:2] = [age, float(context['sex'] == 'M')]
        used = set()
        for token in sorted(set(tokens)):
            column, value = self.token(token)
            if column in used and x[column] != value:
                raise ValueError('Conflicting values in one feature')
            used.add(column)
            x[column] = value
        return x

    def groups(self, tokens):
        groups = {}
        for token in sorted(set(tokens)):
            code = token.split('_@_', 1)[0]
            self.token(token)
            group = self.features[code]['group_id']
            groups.setdefault(group, []).append(token)
        return groups

    def masked(self, tokens, fraction, key):
        groups = self.groups(tokens)
        keys = sorted(groups)
        random.Random(str(key)).shuffle(keys)
        retained = keys[:math.ceil(len(keys) * fraction)]
        return sorted(t for g in retained for t in groups[g])

    def export(self):
        return dict(columns=self.columns, categorical_columns=self.categorical,
                    value_maps=self.value_maps, missing='NaN_unknown_not_negative',
                    fixed_context=['age', 'sex'])


def checked_contributions(model, x, class_count):
    # Native LightGBM output is class-major: features + bias for each class.
    contrib = np.asarray(model.predict(x, pred_contrib=True, num_threads=1))
    expected = (len(x), class_count * (x.shape[1] + 1))
    if contrib.shape != expected:
        raise ValueError('Unexpected multiclass contribution layout: ' + str(contrib.shape))
    contrib = contrib.reshape(len(x), class_count, x.shape[1] + 1)
    raw = np.asarray(model.predict(x, raw_score=True, num_threads=1))
    if not np.allclose(contrib.sum(axis=2), raw, atol=1e-5, rtol=1e-5):
        raise ValueError('TreeSHAP contributions do not reconstruct raw scores')
    shifted = raw - raw.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(probabilities, model.predict(x, num_threads=1), atol=1e-6):
        raise ValueError('Multiclass raw-score/probability mismatch')
    return contrib


def metrics(probabilities, targets):
    p = np.asarray(probabilities)
    y = np.asarray(targets, dtype=int)
    if p.ndim != 2 or len(p) != len(y) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid probability matrix')
    acc = p.argmax(1) == y
    return dict(n=len(y), accuracy=float(acc.mean()),
        macro_accuracy=float(np.mean([acc[y == d].mean() for d in sorted(set(y))])),
        nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        mean_confidence=float(p.max(1).mean()))
