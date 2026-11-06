"""TabR-S adaptation: official core, exact torch search, complete-pattern exclusion.

Official class snapshot lives alongside this file as tabr_official_core.py.
Only the candidate retrieval path is replaced; E, K, T, label encoder and P
are inherited unchanged. No query labels are accepted by adapted forward.
"""
import numpy as np
import torch
from .tabr_official_core import Model
from .tree_subset import Encoding


class DenseEncoding:
    def __init__(self, dictionary, fit):
        self.base = Encoding(dictionary)
        # Token presence plus a per-feature unknown indicator; explicit false
        # has its own coordinate, and categorical codes are never ordinal.
        self.tokens = []
        self.features = sorted(self.base.features)
        for f in self.features:
            meta = self.base.features[f]
            values = ['true', 'false'] if meta['type'] == 'B' else sorted(map(str, meta['values']))
            self.tokens.extend((f, v) for v in values)
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.fi = {f: len(self.tokens)+i for i, f in enumerate(self.features)}
        ages = np.array([float(r['context']['age']) for r in fit])
        self.mean = float(ages.mean()); self.std = max(1., float(ages.std()))
        self.width = len(self.tokens)+len(self.features)+2

    def encode(self, tokens, context):
        self.base.encode(tokens, context)  # validate values, conflicts, demographics
        x = np.zeros(self.width, dtype=np.float32)
        x[len(self.tokens):len(self.tokens)+len(self.features)] = 1
        for t in set(tokens):
            parts = t.split('_@_', 1)
            f, v = parts if len(parts)==2 else (parts[0], 'true')
            x[self.index[(f,v)]] = 1
            x[self.fi[f]] = 0
        x[-2:] = [(float(context['age'])-self.mean)/self.std, float(context['sex']=='M')]
        return x


class TabRS(Model):
    def forward(self, query, bank, bank_y, query_pattern, bank_pattern, context_size=96,
                use_retrieval=True, use_difference=True):
        x, k = self._encode({'num': query})
        if use_retrieval:
            # Search is nondifferentiable. Selected keys are then re-encoded with
            # autograd, as in the official memory-efficient path.
            with torch.no_grad():
                candidate_k = self._encode({'num': bank})[1]
                distance = k.detach().square().sum(1, keepdim=True) - 2*k.detach()@candidate_k.T + candidate_k.square().sum(1)[None]
                distance.masked_fill_(query_pattern[:,None] == bank_pattern[None,:], float('inf'))
                if (torch.isfinite(distance).sum(1) < context_size).any():
                    raise ValueError('Insufficient non-self patterns for context_size')
                # Stable bank order resolves exact distance ties without labels.
                indices = torch.argsort(distance, dim=1, stable=True)[:,:context_size]
            unique, inverse = indices.flatten().unique(sorted=True, return_inverse=True)
            selected = self._encode({'num': bank[unique]})[1]
            context_k = selected[inverse].reshape(len(query), context_size, -1)
            similarity = -k.square().sum(-1,keepdim=True) + (2*(k[...,None,:]@context_k.transpose(-1,-2))).squeeze(-2) - context_k.square().sum(-1)
            probs = self.dropout(similarity.softmax(-1))
            values = self.label_encoder(bank_y[indices][...,None])
            if use_difference:
                values = values+self.T(k[:,None]-context_k)
            x = x+(probs[:,None]@values).squeeze(1)
        for block in self.blocks1:
            x = x+block(x)
        return self.head(x)


def make_model(width, n_classes, config):
    return TabRS(n_num_features=width,n_bin_features=0,cat_cardinalities=[],
                 n_classes=n_classes,**config)
