"""Frozen-recipe association source with explicit demographic discretization."""
import math
from collections import Counter, defaultdict
from .cohort_stats import Counts


def context_tokens(context):
    age=float(context['age']);sex=context['sex']
    if not math.isfinite(age) or age<0 or sex not in ('M','F'):raise ValueError('Invalid context')
    return [f'__CTX_AGE10_{int(age//10)}',f'__CTX_SEX_{sex}']


class AssociationSource:
    def __init__(self,labels,alpha=5.):
        self.counts=Counts(labels);self.alpha=alpha

    def fit(self,records):
        for r in records:self.counts.add(list(r['tokens'])+context_tokens(r['context']),r['label'])
        self.counts.validate()

    def distribution(self,tokens,context):
        if any(self.counts.feature[t]==0 for t in tokens):return None
        augmented=list(tokens)+context_tokens(context)
        # An unseen demographic bin has zero association contribution; record it.
        result=self.counts.score(augmented,alpha=self.alpha)
        return [r['support'] for r in result['scores']]

    def export(self):
        return dict(counts=self.counts.export(),alpha=self.alpha,
            context_rule='floor(age/10), sex; fixed observed tokens in mean-log-lift aggregation',
            unseen_context='zero association term; remaining known terms retained',
            semantics='COHD-inspired counts and mean-log-lift diagnostic adapter, not original COHD classifier')

    @classmethod
    def restore(cls,data):
        raw=data['counts'];obj=cls(raw['labels'],data['alpha']);c=obj.counts
        c.n=raw['n'];c.disease=Counter(raw['disease']);c.feature=Counter(raw['feature'])
        c.pair=defaultdict(Counter,{k:Counter(v) for k,v in raw['pair'].items()});c.validate()
        return obj
