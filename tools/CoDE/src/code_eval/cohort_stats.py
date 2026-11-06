"""COHD-inspired concept counts with an explicitly NEW diagnostic aggregator.

Only observed tokens contribute. Unlisted features are not negatives.
The output is a normalized support distribution, not a calibrated diagnosis probability.
"""
import hashlib
import json
import math
from collections import Counter,defaultdict

VERSION='cohd_mean_log_lift_v0.1'

def identity(age,sex,tokens):
    if isinstance(age,bool) or not isinstance(age,(int,float)) or not math.isfinite(age) or age<0:
        raise ValueError('Invalid age')
    if sex not in ('M','F'):raise ValueError('Invalid sex')
    if not isinstance(tokens,list) or not all(isinstance(t,str) and t for t in tokens):raise ValueError('Invalid tokens')
    raw=json.dumps([float(age),sex,sorted(set(tokens))],ensure_ascii=False,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()

def fold_for(group,folds,seed):
    if folds<2:raise ValueError('At least two folds required')
    return int(hashlib.sha256(f'{seed}|{group}'.encode()).hexdigest(),16)%folds

def query_patient(row):
    # Match existing TBBOS patient_inputs.jsonl, deny target/extra fields.
    if set(row)!={'case_id','patient'} or not isinstance(row['case_id'],str) or not row['case_id']:
        raise ValueError('Query must contain only case_id and patient; gold labels are forbidden')
    p=row['patient']
    required={'age','sex','observed_evidence_tokens'}
    if not isinstance(p,dict) or not required<=set(p) or set(p)-required-{'observed_evidences'}:
        raise ValueError('Unexpected patient fields; labels and reference differentials forbidden')
    group=identity(p['age'],p['sex'],p['observed_evidence_tokens'])
    return group,sorted(set(p['observed_evidence_tokens']))

class Counts:
    def __init__(self,labels):
        if len(labels)<2 or len(set(labels))!=len(labels) or not all(isinstance(x,str) and x for x in labels):
            raise ValueError('Unique nonempty disease labels required')
        self.labels=list(labels);self.n=0;self.disease=Counter();self.feature=Counter();self.pair=defaultdict(Counter)

    def add(self,tokens,label):
        if label not in self.labels:raise ValueError('Unknown disease')
        self.n+=1;self.disease[label]+=1
        for f in set(tokens):self.feature[f]+=1;self.pair[f][label]+=1

    def subtract(self,held):
        result=Counts(self.labels);result.n=self.n-held.n
        result.disease=self.disease-held.disease;result.feature=self.feature-held.feature
        for f in result.feature:result.pair[f]=self.pair[f]-held.pair[f]
        return result

    def validate(self):
        if self.n<=0 or any(self.disease[d]<=0 for d in self.labels):
            raise ValueError('Every disease must have training support in every fitting fold')

    def score(self,tokens,alpha=5.0):
        self.validate()
        if not math.isfinite(alpha) or alpha<=0:raise ValueError('alpha must be positive and finite')
        tokens=sorted(set(tokens));known=[f for f in tokens if self.feature[f]>0]
        prior=[self.disease[d]/self.n for d in self.labels]
        logits=[];details=[]
        for d,p in zip(self.labels,prior):
            terms=[]
            for f in known:
                conditional=(self.pair[f][d]+alpha*p)/(self.feature[f]+alpha)
                terms.append(math.log(conditional/p))
            loglift=sum(terms)/len(terms) if terms else 0.0
            logits.append(math.log(p)+loglift)
            details.append({'diagnosis':d,'prior':p,'mean_log_lift':loglift})
        peak=max(logits);exp=[math.exp(v-peak) for v in logits];total=sum(exp)
        for detail,value in zip(details,exp):detail['support']=value/total
        return dict(method_id=VERSION,score_semantics='normalized_diagnostic_support_not_calibrated_probability',scores=details,observed_token_count=len(tokens),matched_token_count=len(known),unseen_tokens=sorted(set(tokens)-set(known)),supervision_mask=bool(known),fallback='none' if known else 'prior_only',training_unique_groups=self.n)

    def export(self):
        return dict(labels=self.labels,n=self.n,disease=dict(self.disease),feature=dict(self.feature),pair={k:dict(v) for k,v in self.pair.items()})
