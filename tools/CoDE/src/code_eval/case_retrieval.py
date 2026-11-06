"""Transparent query-conditioned case retrieval; no learned metric or query label."""
from collections import defaultdict, Counter
import hashlib, json, math
import numpy as np

def pattern(tokens):
    return hashlib.sha256(json.dumps(sorted(set(tokens)),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def by_feature(tokens):
    out=defaultdict(set)
    for t in set(tokens):
        parts=t.split('_@_',1);out[parts[0]].add(parts[1] if len(parts)==2 else 'true')
    return dict(out)

def context(h):
    age=float(h['age'])
    if not math.isfinite(age) or age<0 or h['sex'] not in ('M','F'):raise ValueError('Invalid context')
    return age,h['sex']

class CaseRetrieval:
    def __init__(self, records, labels, k=5):
        if k<1 or len(set(labels))!=len(labels):raise ValueError('Invalid label/k configuration')
        self.labels=list(labels);self.k=k;groups=defaultdict(list)
        for r in records:
            if r['pattern_id']!=pattern(r['tokens']) or r['label'] not in labels:raise ValueError('Invalid bank row')
            context(r['context']);groups[r['pattern_id']].append(r)
        self.conflicting_patterns={p for p,rs in groups.items() if len({r['label'] for r in rs})>1}
        # One seat per complete evidence pattern, chosen without outcome ranking.
        self.rows=sorted([min(rs,key=lambda r:r['input_group_id']) for p,rs in groups.items() if p not in self.conflicting_patterns],key=lambda r:r['input_group_id'])
        self.n=len(self.rows)
        if self.n<k:raise ValueError('Insufficient unique bank patterns')
        self.counts={};self.postings=defaultdict(list);self.vocabulary=set()
        for i,r in enumerate(self.rows):
            for f,vs in by_feature(r['tokens']).items():
                self.counts.setdefault(f,np.zeros(self.n))[i]=len(vs)
                for v in vs:self.postings[(f,v)].append(i)
            self.vocabulary.update(r['tokens'])
        self.ages=np.array([context(r['context'])[0] for r in self.rows]);self.sexes=np.array([r['context']['sex'] for r in self.rows])
        self.audit={'input_rows':len(records),'bank_rows':self.n,'conflicting_patterns_excluded':len(self.conflicting_patterns),'same_label_duplicate_rows_removed':sum(len(rs)-1 for p,rs in groups.items() if p not in self.conflicting_patterns)}

    def query(self,tokens,h,*,excluded_pattern,excluded_identity):
        age,sex=context(h);ts=set(tokens)
        if not ts:return {'available':False,'reason':'empty_evidence'}
        if not ts<=self.vocabulary:return {'available':False,'reason':'unseen_token'}
        features=by_feature(ts);s=np.zeros(self.n)
        for f,vs in features.items():
            intersection=np.zeros(self.n)
            for v in vs:intersection[self.postings.get((f,v),[])]+=1
            s+=intersection/(len(vs)+self.counts[f]-intersection)
        evidence=s/len(features)
        # Unqueried features do not enter numerator or denominator.
        similarity=(s+(self.sexes==sex)+np.maximum(0,1-np.abs(self.ages-age)/100))/(len(features)+2)
        allowed=np.array([r['pattern_id']!=excluded_pattern and r['input_group_id']!=excluded_identity for r in self.rows])&(evidence>0)
        idx=np.flatnonzero(allowed)
        if len(idx)<self.k:return {'available':False,'reason':'fewer_than_k_overlapping_cases','eligible_neighbors':len(idx)}
        # Stable bank-ID secondary order, no disease or query target tie-breaking.
        ranked=idx[np.argsort(-similarity[idx],kind='stable')];chosen=ranked[:self.k];cut=similarity[chosen[-1]]
        probs=np.zeros(len(self.labels))
        for i in chosen:probs[self.labels.index(self.rows[i]['label'])]+=1/self.k
        return {'available':True,'probabilities':probs.tolist(),'prediction':self.labels[int(probs.argmax())],'eligible_neighbors':len(idx),'cutoff_similarity':float(cut),'cutoff_tied_total':int(np.sum(np.abs(similarity[idx]-cut)<=1e-12)),'boundary_tie':bool(len(ranked)>self.k and abs(similarity[ranked[self.k]]-cut)<=1e-12),'mean_evidence_similarity':float(evidence[chosen].mean()),'neighbors':[{'input_group_id':self.rows[i]['input_group_id'],'pattern_id':self.rows[i]['pattern_id'],'label':self.rows[i]['label'],'similarity':float(similarity[i]),'evidence_similarity':float(evidence[i])} for i in chosen]}
