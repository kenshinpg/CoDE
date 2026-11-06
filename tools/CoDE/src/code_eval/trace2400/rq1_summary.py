"""Patient-paired RQ1 summaries, with explicit common-population families."""
import itertools
import numpy as np
from .resampling import cluster_indices

FAMILIES={'sources':['cohd','encoder','grasp','lightgbm','tabr'],
          'tree_rules':['lightgbm','tree_gain','tree_local_abs_shap','tree_class_abs_shap'],
          'retrieval':['tabr','no_retrieval']}
BUDGETS=[.25,.5,.75]

def interval(values,seed=261013,bootstrap=1000,groups=None):
    v=np.asarray(values,dtype=float)
    if len(v)==0:return dict(mean=None,ci95=None,n=0)
    rng=np.random.default_rng(seed)
    bs=[float(v[cluster_indices(groups if groups is not None else list(range(len(v))),rng)].mean()) for _ in range(bootstrap)]
    return dict(mean=float(v.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),n=len(v))

def self_source(method,probe):return (method==probe) or (probe=='lightgbm' and method.startswith('tree_'))

def summarize(records,bootstrap=1000):
    subjects={r["patient_id"]:r["subject_group"] for r in records}
    panels=[];differences=[];overlap=[]
    for probe,budget,population in itertools.product(['encoder','lightgbm'],BUDGETS,['both_full_correct','all_test']):
        rs=[r for r in records if r['probe']==probe and r['fraction']==budget]
        index={(r['patient_id'],r['selector']):r for r in rs}
        if len(index)!=len(rs):raise ValueError('Duplicate patient-method result')
        for family,methods in FAMILIES.items():
            sets=[{r['patient_id'] for r in rs if r['selector']==m and r['available'] and (population=='all_test' or r['both_full_correct'])} for m in methods]
            ids=sorted(set.intersection(*sets))
            common=dict(probe=probe,fraction=budget,population=population,family=family,patients=len(ids))
            for m in methods:
                chosen=[index[pid,m] for pid in ids]
                panels.append(dict(common,selector=m,self_source=self_source(m,probe),
                    accuracy=interval([r['correct'] for r in chosen],bootstrap=bootstrap,groups=[subjects[pid] for pid in ids]),
                    matched_random_accuracy=interval([r['random_accuracy'] for r in chosen],bootstrap=bootstrap,groups=[subjects[pid] for pid in ids]),
                    gain=interval([r['correct']-r['random_accuracy'] for r in chosen],bootstrap=bootstrap,groups=[subjects[pid] for pid in ids]),
                    mean_facts=float(np.mean([r['fact_count'] for r in chosen])) if chosen else None,
                    individually_available=sum(r['selector']==m and r['available'] for r in rs)))
            for m,n in itertools.combinations(methods,2):
                differences.append(dict(common,left=m,right=n,
                    selected_accuracy_difference=interval([index[pid,m]['correct']-index[pid,n]['correct'] for pid in ids],bootstrap=bootstrap,groups=[subjects[pid] for pid in ids]),
                    matched_gain_difference=interval([(index[pid,m]['correct']-index[pid,m]['random_accuracy'])-(index[pid,n]['correct']-index[pid,n]['random_accuracy']) for pid in ids],bootstrap=bootstrap,groups=[subjects[pid] for pid in ids])))
                j=[];same=[]
                for pid in ids:
                    a=set(index[pid,m]['selected_tokens']);b=set(index[pid,n]['selected_tokens'])
                    j.append(len(a&b)/len(a|b) if a|b else 1.);same.append(a==b)
                overlap.append(dict(common,left=m,right=n,mean_patient_jaccard=float(np.mean(j)) if j else None,identical_set_fraction=float(np.mean(same)) if same else None))
    return dict(panels=panels,paired_method_differences=differences,selection_overlap=overlap,bootstrap=bootstrap,
        interval_unit='inferred subject cluster; admission-weighted estimates; paired selection minus mean of ten matched draws; percentile 95% intervals',
        inference_scope='Retrospective test-role evaluation; exploratory secondary contrasts without multiplicity adjustment')
