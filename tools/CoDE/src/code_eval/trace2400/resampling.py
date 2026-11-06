"""Subject-cluster bootstrap; point estimates retain admission weighting."""
from collections import defaultdict
import numpy as np
def cluster_indices(groups,rng):
    by=defaultdict(list)
    for i,g in enumerate(groups):by[g].append(i)
    keys=sorted(by)
    if not keys:return np.array([],dtype=int)
    return np.concatenate([by[keys[k]] for k in rng.integers(0,len(keys),len(keys))])
