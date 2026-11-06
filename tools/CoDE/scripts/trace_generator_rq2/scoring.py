"""CPU adapter for the unchanged TRACE Association/Tree/Representation equations."""
import concurrent.futures as cf
import multiprocessing
import time
from collections import Counter, defaultdict
import numpy as np
from .common import read, dump, locked, sha, index, LABELS

_STATE = None

def initialize(spec):
    global _STATE
    import torch
    import lightgbm as lgb
    from code_eval.trace2400.bank import TraceBank
    from code_eval.trace2400.encoding import TraceEncoding
    from code_eval.cohort_stats import Counts
    from code_eval.context_encoder import ContextEncoder
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Reuse TraceBank.predict verbatim, loading only the three required sources.
    bank = TraceBank.__new__(TraceBank)
    bank.torch = torch; bank.device = 'cpu'; bank.labels = LABELS
    bank.encoding = TraceEncoding(read(spec['dictionary']))
    bank.tree = lgb.Booster(model_file=spec['models']['lightgbm'])
    raw = read(spec['models']['cohd']); v = raw['counts']; bank.alpha = raw['alpha']
    bank.co = Counts(v['labels']); bank.co.n = v['n']
    bank.co.disease = Counter(v['disease']); bank.co.feature = Counter(v['feature'])
    bank.co.pair = defaultdict(Counter, {k: Counter(x) for k, x in v['pair'].items()})
    ck = torch.load(spec['models']['encoder'], map_location='cpu', weights_only=True)
    if ck['labels'] != LABELS or bank.co.labels != LABELS:
        raise ValueError('Checkpoint label order changed')
    model = ContextEncoder(len(ck['vocabulary']), len(LABELS), ck['hidden'])
    model.load_state_dict(ck['state']); model.eval()
    bank.neural = {'encoder': (model, ck['vocabulary'])}; bank.retrieval = {}
    _STATE = bank

def score_patient(task):
    from code_eval.tree_subset import checked_contributions
    from run_coverage_controls import coverage
    patient, record, ledger = task
    bank = _STATE; enc = bank.encoding; pid = patient['patient_id']
    tokens = {f['fact_id']: f['feature_id'] + '_@_' + f['state'] for f in patient['facts']}
    if set(tokens.values()) != set(record['tokens']):
        raise ValueError('Frozen F/token mismatch')
    full = set(record['tokens']); groups = enc.groups(record['tokens'])
    minus = {g: tuple(sorted(full - set(ts))) for g, ts in groups.items()}
    fullkey = tuple(sorted(full)); keys = {fullkey, *minus.values()}; cited = {}
    for e in ledger:
        if e['reviewed_available']:
            if not set(e['cited_fact_ids']) <= tokens.keys():
                raise ValueError('Nonpatient citation')
            cited[e['explanation_id']] = tuple(sorted({tokens[f] for f in e['cited_fact_ids']}))
            keys.add(cited[e['explanation_id']])
    keys = sorted(keys)
    probs = bank.predict([list(k) for k in keys], {}, sources=['cohd', 'encoder', 'lightgbm'])
    cache = {n: dict(zip(keys, vs)) for n, vs in probs.items()}
    fp = {n: cache[n][fullkey] for n in cache}
    both = all(fp[n] is not None and LABELS[int(fp[n].argmax())] == record['label']
               for n in ['encoder', 'lightgbm'])
    shap = checked_contributions(bank.tree, enc.encode(record['tokens'], {})[None], len(LABELS))[0]
    cols = {g: sorted({enc.token(t)[0] for t in ts}) for g, ts in groups.items()}
    out = []
    for e in ledger:
        scores = {}; reasons = {}; mapped = e['reviewed_available']
        ts = cited.get(e['explanation_id']); d = LABELS.index(e['diagnosis']) if e['target_supported'] else None
        if mapped:
            cs = set(ts)
            scores.update(fact_coverage=len(cs) / len(full), group_coverage=coverage({g: 1. for g in groups}, groups, cs))
            cp = fp['cohd']
            w = ({g: max(0., float(cp[d] - cache['cohd'][v][d])) for g, v in minus.items()}
                 if d is not None and cp is not None and all(cache['cohd'][v] is not None for v in minus.values()) else None)
            scores['cohd_coverage'] = coverage(w, groups, cs) if w is not None else None
            reasons['cohd_coverage'] = (None if scores['cohd_coverage'] is not None else
                                       'zero_positive_mass' if w is not None else 'source_or_target_unavailable')
            positive = {g: max(0., float(shap[d, c].sum())) for g, c in cols.items()} if d is not None else None
            scores['tree_positive_shap'] = coverage(positive, groups, cs) if positive is not None else None
            reasons['tree_positive_shap'] = None if scores['tree_positive_shap'] is not None else 'zero_positive_mass_or_target_unavailable'
            p = cache['lightgbm'][ts]
            scores['lightgbm_absolute'] = float(p[d]) if d is not None and p is not None else None
        for probe in ['encoder', 'lightgbm']:
            pp = cache[probe][ts] if mapped else None; available = pp is not None and fp[probe] is not None
            out.append(dict(explanation_id=e['explanation_id'], patient_id=pid, subject_group=e['subject_group'],
                            probe=probe, generator=e.get('generator'), diagnosis=e['diagnosis'], reference_diagnosis=record['label'],
                            diagnosis_correct=e['diagnosis_correct'], target_supported=e['target_supported'],
                            mapping_available=mapped, scan_status=e['repair_status'], repair_status=e['repair_status'],
                            probe_available=available, both_full_correct=both,
                            full_correct=bool(fp[probe] is not None and LABELS[int(fp[probe].argmax())] == record['label']),
                            reference_failure=bool(LABELS[int(pp.argmax())] != record['label']) if available else None,
                            proposal_failure=bool(LABELS[int(pp.argmax())] != e['diagnosis']) if available else None,
                            cited_fact_count=len(ts) if mapped else None,
                            full_probabilities=fp[probe].tolist() if fp[probe] is not None else None,
                            cited_probabilities=pp.tolist() if pp is not None else None,
                            scores=scores, coverage_unavailability=reasons))
    return pid, out

def parallel_score(tasks, spec, folder, workers, binding, label):
    folder.mkdir(parents=True, exist_ok=True)
    locked(folder / 'binding.json', binding)
    done = {}; pending = []
    for task in tasks:
        pid = task[0]['patient_id']; p = folder / ('patient_' + str(pid) + '.json')
        if p.exists() and p.with_suffix('.lock').exists():
            r = read(p)
            if r['patient_id'] != pid or sha(p) != read(p.with_suffix('.lock'))['sha256']:
                raise ValueError('Score cache changed')
            done[pid] = r['rows']
        else:
            pending.append(task)
    initial = len(done); start = time.monotonic(); last = 0.
    print(label, 'CPU scoring', workers, 'processes x 1 thread;', initial, '/', len(tasks), 'patients cached', flush=True)
    if pending:
        with cf.ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                    mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(spec,)) as pool:
            active = {pool.submit(score_patient, t) for t in pending}
            while active:
                finished, active = cf.wait(active, timeout=20, return_when=cf.FIRST_COMPLETED)
                for future in finished:
                    pid, rs = future.result(); p = folder / ('patient_' + str(pid) + '.json')
                    dump(p, dict(patient_id=pid, rows=rs)); dump(p.with_suffix('.lock'), dict(sha256=sha(p))); done[pid] = rs
                if time.monotonic() - last >= 20 or not active:
                    elapsed = time.monotonic() - start; speed = (len(done) - initial) / max(elapsed, .001)
                    progress = dict(stage=label, patients_done=len(done), total_patients=len(tasks), cached=initial,
                                    workers=workers, elapsed_seconds=round(elapsed, 1),
                                    eta_seconds=(len(tasks) - len(done)) / speed if speed else None)
                    dump(folder / 'progress.json', progress); print(progress, flush=True); last = time.monotonic()
    return [r for t in tasks for r in done[t[0]['patient_id']]]
