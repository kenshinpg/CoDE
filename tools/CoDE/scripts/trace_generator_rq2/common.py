"""Shared I/O and fixed TRACE second-generator protocol."""
import csv
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']
METHODS = ['cohd_coverage', 'group_coverage', 'tree_positive_shap', 'lightgbm_absolute']
COMPARISONS = [(METHODS[0], METHODS[1]), (METHODS[2], METHODS[1]), (METHODS[2], METHODS[3])]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]

def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def dump(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(5):
        try:
            os.replace(tmp, p)
            break
        except PermissionError:
            # Windows indexing may briefly hold the replaced file; Linux errors remain immediate.
            if os.name != 'nt' or attempt == 4:
                raise
            time.sleep(.05 * (attempt + 1))

def emit(p, records):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, p)

def table(p, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader()
        for r in records:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in r.items()})

def locked(p, value):
    p = Path(p)
    if p.exists():
        if read(p) != value:
            raise ValueError('Resume binding changed: ' + str(p))
    else:
        dump(p, value)

def index(records, key):
    result = {r[key]: r for r in records}
    if len(result) != len(records):
        raise ValueError('Duplicate ' + key)
    return result

def local_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

def cpu_count():
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1

def make_tasks(patients, split, model):
    """Input labels remain outside messages; four fixed requests per test admission."""
    from .original_generation import messages
    result = []
    for pid in sorted(patients):
        if split[pid]['partition'] != 'test':
            continue
        p = patients[pid]
        for j in range(4):
            old_id = 'TRACE_' + digest([p['view_id'], pid, j])[:24]
            result.append(dict(task_id='TRACE27_' + digest([p['view_id'], pid, j, model])[:24],
                               baseline_task_id=old_id, patient_id=pid, rollout_index=j,
                               input_view_id=p['view_id'], input_digest=digest(p), messages=messages(p)))
    return result

def analysis_census(rs):
    failed = [r for r in rs if r.get('reference_failure') is True]
    return dict(explanations=len(rs), patients=len({r['patient_id'] for r in rs}),
                failures=len(failed), failure_patients=len({r['patient_id'] for r in failed}))
