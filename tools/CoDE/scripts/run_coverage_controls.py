"""Development-only class SHAP and permutation controls; reuse frozen predictions."""
import argparse
import json
from pathlib import Path
import random
import sys
import tarfile
import traceback
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_lantern_cohort import read, rows, sha, dump


def coverage(weights, groups, cited):
    total = sum(weights.values())
    return sum(weights[g] * len(set(fs) & cited) / len(fs)
               for g, fs in groups.items()) / total if total > 0 else None


def permute(weights, key):
    keys = sorted(weights)
    values = [weights[k] for k in keys]
    random.Random(key).shuffle(values)
    return dict(zip(keys, values))


def run(a):
    import numpy as np
    import lightgbm as lgb
    from code_eval.tree_subset import Encoding, checked_contributions
    from code_eval.proxy_metrics import summarize
    tree = a.source / 'tree'; cohort = a.source / 'cohort'
    tm = read(tree / 'run_manifest.json'); pm = read(a.probe / 'run_manifest.json')
    if tm['status'] != 'complete' or pm['status'] != 'complete':
        raise ValueError('Incomplete upstream')
    if pm['source_tree_manifest_sha256'] != sha(tree / 'run_manifest.json'):
        raise ValueError('Tree/probe mismatch')
    for name in ['mask_augmented.txt', 'teacher_fit_selected_private.jsonl', 'encoding.json']:
        if sha(tree / name) != tm['output_sha256'][name]:
            raise ValueError('Changed tree input: ' + name)
    for name in ['development_records.jsonl', 'evidence_dictionary.json']:
        expected = [h for p, h in tm['source_sha256'].items() if Path(p).name == name]
        if expected != [sha(cohort / name)]: raise ValueError('Changed cohort input: ' + name)
    proxy_path = a.probe / 'development_proxy_records.jsonl'
    if sha(proxy_path) != pm['output_sha256'][proxy_path.name]:
        raise ValueError('Changed frozen probe outcomes')
    encoding = Encoding(read(cohort / 'evidence_dictionary.json'))
    if encoding.export() != read(tree / 'encoding.json'): raise ValueError('Encoding drift')
    labels = tm['labels']; model = lgb.Booster(model_file=str(tree / 'mask_augmented.txt'))
    train = list(rows(tree / 'teacher_fit_selected_private.jsonl'))
    sums = np.zeros((len(labels), len(encoding.columns))); counts = np.zeros(len(labels), dtype=int)
    print('Stage 1: class SHAP from frozen fitting patients only', len(train), flush=True)
    for start in range(0, len(train), 16):
        batch = train[start:start+16]
        x = np.stack([encoding.encode(r['tokens'], r['context']) for r in batch])
        values = checked_contributions(model, x, len(labels))
        for i, r in enumerate(batch):
            d = labels.index(r['label']); sums[d] += np.abs(values[i, d, :-1]); counts[d] += 1
        if start % 160 == 0: print('Class SHAP', min(start+16, len(train)), '/', len(train), flush=True)
    if np.any(counts == 0): raise ValueError('Empty training class')
    means = sums / counts[:, None]
    dump(a.out / 'class_shap.json', dict(labels=labels, columns=encoding.columns,
        counts=counts.tolist(), mean_absolute=means.tolist(),
        definition='Mean absolute class-d logit SHAP over full fitting records whose reference label is d; no development labels'))
    records = list(rows(cohort / 'development_records.jsonl'))
    proxies = {r['explanation_id']: r for r in rows(proxy_path)}
    if set(proxies) != {r['explanation_id'] for r in records}: raise ValueError('Incomplete joins')
    grouped = {}
    for r in records: grouped.setdefault(r['patient_id'], []).append(r)
    output = []
    print('Stage 2: score natural citations; reuse independent probe outcomes', flush=True)
    for n, (pid, rs) in enumerate(grouped.items(), 1):
        first = rs[0]; facts = {f['fact_id']: f for f in first['facts']}; groups = first['groups']
        tokens = {fid: f['provenance']['source_token'] for fid, f in facts.items()}
        x = encoding.encode(list(tokens.values()), first['demographics'])[None, :]
        values = checked_contributions(model, x, len(labels))[0]
        cols = {g: sorted({encoding.token(tokens[f])[0] for f in fs}) for g, fs in groups.items()}
        for r in rs:
            if r['facts'] != first['facts'] or r['groups'] != groups or r['demographics'] != first['demographics']:
                raise ValueError('Within-patient input drift')
            p = proxies[r['explanation_id']]
            if p['patient_id'] != pid or p['diagnosis'] != r['ranked_diagnoses'][0]:
                raise ValueError('Proxy join drift')
            d = labels.index(p['diagnosis']); cited = set(r['provisional_cited_fact_ids'])
            local = {g: max(0., float(values[d, cs].sum())) for g, cs in cols.items()}
            actual = coverage(local, groups, cited); saved = p['scores']['local_positive_group_shap']
            if (actual is None) != (saved is None) or (actual is not None and abs(actual-saved)>1e-8):
                raise ValueError('Local SHAP does not reproduce saved coverage')
            scores = dict(p['scores'])
            scores['class_mean_absolute_shap'] = coverage({g: float(means[d, cs].sum()) for g, cs in cols.items()}, groups, cited)
            for repeat in range(a.repeats):
                # Same permutation for all explanations of a patient/diagnosis.
                scores[f'permuted_local_{repeat:02d}'] = coverage(
                    permute(local, f'{a.seed}|{pid}|{p["diagnosis"]}|{repeat}'), groups, cited)
            output.append(dict(p, scores=scores))
        if n % 50 == 0: print('Scored patients', n, '/', len(grouped), flush=True)
    with (a.out / 'coverage_proxy_records.jsonl').open('w', encoding='utf8') as f:
        for r in output: f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
    methods = ['words', 'fact_coverage', 'group_coverage', 'global_gain', 'local_positive_group_shap',
               'class_mean_absolute_shap', 'positive_leave_group_out', 'absolute_support', 'retained_support']
    methods += [f'permuted_local_{i:02d}' for i in range(a.repeats)]
    panels = {}
    print('Stage 3: common-case paired evaluation; all permutation repeats reported', flush=True)
    for component in ['all', 'coverage', 'challenge']:
        for target in ['correct_proposal', 'all_proposals']:
            rs = [r for r in output if (component == 'all' or r['component'] == component)
                  and (target == 'all_proposals' or r['diagnosis_correct'])]
            panel = summarize(rs, methods, a.bootstrap, a.seed)
            panel['input_explanations'] = len(rs)
            panels[component+'|'+target] = panel
            print(component, target, 'common', panel['common'], flush=True)
    dump(a.out / 'rule_comparison.json', panels)
    return dict(status='complete', scope='development coverage controls; not additional cohort sources',
        fit_patients=len(train), explanations=len(output), models_trained=0, api_calls=0,
        evaluation_scoring=False, seed=a.seed, permutation_repeats=a.repeats,
        bootstrap=a.bootstrap, lightgbm_version=lgb.__version__,
        source_tree_sha256=sha(tree/'run_manifest.json'), source_probe_sha256=sha(a.probe/'run_manifest.json'),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/code_eval/tree_subset.py',ROOT/'src/code_eval/proxy_metrics.py']},
        notes=['Class weights are an explicit mechanism adaptation, not exact PPMI reproduction',
               'All permutation replicates retained; no best replicate selection',
               'Common-case population may change when adding methods; use within-batch paired comparisons'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('artifacts/lantern_development_v1'))
    p.add_argument('--probe', type=Path, default=Path('artifacts/encoder_probe_v1'))
    p.add_argument('--out', type=Path, default=Path('artifacts/coverage_controls_v1'))
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--seed', type=int, default=261006)
    a = p.parse_args()
    if min(a.repeats,a.bootstrap)<1: p.error('Positive repeats required')
    a.out.mkdir(parents=True, exist_ok=False); result={}; start=time.time()
    try: result=run(a)
    except Exception: result=dict(status='failed',error=traceback.format_exc()); raise
    finally:
        result['elapsed_seconds']=time.time()-start
        result['output_sha256']={f.name:sha(f) for f in a.out.iterdir() if f.is_file()}
        dump(a.out/'run_manifest.json',result)
        archive=a.out.with_name(a.out.name+'_reports.tar.gz')
        with tarfile.open(archive,'w:gz') as t:
            for f in a.out.iterdir(): t.add(f,arcname=f.name)
        print('Return report:',archive,flush=True)


if __name__ == '__main__': main()
