"""CPU development pilot. No API calls; no official test-set access."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_subset_study import build
from code_eval.cohort_stats import identity
from code_eval.tree_subset import Encoding, pattern, checked_contributions, metrics
from code_eval.subset_selectors import (EvidenceSupport, leave_group_out,
    random_order, ranked_subset, evidence_support_record)
import numpy as np
import lightgbm as lgb


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def emit(path, values):
    Path(path).write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False)+'\n' for v in values), encoding='utf8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def select_bank(bank, forbidden, fit_cap, dev_cap, seed):
    m = read(bank / 'manifest.json')
    if m['status'] != 'complete' or len(m['labels']) < 2:
        raise ValueError('Incomplete bank')
    selected, audit, occupied = {}, {}, set()
    for role, cap in [('teacher_fit', fit_cap), ('fusion', dev_cap)]:
        path = bank / (role + '.jsonl')
        if sha(path) != m['artifacts_sha256'][path.name]:
            raise ValueError('Bank hash mismatch: ' + role)
        source = rows(path)
        ids, patterns = set(), set()
        for r in source:
            if r['input_group_id'] in ids or r['pattern_id'] != pattern(r['tokens']) or r['label'] not in m['labels']:
                raise ValueError('Bank identity/pattern/label mismatch')
            ids.add(r['input_group_id']); patterns.add(r['pattern_id'])
        if patterns & occupied:
            raise ValueError('Fitting and development patterns overlap')
        occupied.update(patterns)
        kept = [r for r in source if r['pattern_id'] not in forbidden]
        chosen = []
        for label in m['labels']:
            candidates = [r for r in kept if r['label'] == label]
            candidates.sort(key=lambda r: hashlib.sha256(f"{seed}|{r['input_group_id']}".encode()).hexdigest())
            if not candidates:
                raise ValueError('No remaining examples for ' + label)
            chosen.extend(candidates[:cap])
        selected[role] = chosen
        audit[role] = dict(source=len(source), overlap_removed=len(source)-len(kept),
                          selected=len(chosen), by_class=dict(Counter(r['label'] for r in chosen)))
    return m['labels'], selected, audit


def recover_context(prepared, selected):
    m = read(prepared / 'manifest.json')
    path = prepared / 'train.csv'
    if m['status'] != 'complete' or sha(path) != m['artifacts_sha256']['train.csv']:
        raise ValueError('Prepared train provenance mismatch')
    wanted = {r['input_group_id']: r for rs in selected.values() for r in rs}
    found = set()
    with path.open(encoding='utf-8-sig', newline='') as f:
        for n, raw in enumerate(csv.DictReader(f), 1):
            tokens = ast.literal_eval(raw['EVIDENCES'])
            key = identity(float(raw['AGE']), raw['SEX'], tokens)
            if key in wanted:
                r = wanted[key]
                if key in found or r['label'] != raw['PATHOLOGY'] or set(tokens) != set(r['tokens']):
                    raise ValueError('Prepared row disagrees with bank')
                r['context'] = dict(age=float(raw['AGE']), sex=raw['SEX'])
                found.add(key)
            if n % 100000 == 0:
                print('Context recovery scanned', n, 'matched', len(found), '/', len(wanted), flush=True)
    if found != set(wanted):
        raise ValueError('Missing contexts: ' + str(len(set(wanted)-found)))


def fit_models(encoding, labels, bank, args):
    train, dev = bank['teacher_fit'], bank['fusion']
    fractions = [0, .25, .5, .75]
    full = np.stack([encoding.encode(r['tokens'], r['context']) for r in train])
    sparse = np.stack([encoding.encode(encoding.masked(r['tokens'], fractions[i % 4],
        f"{args.seed}|fit|{r['input_group_id']}"), r['context']) for i, r in enumerate(train)])
    y = np.array([labels.index(r['label']) for r in train])
    params = dict(objective='multiclass', num_class=len(labels), learning_rate=.05,
        num_leaves=31, min_data_in_leaf=20, verbosity=-1, num_threads=args.threads,
        seed=args.seed, deterministic=True, force_col_wise=True)
    models, results = {}, {}
    for condition, second in [('full_only', full), ('mask_augmented', sparse)]:
        print('Training', condition, 'rows', 2*len(train), 'rounds', args.rounds, flush=True)
        dataset = lgb.Dataset(np.concatenate([full, second]), label=np.concatenate([y, y]),
            categorical_feature=encoding.categorical, feature_name=[f'x{i}' for i in range(full.shape[1])])
        def progress(env):
            if (env.iteration+1) % 20 == 0:
                print(condition, 'round', env.iteration+1, '/', args.rounds, flush=True)
        model = lgb.train(params, dataset, num_boost_round=args.rounds, callbacks=[progress])
        model.save_model(str(args.out / (condition + '.txt')))
        models[condition] = model
        checks = {}
        for fraction in [0, .25, .5, .75, 1]:
            x = np.stack([encoding.encode(encoding.masked(r['tokens'], fraction,
                f"{args.seed}|development|{r['input_group_id']}"), r['context']) for r in dev])
            p = model.predict(x, num_threads=args.threads)
            checks[str(fraction)] = metrics(p, [labels.index(r['label']) for r in dev])
            np.savez_compressed(args.out / f'{condition}_keep{fraction}.npz',
                probabilities=p, labels=np.array(labels), ids=np.array([r['input_group_id'] for r in dev]))
        results[condition] = checks
        dump(args.out / 'sparse_validation.json', results)
        print(condition, json.dumps(checks), flush=True)
    return models, params


def score_explanations(model, encoding, labels, prepared, args):
    patient_rows, curve_rows, score_rows = [], [], []
    gain = model.feature_importance(importance_type='gain')
    by_patient = {}
    for e in prepared['explanations']:
        if e['scoring_eligible']:
            by_patient.setdefault(e['patient_id'], []).append(e)
    for n, patient in enumerate(prepared['patients'], 1):
        facts = {f['fact_id']: f for f in patient['facts']}
        context = patient['fixed_context']
        # Confirm dictionary and LANTERN define exactly the same dependency units.
        for fid, f in facts.items():
            if encoding.features[f['feature_id']]['group_id'] != f['group_id']:
                raise ValueError('Dictionary/record dependency group mismatch')
            code, *value = f['source_token'].split('_@_', 1)
            if code != f['feature_id'] or (value[0] if value else 'true') != str(f['value']):
                raise ValueError('Fact value and source token disagree')
        def predict(subset):
            x = encoding.encode([facts[f]['source_token'] for f in sorted(subset)], context)[None, :]
            return dict(zip(labels, map(float, model.predict(x, num_threads=1)[0])))
        source = EvidenceSupport(list(facts), patient['groups'], labels, predict, allow_empty=True)
        # Empty support is descriptive only: no automatic clinical sufficiency threshold.
        target = source.source_target()
        x = encoding.encode([f['source_token'] for f in facts.values()], context)[None, :]
        shap = checked_contributions(model, x, len(labels))[0]
        cols = {f: encoding.token(v['source_token'])[0] for f, v in facts.items()}
        groupcols = {g: sorted({cols[f] for f in fs}) for g, fs in source.groups.items()}
        lofo = leave_group_out(source)
        ci = labels.index(target)
        weights = dict(global_gain={g: float(gain[cs].sum()) for g, cs in groupcols.items()},
            local_absolute_shap={g: float(np.abs(shap[ci, cs]).sum()) for g, cs in groupcols.items()},
            absolute_leave_group_out={g: abs(v) for g, v in lofo['signed_effects'].items()})
        orders = {method: sorted(w, key=lambda g: (-w[g], g)) for method, w in weights.items()}
        orders['random'] = random_order(source, f'{args.seed}|{patient["patient_id"]}')
        full = source.distribution(source.facts)
        for method, order in orders.items():
            for fraction in [.25, .5, .75, 1]:
                selected = ranked_subset(source, order, math.ceil(fraction*len(order)))
                distribution = source.distribution(selected['fact_ids'])
                curve_rows.append(dict(patient_id=patient['patient_id'], selector=method, fraction=fraction,
                    **selected, source_target=target, target_support=distribution[target],
                    prediction=max(labels, key=lambda d: distribution[d]),
                    full_prediction_agreement=max(labels, key=lambda d: distribution[d]) == target,
                    evaluation_role='SELF_SOURCE_MECHANISM_DIAGNOSTIC'))
        for e in by_patient.get(patient['patient_id'], []):
            cited = set(e['cited_fact_ids']); diagnosis = e['proposed_diagnosis']
            d = labels.index(diagnosis)
            full_d = full[diagnosis]
            effects = {g: full_d-source.distribution(source.facts-fs)[diagnosis] for g, fs in source.groups.items()}
            targetweights = dict(global_gain=weights['global_gain'],
                local_absolute_shap={g: float(np.abs(shap[d, cs]).sum()) for g, cs in groupcols.items()},
                local_positive_group_shap={g: max(0., float(shap[d, cs].sum())) for g, cs in groupcols.items()},
                positive_leave_group_out={g: max(0., v) for g, v in effects.items()})
            coverage = {}
            for method, w in targetweights.items():
                denominator = sum(w.values())
                coverage[method] = (sum(w[g]*len(fs & cited)/len(fs) for g, fs in source.groups.items())/denominator
                                    if denominator > 0 else None)
            score_rows.append(dict(explanation_id=e['explanation_id'], patient_id=patient['patient_id'],
                fact_count=len(facts), cited_fact_count=len(cited), equal_coverage=len(cited)/len(facts),
                equal_group_coverage=sum(len(fs & cited)/len(fs) for fs in source.groups.values())/len(source.groups),
                weighted_coverage=coverage, signed_group_effects=effects,
                **evidence_support_record(source, cited, diagnosis),
                semantic_mapping='AUTOMATIC_UNREVIEWED'))
        observed = set(cols.values())
        patient_rows.append(dict(patient_id=patient['patient_id'], source_target=target,
            full_distribution=full, context_only_distribution=source.distribution([]),
            signed_leave_group_out=lofo['signed_effects'], group_weights=weights,
            shap_target_bias=float(shap[ci, -1]), shap_context_sum=float(shap[ci, :2].sum()),
            shap_unobserved_sum=float(sum(shap[ci, j] for j in range(2, x.shape[1]) if j not in observed)),
            unique_queries=len(source.cache)))
        print('Scored patient', n, '/', len(prepared['patients']), 'queries', len(source.cache), flush=True)
    emit(args.out / 'patient_attributions.jsonl', patient_rows)
    emit(args.out / 'subset_curves.jsonl', curve_rows)
    emit(args.out / 'explanation_scores.jsonl', score_rows)
    return dict(patients=len(patient_rows), explanations=len(score_rows), subset_points=len(curve_rows),
        selectors=['random', 'global_gain', 'local_absolute_shap', 'absolute_leave_group_out'],
        api_calls=0, clinical_omission_labels=0, independent_rq_conclusion=False,
        deferred=['class_aggregate_shap', 'SIS_threshold_freeze', 'cross_source_evaluation'])


def run(args):
    print('Stage 1: verify shared records and cohort split', flush=True)
    prepared, summary = build(rows(args.records))
    encoding = Encoding(read(args.dictionary))
    forbidden = set()
    for p in prepared['patients']:
        pid = pattern([f['source_token'] for f in p['facts']])
        if pid != p['source_pattern_id']:
            raise ValueError('Explanation patient pattern mismatch')
        forbidden.add(pid)
    extra_patterns = getattr(args, 'exclude_patterns', None)
    if extra_patterns:
        extra = set(extra_patterns.read_text(encoding='utf8').splitlines())
        if any(len(s)!=64 or any(c not in '0123456789abcdef' for c in s) for s in extra):
            raise ValueError('Invalid exclusion pattern list')
        forbidden.update(extra)
    labels, bank, audit = select_bank(args.bank, forbidden, args.fit_per_class, args.dev_per_class, args.seed)
    if len(labels) != args.expected_classes:
        raise ValueError('Unexpected disease class count')
    dump(args.out / 'split_audit.json', dict(roles=audit, excluded_fit_patterns=len(forbidden),
        scored_patients=len(prepared['patients']),
        isolation='token_pattern_disjoint', scope='DEVELOPMENT_PILOT_NOT_FINAL_TEST'))
    print('Stage 2: recover fixed context from verified prepared train.csv', flush=True)
    recover_context(args.prepared, bank)
    for role, values in bank.items():
        emit(args.out / (role + '_selected_private.jsonl'), values)
    dump(args.out / 'encoding.json', encoding.export())
    dump(args.out / 'input_summary.json', summary)
    print('Stage 3: full-input and masked-input LightGBM', flush=True)
    models, params = fit_models(encoding, labels, bank, args)
    print('Stage 4: TreeSHAP checks, subset curves and natural explanation scores', flush=True)
    result = score_explanations(models['mask_augmented'], encoding, labels, prepared, args)
    return dict(result, labels=labels, model_params=params, seed=args.seed,
        rounds=args.rounds, lightgbm_version=lgb.__version__, numpy_version=np.__version__,
        python_version=sys.version,
        implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in
            [Path(__file__), ROOT/'scripts/prepare_subset_study.py',
             ROOT/'src/code_eval/tree_subset.py', ROOT/'src/code_eval/subset_selectors.py',
             ROOT/'src/code_eval/cohort_stats.py']},
        training='full plus one group-masked copy; full-only duplicates full input to match row count',
        source_sha256={str(p): sha(p) for p in [args.records, args.dictionary,
            args.bank/'manifest.json', args.prepared/'manifest.json'] + ([extra_patterns] if extra_patterns else [])})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, default=Path('../COSI_download/artifacts/cosi_go_nogo_v1/data'))
    p.add_argument('--prepared', type=Path, default=Path('../COSI_download/artifacts/cosi49_data_v01'))
    p.add_argument('--records', type=Path, default=Path('data/legacy50/records.jsonl'))
    p.add_argument('--dictionary', type=Path, default=Path('data/legacy50/evidence_dictionary.json'))
    p.add_argument('--exclude-patterns', type=Path,
        help='All research-cohort patterns, including partitions not scored in this run')
    p.add_argument('--out', type=Path, default=Path('artifacts/tree_subset_pilot_v1'))
    p.add_argument('--threads', type=int, default=32)
    p.add_argument('--rounds', type=int, default=120)
    p.add_argument('--fit-per-class', type=int, default=200)
    p.add_argument('--dev-per-class', type=int, default=20)
    p.add_argument('--expected-classes', type=int, default=49)
    p.add_argument('--seed', type=int, default=261006)
    args = p.parse_args()
    if min(args.threads, args.rounds, args.fit_per_class, args.dev_per_class) < 1:
        p.error('Positive counts required')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    status = dict(status='running', scope='development_pilot', arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    dump(args.out/'run_manifest.json', status)
    try:
        status.update(run(args), status='complete')
    except Exception:
        status.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        status['elapsed_seconds'] = time.time()-started
        status['output_sha256'] = {f.name: sha(f) for f in args.out.iterdir() if f.is_file() and f.name != 'run_manifest.json'}
        dump(args.out/'run_manifest.json', status)
        report = args.out.with_name(args.out.name+'_reports.tar.gz')
        with tarfile.open(report, 'w:gz') as tar:
            for f in args.out.iterdir():
                if f.suffix in ('.json', '.jsonl') and not f.name.endswith('_private.jsonl'):
                    tar.add(f, arcname=f.name)
        print('Return report:', report, flush=True)


if __name__ == '__main__':
    main()
