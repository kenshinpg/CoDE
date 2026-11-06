"""TRACE second generator: frozen 359 x 4 inputs, unchanged mapping, three RQ2 contrasts."""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
import argparse
import collections
import importlib.metadata
import json
from pathlib import Path
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from trace_generator_rq2.common import (read, rows, dump, emit, table, locked, sha, digest, index,
                                       local_path, make_tasks, cpu_count, LABELS)


def preflight(a):
    import numpy, torch, lightgbm, pydantic
    from code_eval.trace2400.encoding import TraceEncoding
    c = read(a.config)
    if a.endpoints_file:
        from trace_generator_rq2.multi import endpoints
        nodes = endpoints(a.endpoints_file)
        print('Configured API nodes:', len(nodes), '; total task workers:', len(nodes)*a.workers, flush=True)
    if a.target_minutes <= a.local_reserve_minutes or a.local_reserve_minutes < 0:
        raise ValueError('Target duration must exceed the local-work reserve')
    if a.model != 'Qwen3.8-27B' or a.workers != 128 or a.bootstrap != 5000 or a.seed != 261016:
        raise ValueError('This package fixes model=Qwen3.8-27B, API workers=128, bootstrap=5000, seed=261016')
    if min(a.cpu_workers, a.summary_workers) < 1 or max(a.cpu_workers, a.summary_workers) > cpu_count():
        raise ValueError('Requested CPU processes exceed available cores; set --cpu-workers/--summary-workers explicitly')
    paths = {k: local_path(v) for k, v in c['paths'].items()}
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError('Required accepted artifact missing: ' + str(p))
    view = paths['view']; manifest = read(view / 'manifest.json')
    names = ['patients_model_input_PRIVATE.jsonl', 'test_records_PRIVATE.jsonl', 'split_PRIVATE.jsonl', 'evidence_dictionary.json']
    hashes = {}
    for n in names:
        actual = sha(view / n)
        if manifest['output_sha256'].get(n) != actual:
            raise ValueError('Frozen TRACE view changed: ' + n)
        hashes[str(view / n)] = actual
    if hashes[str(view / names[0])] != c['frozen_patients_sha256']:
        raise ValueError('Generation F differs from original Qwen3-4B input')
    if sha(paths['baseline_scores']) != c['baseline_scores_sha256']:
        raise ValueError('Accepted Qwen3-4B score cache changed')
    hashes[str(paths['baseline_scores'])] = sha(paths['baseline_scores'])
    hashes[str(paths['baseline_ledger'])] = sha(paths['baseline_ledger'])
    split = index(rows(view / 'split_PRIVATE.jsonl'), 'admission_id')
    patients = index(rows(view / 'patients_model_input_PRIVATE.jsonl'), 'patient_id')
    records = index(rows(view / 'test_records_PRIVATE.jsonl'), 'patient_id')
    test_ids = {p for p, s in split.items() if s['partition'] == 'test'}
    if len(test_ids) != 359 or set(records) != test_ids or not test_ids <= patients.keys():
        raise ValueError('TRACE 359-patient census mismatch')
    by_subject = collections.defaultdict(set)
    for s in split.values():
        by_subject[s['subject_group']].add(s['partition'])
    if any(len(v) > 1 for v in by_subject.values()):
        raise ValueError('Subject crosses partitions')
    enc = TraceEncoding(read(view / 'evidence_dictionary.json'))
    for pid, r in records.items():
        p = patients[pid]; s = split[pid]
        if p['context'] or r['context'] or r['label'] != s['reference_class'] or r['partition'] != 'test':
            raise ValueError('Test input metadata mismatch')
        if len({f['fact_id'] for f in p['facts']}) != len(p['facts']):
            raise ValueError('Duplicate fact ID')
        if {f['feature_id'] + '_@_' + f['state'] for f in p['facts']} != set(r['tokens']):
            raise ValueError('Facts/tokens mismatch')
        enc.encode(r['tokens'], {})
    tasks = make_tasks(patients, split, a.model)
    if len(tasks) != 1436 or sum((len(patients[t['patient_id']]['facts']) + 7)//8 for t in tasks) != 12784:
        raise ValueError('Frozen task or matching-group census mismatch')
    old_ledger = rows(paths['baseline_ledger']); old = rows(paths['baseline_scores'])
    if len(old_ledger) != 1436 or len(old) != 2872:
        raise ValueError('Accepted baseline task census mismatch')
    old_index = index(old_ledger, 'explanation_id')
    if set(old_index) != {t['baseline_task_id'] for t in tasks}:
        raise ValueError('Task identity rule differs from accepted 4B tasks')
    for t in tasks:
        e = old_index[t['baseline_task_id']]
        if e['patient_id'] != t['patient_id'] or e['subject_group'] != split[t['patient_id']]['subject_group']:
            raise ValueError('Baseline patient/subject join differs')
    fm = read(paths['models'] / 'fitted_sources.json'); model_paths = {}
    for n, expected in c['model_sha256'].items():
        p = local_path(fm['models'][n]['path'])
        if fm['models'][n]['sha256'] != expected or sha(p) != expected:
            raise ValueError('Frozen source changed: ' + n)
        model_paths[n] = str(p); hashes[str(p)] = expected
    package = Path(__file__).parent / 'trace_generator_rq2'
    code = {str(p.relative_to(ROOT)): sha(p) for p in sorted(package.glob('*.py'))}
    code[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    for n, expected in c['reused_code_sha256'].items():
        if sha(ROOT / n) != expected:
            raise ValueError('Accepted implementation differs: ' + n + '; do not overwrite it blindly')
        code[n] = expected
    binding = dict(config=sha(a.config), code=code, files=hashes, tasks_digest=digest(tasks),
                   model=a.model, endpoint='multi_service_pool' if a.endpoints_file else a.endpoint,
                   model_dir=str(a.model_dir), api_workers=a.workers,
                   generation_reference=c.get('generation_reference'),infrastructure_recovery=c.get('infrastructure_recovery'),
                   bootstrap=a.bootstrap, seed=a.seed, generator_parameters=dict(temperature=.7, top_p=.8,
                   max_tokens=2048, thinking=False, attempts=3), grouping='8 facts; unchanged full_v1 prompts and fallback')
    locked(a.out / 'input_binding.json', binding)
    environment = dict(python=sys.version, cpu_count=cpu_count(), score_processes=a.cpu_workers,
                       summary_processes=a.summary_workers, thread_limit=1,
                       packages={n: importlib.metadata.version(n) for n in ['numpy', 'torch', 'lightgbm', 'pydantic']})
    dump(a.out / 'environment.json', environment)
    emit(a.out / 'generation_tasks_PRIVATE.jsonl', tasks)
    spec = dict(dictionary=str(view / 'evidence_dictionary.json'), models=model_paths)
    return c, patients, split, records, tasks, old_ledger, old, spec


def patient_jobs(patients, records, ledger):
    by = collections.defaultdict(list)
    for e in ledger:
        by[e['patient_id']].append(e)
    return [(patients[pid], records[pid], sorted(by[pid], key=lambda r: r['explanation_id'])) for pid in sorted(by)]


def replay(a, patients, records, ledger, old, spec):
    from trace_generator_rq2.scoring import parallel_score
    from code_eval.adjudication_sensitivity import automatic_gate
    baseline = [dict(e, generator='Qwen3-4B') for e in ledger]
    scored = parallel_score(patient_jobs(patients, records, baseline), spec, a.out / 'baseline_replay', a.cpu_workers,
                            dict(input_binding=sha(a.out / 'input_binding.json'), ledger=digest(baseline)), '4B replay')
    previous = {(r['explanation_id'], r['probe']): r for r in old}
    if len(previous) != len(old) or {(r['explanation_id'], r['probe']) for r in scored} != set(previous):
        raise ValueError('Replay row census mismatch')
    maximum = 0.
    for r in scored:
        maximum = max(maximum, automatic_gate(r, previous[r['explanation_id'], r['probe']], tolerance=2e-5))
    dump(a.out / 'baseline_replay_audit.json', dict(status='passed', rows=len(scored), patients=len(records),
         score_tolerance=2e-5, maximum_core_score_difference=maximum,
         endpoint_changes=0, fitted_models_changed=False,
         note='CPU adapter checks all accepted baseline rows before new generation. Baseline statistics retain the accepted score cache.'))
    print('4B replay PASSED:', len(scored), 'rows; maximum core difference', maximum, flush=True)


def matching_tasks(tasks, generations, patients):
    result = []
    for task in tasks:
        r = generations[task['task_id']]
        if r['status'] != 'complete':
            continue
        facts = patients[task['patient_id']]['facts']
        for start in range(0, len(facts), 8):
            result.append(dict(id=task['task_id'] + f'_g{start//8:04d}', task_id=task['task_id'],
                               patient_id=task['patient_id'], explanation=r['parsed_explanation'],
                               facts=[{k: f[k] for k in ['fact_id', 'feature_id', 'state', 'display_text']}
                                      for f in facts[start:start+8]]))
    return result


def make_ledger(tasks, generations, matching, patients, split):
    by = collections.defaultdict(list)
    for r in matching:
        by[r['task_id']].append(r)
    result = []; matches = []
    for t in tasks:
        g = generations[t['task_id']]; pid = t['patient_id']; s = split[pid]
        valid = g['status'] == 'complete'; rs = by[t['task_id']]; facts = patients[pid]['facts']
        n = (len(facts) + 7)//8
        scanned = valid and len(rs) == n and all(r['status'] == 'COMPLETE' for r in rs)
        decisions = [d for r in rs if r['status'] == 'COMPLETE' for d in r['parsed']['decisions']]
        if scanned and (len(decisions) != len(facts) or {d['fact_id'] for d in decisions} != {f['fact_id'] for f in facts}):
            raise ValueError('Full visible fact census mismatch')
        for d in decisions:
            for q in d['quotes']:
                if g['parsed_explanation'][q['start']:q['end']] != q['text']:
                    raise ValueError('Matching quote changed')
        ambiguous = any(d['status'] == 'ambiguous' for d in decisions)
        status = 'UNAVAILABLE' if not valid else ('COMPLETE' if scanned and not ambiguous else 'PARTIAL')
        reason = 'generation_failed' if not valid else ('matching_groups_incomplete' if not scanned else
                 'ambiguous_decisions' if ambiguous else None)
        failed_groups = sum(r['status'] != 'COMPLETE' for r in rs)
        observed_exact = sorted({d['fact_id'] for d in decisions if d['status'] == 'exact'})
        diagnosis = g['parsed_diagnoses'][0] if valid else None
        cited = sorted({d['fact_id'] for d in decisions if d['status'] == 'exact'}) if scanned else None
        result.append(dict(explanation_id=t['task_id'], patient_id=pid, subject_group=s['subject_group'],
                           partition='test', reference_diagnosis=s['reference_class'], diagnosis=diagnosis,
                           diagnosis_correct=diagnosis == s['reference_class'], target_supported=diagnosis in LABELS,
                           reviewed_available=scanned, repair_status=status, cited_fact_ids=cited,
                           generation_status=g['status'], visible_fact_scan_status=status, scan_complete=scanned,
                           analysis_mapping_available=scanned and not ambiguous, unavailability_reason=reason,
                           generator='Qwen3.8-27B', word_count=len(g['parsed_explanation'].split()) if valid else None))
        matches.append(dict(task_id=t['task_id'], patient_id=pid, status=status, decisions=decisions,
                            generation_status=g['status'], visible_fact_scan_status=status, scan_complete=scanned,
                            analysis_mapping_available=scanned and not ambiguous, unavailability_reason=reason,
                            observed_exact_fact_ids=observed_exact if valid else None,
                            observed_exact_is_complete=scanned, failed_groups=failed_groups,
                            missing_groups=max(0,n-len(rs)) if valid else 0, expected_groups=n if valid else 0))
    return result, matches


def audit_sample(a, tasks, generations, patients, split, matches):
    selected = []; by_pid = collections.defaultdict(list)
    for t in tasks:
        by_pid[t['patient_id']].append(t)
    for disease in LABELS:
        ids = sorted([pid for pid in by_pid if split[pid]['reference_class'] == disease],
                     key=lambda pid: digest(['new-generator-audit-261027', pid]))[:8]
        for pid in ids:
            selected.append(sorted(by_pid[pid], key=lambda t: digest(['audit-rollout', t['task_id']]))[0])
    # Membership uses only fixed disease strata and IDs, including generation/mapping failures.
    lookup = index(matches, 'task_id'); export = []
    for i, t in enumerate(selected, 1):
        g = generations[t['task_id']]
        export.append(dict(audit_id=f'G{i:03d}', task_id=t['task_id'], patient_id=t['patient_id'],
                           facts=patients[t['patient_id']]['facts'], explanation=g['parsed_explanation'],
                           generation_status=g['status'], mapping=lookup[t['task_id']],
                           review_status='pending', instructions='Check mismatched and missed citations against every visible fact; no probe outcomes shown.'))
    emit(a.out / 'mapping_spotcheck_32_PRIVATE.jsonl', export)
    dump(a.out / 'mapping_spotcheck_plan.json', dict(sample_size=32, diseases=4, per_disease=8,
         selected_before_scores=True, selection='fixed hashes of IDs; one explanation per selected admission; failures retained',
         status='prepared_not_human_reviewed'))


def run(a):
    budget_start = time.monotonic()
    print('Stage 1/6: frozen inputs, checkpoints and dependencies', flush=True)
    c, patients, split, records, tasks, old_ledger, old, spec = preflight(a)
    print('Stage 2/6: replay existing 4B scores with CPU adapter', flush=True)
    replay(a, patients, records, old_ledger, old, spec)
    if a.preflight_only:
        return dict(status='preflight_passed', api_generation_calls=0)
    from trace_generator_rq2.service import service, Client, queue
    api_pool = None
    if a.endpoints_file:
        from trace_generator_rq2.multi import setup
        api_pool = setup(a.endpoints_file, a.model, a.model_dir, a.out, budget_start,
                         a.target_minutes, a.local_reserve_minutes)
        total_workers = a.workers * len(api_pool.nodes)
        api_pool.begin('generate')
        api_pool.warmup(tasks, 'task_id', a.out / 'generation_results_PRIVATE')
        generate = api_pool.execute
    else:
        limit = service(a.endpoint, a.model, a.model_dir, a.out)
        client = Client(a.endpoint, a.model, limit, a.out)
        total_workers = a.workers; generate = client.generate
    print('Stage 3/6: generate 1436 explanations; total API workers=', total_workers, flush=True)
    gens = queue(tasks, 'task_id', generate, a.out / 'generation_results_PRIVATE', total_workers, 'generation',
                 progress_hook=api_pool.progress if api_pool else None, verify_first=api_pool is None)
    generations = index(gens, 'task_id')
    locked(a.out / 'generation_complete.json', dict(records=digest(gens), tasks=digest(tasks)))
    emit(a.out / 'generation_records_PRIVATE.jsonl', gens)
    generation_summary = dict(planned=1436, statuses=dict(collections.Counter(g['status'] for g in gens)),
                              admissions=359, by_disease={})
    for d in LABELS:
        rr = [g for g in gens if split[g['patient_id']]['reference_class'] == d]
        generation_summary['by_disease'][d] = dict(planned=len(rr), valid=sum(r['status']=='complete' for r in rr),
            correct_top1=sum(r['status']=='complete' and r['parsed_diagnoses'][0]==d for r in rr))
    dump(a.out / 'generation_summary.json', generation_summary)
    print('Stage 4/6: whole-explanation visible-fact matching; total API workers=', total_workers, flush=True)
    mt = matching_tasks(tasks, generations, patients)
    locked(a.out / 'matching_input_lock.json', dict(tasks_digest=digest(mt), tasks=len(mt)))
    if api_pool:
        api_pool.begin('match')
        api_pool.warmup(mt, 'id', a.out / 'matching_results_PRIVATE')
    matched = queue(mt, 'id', api_pool.execute if api_pool else client.match,
                    a.out / 'matching_results_PRIVATE', total_workers, 'matching',
                    progress_hook=api_pool.progress if api_pool else None, verify_first=api_pool is None)
    locked(a.out / 'matching_complete.json', dict(records=digest(matched)))
    ledger, matches = make_ledger(tasks, generations, matched, patients, split)
    emit(a.out / 'explanation_ledger_PRIVATE.jsonl', ledger)
    emit(a.out / 'matching_candidates_PRIVATE.jsonl', matches)
    dump(a.out / 'mapping_summary.json', dict(planned_explanations=1436, matching_groups=len(mt),
         group_statuses=dict(collections.Counter(r['status'] for r in matched)),
         explanation_statuses=dict(collections.Counter(r['repair_status'] for r in ledger)),
         semantic_accuracy='not estimated by completion status'))
    audit_sample(a, tasks, generations, patients, split, matches)
    print('Stage 5/6: score cited sets, frozen models; CPU processes=', a.cpu_workers, flush=True)
    from trace_generator_rq2.scoring import parallel_score
    scored = parallel_score(patient_jobs(patients, records, ledger), spec, a.out / 'new_scores', a.cpu_workers,
                            dict(input_binding=sha(a.out / 'input_binding.json'), ledger=digest(ledger)), '27B scoring')
    if len(scored) != 2872 or len({(r['explanation_id'], r['probe']) for r in scored}) != 2872:
        raise ValueError('New score row census mismatch')
    baseline = [dict(r, generator='Qwen3-4B') for r in old]
    all_records = baseline + scored
    emit(a.out / 'scores_PRIVATE.jsonl', all_records)
    print('Stage 6/6: three paired AP contrasts; 5000 subject-cluster draws; processes=', a.summary_workers, flush=True)
    from trace_generator_rq2.statistics import summarize
    result = summarize(all_records, a.out, a.summary_workers, a.bootstrap, a.seed)
    return dict(status='complete', admissions=359, new_explanations=1436, matching_groups=len(mt),
                new_score_rows=2872, contrast_rows=len(result), primary_contrasts=sum(r['primary'] for r in result),
                generators=['Qwen3-4B', 'Qwen3.8-27B'], models_trained=0, source_facts_changed=False,
                independent_mapping_spotcheck='pending; 32 fixed records exported')


def package(a):
    # The return archive contains numeric per-record evidence and the fixed audit subset.
    # Full clinical texts and raw API responses remain on the server.
    names = ['run_manifest.json', 'input_binding.json', 'environment.json', 'service_binding.json',
             'node_topology.json', 'node_failover.json',
             'READY.json', 'shard_plan.json', 'shard_acceptance.json', 'infrastructure_summary.json', 'measurement_flow.json',
             'baseline_replay_audit.json', 'generation_summary.json', 'mapping_summary.json',
             'generation_progress.json', 'matching_progress.json', 'bootstrap_progress.json',
             'statistics.json', 'population_flow.csv', 'three_comparisons_ALL.csv', 'three_comparisons_PRIMARY.csv',
             'scores_PRIVATE.jsonl', 'common_population_PRIVATE.json', 'mapping_spotcheck_plan.json',
             'mapping_spotcheck_32_PRIVATE.jsonl']
    target = a.out.parent / (a.out.name + '_reports.tar.gz')
    with tarfile.open(target, 'w:gz') as tf:
        for name in names:
            if (a.out / name).exists():
                tf.add(a.out / name, arcname=a.out.name + '/' + name)
        for p in [a.config, Path(__file__)]:
            tf.add(p, arcname=a.out.name + '/code/' + p.name)
        for p in sorted((a.out / 'node_services').glob('*/service_binding.json')):
            tf.add(p, arcname=a.out.name + '/' + p.relative_to(a.out).as_posix())
        for p in sorted((a.out / 'runtime').glob('nodes_*.json')):
            tf.add(p, arcname=a.out.name + '/' + p.relative_to(a.out).as_posix())
        for pattern in ['nodes/node_*/DONE.json','nodes/node_*/status.json','nodes/node_*/service_binding.json',
                        'nodes/node_*/*_progress.json','nodes/node_*/execution_summary.json',
                        'nodes/node_*/infrastructure_events_PRIVATE/*.json',
                        'nodes/node_*/infrastructure_events_summary_PRIVATE.jsonl',
                        'nodes/node_*/queue_errors_PRIVATE/*.json',
                        'logs/*.log','preparation/status.json','finalization/status.json']:
            for p in sorted(a.out.glob(pattern)):
                tf.add(p,arcname=a.out.name+'/'+p.relative_to(a.out).as_posix())
        for p in sorted((ROOT/'scripts/trace_generator_rq2').glob('*.py')):
            tf.add(p,arcname=a.out.name+'/code/trace_generator_rq2/'+p.name)
    print('Return report (controlled-access):', target, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/trace_generator_rq2_20261027.json')
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/trace_qwen27b_rq2_v1')
    p.add_argument('--endpoint', default=os.environ.get('QWEN_ENDPOINT', 'http://127.0.0.1:8802/v1/chat/completions'))
    p.add_argument('--endpoints-file', type=Path, help='Identical local model services; 128 task workers per service')
    p.add_argument('--target-minutes', type=float, default=120, help='Throughput forecast target, not a forced early stop')
    p.add_argument('--local-reserve-minutes', type=float, default=30, help='Time reserve for CPU scoring/statistics in ETA')
    p.add_argument('--model', default='Qwen3.8-27B')
    p.add_argument('--model-dir', type=Path, default=Path('models/Qwen3.8-27B'))
    p.add_argument('--workers', type=int, default=128)
    p.add_argument('--cpu-workers', type=int, default=128)
    p.add_argument('--summary-workers', type=int, default=128)
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--seed', type=int, default=261016)
    p.add_argument('--preflight-only', action='store_true', help='Replay frozen 4B scores without any LLM call')
    a = p.parse_args(); a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (a.out / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time(); state = {'status': 'failed'}
        try:
            state = run(a)
        except Exception:
            state['error'] = traceback.format_exc(); raise
        finally:
            state.update(started_at=started, elapsed_seconds=time.time()-started)
            dump(a.out / 'run_manifest.json', state); package(a)

if __name__ == '__main__':
    main()
