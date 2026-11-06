"""Three prespecified AP contrasts per generator/probe/proposal stratum."""
import concurrent.futures as cf
import multiprocessing
import time
from collections import Counter
import numpy as np
from .common import COMPARISONS, analysis_census, dump, table, locked, digest, sha, read
from code_eval.round1_followup import APContrast, intervals
from code_eval.proxy_metrics import average_precision

def panels(records):
    result = []; flows = []
    for generator in ['Qwen3-4B', 'Qwen3.8-27B']:
        for probe in ['encoder', 'lightgbm']:
            for stratum in ['correct', 'incorrect', 'all']:
                meta = dict(generator=generator, probe=probe, proposal_stratum=stratum,
                            primary=generator == 'Qwen3.8-27B' and probe == 'encoder' and stratum == 'correct')
                base = [r for r in records if r['generator'] == generator and r['probe'] == probe]
                flows.append(dict(meta, stage='planned', **analysis_census(base)))
                for name, predicate in [
                    ('complete_mapping', lambda r: r['mapping_available'] and r['scan_status'] == 'COMPLETE'),
                    ('supported_proposal', lambda r: r['target_supported']),
                    ('both_full_correct', lambda r: r['both_full_correct']),
                    ('probe_available', lambda r: r['probe_available']),
                    ('proposal_stratum', lambda r: stratum == 'all' or r['diagnosis_correct'] == (stratum == 'correct'))]:
                    base = [r for r in base if predicate(r)]
                    flows.append(dict(meta, stage=name, **analysis_census(base)))
                comparisons = []
                for left, right in COMPARISONS:
                    needed = sorted({left, right, 'fact_coverage', 'group_coverage'})
                    common = sorted([r for r in base if all(r['scores'].get(m) is not None for m in needed)],
                                    key=lambda r: r['explanation_id'])
                    comparisons.append(dict(left=left, right=right, records=common, eligible_before_common=len(base)))
                result.append(dict(meta, id=len(result), comparisons=comparisons))
    return result, flows

def block(task):
    panel, start, counts, keys = task
    cs = [APContrast(c['records'], c['left'], c['right'], 'subject_group') for c in panel['comparisons']]
    lookup = {k: i for i, k in enumerate(keys)}
    indices = [np.asarray([lookup[k] for k in c.cluster], int) for c in cs]
    values = np.full((len(counts), 3), np.nan)
    for i, w in enumerate(counts):
        for j, (c, ix) in enumerate(zip(cs, indices)):
            values[i, j] = c.evaluate(w[ix])
    return panel['id'], start, values

def summarize(records, out, workers, repetitions=5000, seed=261016):
    pp, flows = panels(records); table(out / 'population_flow.csv', flows)
    directory = out / 'bootstrap'; directory.mkdir(exist_ok=True)
    locked(directory / 'binding.json', dict(records=digest(records), repetitions=repetitions, seed=seed,
                                            policy='three contrasts jointly within each generator/probe/stratum'))
    arrays = {}; jobs = []; completed = 0; common = []
    for p in pp:
        keys = sorted({r['subject_group'] for c in p['comparisons'] for r in c['records']})
        arrays[p['id']] = np.full((repetitions, 3), np.nan)
        rng = np.random.default_rng(seed)
        for c in p['comparisons']:
            common.append(dict(generator=p['generator'], probe=p['probe'], stratum=p['proposal_stratum'],
                               left=c['left'], right=c['right'], explanation_ids=[r['explanation_id'] for r in c['records']]))
        for start in range(0, repetitions, 250):
            n = min(250, repetitions - start)
            counts = np.asarray([np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))
                                 for _ in range(n)], dtype=np.int32) if keys else np.zeros((n, 0), np.int32)
            dest = directory / f'panel_{p["id"]:02d}_{start:05d}.npz'
            if dest.exists() and dest.with_suffix('.json').exists():
                if sha(dest) != read(dest.with_suffix('.json'))['sha256']:
                    raise ValueError('Bootstrap block changed')
                values = np.load(dest)['draws']
                if values.shape != (n, 3):
                    raise ValueError('Bootstrap block shape mismatch')
                arrays[p['id']][start:start+n] = values; completed += 1
            elif not keys:
                completed += 1
            else:
                jobs.append((p, start, counts, keys))
    dump(out / 'common_population_PRIVATE.json', common)
    total = completed + len(jobs); start_time = time.monotonic(); last = 0.
    print('Bootstrap', workers, 'single-thread processes;', total, 'blocks;', completed, 'cached/empty', flush=True)
    if jobs:
        with cf.ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context('spawn')) as pool:
            active = {pool.submit(block, t) for t in jobs}
            while active:
                finished, active = cf.wait(active, timeout=20, return_when=cf.FIRST_COMPLETED)
                for future in finished:
                    pid, start, values = future.result(); arrays[pid][start:start+len(values)] = values
                    dest = directory / f'panel_{pid:02d}_{start:05d}.npz'
                    tmp = dest.with_suffix('.tmp')
                    with tmp.open('wb') as f:
                        np.savez_compressed(f, draws=values)
                    tmp.replace(dest); dump(dest.with_suffix('.json'), dict(sha256=sha(dest))); completed += 1
                if time.monotonic() - last >= 20 or not active:
                    print('Bootstrap blocks', completed, '/', total, 'elapsed', round(time.monotonic()-start_time, 1), 's', flush=True)
                    dump(out / 'bootstrap_progress.json', dict(completed=completed, total=total,
                                                               seconds=time.monotonic()-start_time)); last = time.monotonic()
    output = []
    for p in pp:
        cs = [APContrast(c['records'], c['left'], c['right'], 'subject_group') for c in p['comparisons']]
        cis = intervals(cs, arrays[p['id']])
        for c, ci in zip(p['comparisons'], cis):
            rr = c['records']; y = [r['reference_failure'] for r in rr]
            meta = {k: v for k, v in p.items() if k not in ['id', 'comparisons']}
            output.append(dict(meta, left=c['left'], right=c['right'], eligible_before_common=c['eligible_before_common'],
                               **analysis_census(rr), failures_by_disease=dict(Counter(r['reference_diagnosis'] for r in rr if r['reference_failure'])),
                               left_ap=average_precision(y, [-r['scores'][c['left']] for r in rr]),
                               right_ap=average_precision(y, [-r['scores'][c['right']] for r in rr]),
                               failure_prevalence=sum(y)/len(y) if y else None,
                               outcome_classes=len(set(y)),
                               cluster_unit='subject_group', seed=seed, **ci))
    table(out / 'three_comparisons_ALL.csv', output)
    table(out / 'three_comparisons_PRIMARY.csv', [r for r in output if r['primary']])
    dump(out / 'statistics.json', dict(results=output, family_size=3, repetitions=repetitions, seed=seed,
                                      note='AP contrasts are within generator. Absolute AP across generators is not a paired performance gain. Undefined resamples are explicitly counted.'))
    return output
