"""Eight independent local API workers sharing immutable task lists and a finalizer."""
import argparse
import collections
import contextlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from .common import ROOT, read, rows, dump, emit, locked, digest, sha, index, local_path, LABELS

NODES = 8
DEFAULT_OUT = ROOT / 'artifacts/trace_qwen27b_rq2_shared_v2'


@contextlib.contextmanager
def exclusive(path, wait=False):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError('Another process owns this shared lock: ' + str(path))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def plan_tasks(tasks, patients):
    costs = {t['task_id']: (len(patients[t['patient_id']]['facts'])+7)//8 for t in tasks}
    shards = [dict(node_index=i, task_ids=[], planned_matching_groups=0) for i in range(NODES)]
    for t in sorted(tasks, key=lambda t: (-costs[t['task_id']], t['task_id'])):
        s = min(shards, key=lambda x: (x['planned_matching_groups'], len(x['task_ids']), x['node_index']))
        s['task_ids'].append(t['task_id']); s['planned_matching_groups'] += costs[t['task_id']]
    for s in shards:
        s['task_ids'].sort(); s['planned_generations'] = len(s['task_ids'])
    return dict(nodes=NODES, tasks_digest=digest(tasks), tasks=len(tasks),
                allocation='greedy balance by fixed fact-group count; no outcomes used', shards=shards)


def legacy_args(a):
    return SimpleNamespace(config=a.config, out=a.out, model=a.model, model_dir=a.model_dir,
        endpoint='shared_storage_8_fixed_shards', endpoints_file=None, workers=128,
        cpu_workers=a.cpu_workers, summary_workers=a.summary_workers, bootstrap=5000, seed=261016,
        target_minutes=120, local_reserve_minutes=30, preflight_only=False)


class Heartbeat:
    def __init__(self, folder, stage):
        self.folder = folder; self.lock = threading.Lock(); self.stop = threading.Event()
        self.state = dict(status='running', stage=stage, hostname=socket.gethostname(), pid=os.getpid(), started_at=time.time())
        self.thread = threading.Thread(target=self.loop, daemon=True)
    def write(self):
        with self.lock:
            dump(self.folder/'status.json', dict(self.state, updated_at=time.time()))
    def set(self, **fields):
        with self.lock: self.state.update(fields)
        self.write()
    def loop(self):
        while not self.stop.wait(20): self.write()
    def __enter__(self):
        self.write(); self.thread.start(); return self
    def __exit__(self, typ, value, tb):
        self.stop.set(); self.thread.join()
        if typ:
            self.set(status='interrupted' if issubclass(typ, (KeyboardInterrupt, SystemExit)) else 'failed',
                     error_type=typ.__name__)
        else: self.set(status='complete')
        if self.folder.parent.name=='nodes' and self.folder.name.startswith('node_'):
            from .execution_audit import execution_summary
            execution_summary(self.folder)


def prepare(a):
    import run_trace_generator_rq2 as old
    with exclusive(a.out/'prepare.lock'):
        with Heartbeat(a.out/'preparation', 'input_checks') as hb:
            if (a.out/'READY.json').exists():
                verify_ready(a); print('Preparation already complete', flush=True); return
            subprocess.run([sys.executable,'-u','-m','unittest','discover','-s','tests',
                            '-p','test_trace_generator_rq2*.py','-v'], cwd=ROOT, check=True)
            la = legacy_args(a)
            _, patients, _, records, tasks, ledger, baseline, spec = old.preflight(la)
            plan = plan_tasks(tasks, patients)
            locked(a.out/'shard_plan.json', plan)
            ti = index(tasks, 'task_id'); task_files = {}
            for shard in plan['shards']:
                p = a.out/'nodes'/f"node_{shard['node_index']}"/'tasks_PRIVATE.jsonl'
                emit(p, [ti[t] for t in shard['task_ids']]); task_files[p.relative_to(a.out).as_posix()] = sha(p)
            hb.set(stage='baseline_replay')
            old.replay(la, patients, records, ledger, baseline, spec)
            ready = dict(input_binding_sha256=sha(a.out/'input_binding.json'),
                         entry_sha256=sha(ROOT/'scripts/run_trace_generator_rq2_shared.py'),
                         plan_sha256=sha(a.out/'shard_plan.json'),
                         generation_tasks_sha256=sha(a.out/'generation_tasks_PRIVATE.jsonl'),
                         replay_sha256=sha(a.out/'baseline_replay_audit.json'),
                         task_files=task_files, nodes=NODES)
            locked(a.out/'READY.json', ready)
            print('READY: 8 fixed shards; 1436 generations; 12784 maximum matching groups', flush=True)


def verify_ready(a):
    ready = read(a.out/'READY.json'); binding = read(a.out/'input_binding.json')
    if sha(ROOT/'scripts/run_trace_generator_rq2_shared.py') != ready['entry_sha256']:
        raise ValueError('Shared-storage entry point changed')
    for filename, key in [('input_binding.json','input_binding_sha256'),('shard_plan.json','plan_sha256'),
                          ('generation_tasks_PRIVATE.jsonl','generation_tasks_sha256'),
                          ('baseline_replay_audit.json','replay_sha256')]:
        if sha(a.out/filename) != ready[key]: raise ValueError('Prepared file changed: '+filename)
    if binding['config'] != sha(a.config) or binding['model'] != a.model:
        raise ValueError('Prepared model/config changed')
    for path, expected in binding['code'].items():
        if sha(ROOT/path) != expected: raise ValueError('Code changed after preparation: '+path)
    for path, expected in binding['files'].items():
        if sha(path) != expected: raise ValueError('Frozen input/checkpoint changed: '+path)
    for path, expected in ready['task_files'].items():
        if sha(a.out/path) != expected: raise ValueError('Shard task list changed: '+path)
    if read(a.out/'baseline_replay_audit.json')['status'] != 'passed': raise ValueError('Baseline replay not passed')
    return ready


def wait_ready(a):
    started = time.monotonic()
    while not (a.out/'READY.json').exists():
        if time.monotonic()-started > a.wait_timeout: raise TimeoutError('Waiting for node0 preparation timed out')
        p = a.out/'preparation/status.json'
        state = read(p) if p.exists() else {}
        if state.get('status') in ['failed','interrupted']:
            raise RuntimeError('Preparation stopped; inspect shared logs/node_0.log and restart')
        print('WAIT_READY node0 performs input checks and baseline replay;',state.get('stage','not started'),flush=True)
        time.sleep(20)
    return verify_ready(a)


def validate_cache(tasks, key, results):
    expected = index(tasks, key); actual = index(results, key)
    if set(expected) != set(actual): raise ValueError('Task/result census mismatch')
    for tid, r in actual.items():
        if r.get('task_input_digest') != digest(expected[tid]): raise ValueError('Cached task input changed')
        valid = ['complete','failed'] if key == 'task_id' else ['COMPLETE','FAILED']
        if r['status'] not in valid: raise ValueError('Unexpected terminal status')
    return [actual[t[key]] for t in tasks]


def node(a):
    import run_trace_generator_rq2 as old
    from .service import Client, service, queue
    if a.node_index not in range(NODES): raise ValueError('node-index must be 0..7')
    folder = a.out/'nodes'/f'node_{a.node_index}'
    with exclusive(folder/'writer.lock'):
        if a.node_index == 0: prepare(a)
        with Heartbeat(folder, 'waiting_for_ready') as hb:
            ready = wait_ready(a)
            if (folder/'DONE.json').exists():
                verify_node_files(folder, ready)
                print('Node already complete; no API calls', flush=True); return
            tasks = rows(folder/'tasks_PRIVATE.jsonl')
            config = read(a.config); view = local_path(config['paths']['view'])
            patients = index(rows(view/'patients_model_input_PRIVATE.jsonl'), 'patient_id')
            hb.set(stage='service_check')
            limit = service(a.endpoint, a.model, a.model_dir, folder)
            client = Client(a.endpoint, a.model, limit, folder)
            hb.set(stage='generation', planned_generations=len(tasks))
            gens = queue(tasks,'task_id',client.generate,folder/'generation_results_PRIVATE',128,'generation')
            emit(folder/'generation_records_PRIVATE.jsonl',gens)
            mt = old.matching_tasks(tasks,index(gens,'task_id'),patients)
            locked(folder/'matching_input_lock.json',dict(tasks_digest=digest(mt),tasks=len(mt)))
            hb.set(stage='matching', planned_matching_groups=len(mt))
            matched = queue(mt,'id',client.match,folder/'matching_results_PRIVATE',128,'matching')
            emit(folder/'matching_records_PRIVATE.jsonl',matched)
            done = dict(node_index=a.node_index, ready_digest=digest(ready),
                files={n:sha(folder/n) for n in ['tasks_PRIVATE.jsonl','generation_records_PRIVATE.jsonl',
                       'matching_records_PRIVATE.jsonl','service_binding.json']},
                generations=len(gens), matching_groups=len(mt),
                generation_statuses=dict(collections.Counter(r['status'] for r in gens)),
                matching_statuses=dict(collections.Counter(r['status'] for r in matched)))
            locked(folder/'DONE.json',done)
            hb.set(stage='done')
            print('NODE_COMPLETE',a.node_index,done['generation_statuses'],done['matching_statuses'],flush=True)


def verify_node_files(folder, ready):
    done = read(folder/'DONE.json')
    if done['ready_digest'] != digest(ready): raise ValueError('Node belongs to a different prepared run')
    for name, expected in done['files'].items():
        if sha(folder/name) != expected: raise ValueError('Node output changed: '+str(folder/name))
    return done


def collect(a, ready, tasks, patients):
    import run_trace_generator_rq2 as old
    plan = read(a.out/'shard_plan.json'); all_gens = []; all_matches = []; identities=[]; audits=[]
    if plan != plan_tasks(tasks,patients): raise ValueError('Shard assignment drift')
    for shard in plan['shards']:
        folder=a.out/'nodes'/f"node_{shard['node_index']}"
        done=verify_node_files(folder,ready)
        if done['node_index'] != shard['node_index']: raise ValueError('Wrong node index')
        local_tasks=rows(folder/'tasks_PRIVATE.jsonl')
        if [t['task_id'] for t in local_tasks] != shard['task_ids']: raise ValueError('Shard membership changed')
        gens=validate_cache(local_tasks,'task_id',rows(folder/'generation_records_PRIVATE.jsonl'))
        mt=old.matching_tasks(local_tasks,index(gens,'task_id'),patients)
        matches=validate_cache(mt,'id',rows(folder/'matching_records_PRIVATE.jsonl'))
        if done['generations']!=len(gens) or done['matching_groups']!=len(matches): raise ValueError('DONE census changed')
        identity=read(folder/'service_binding.json')
        identities.append({k:identity[k] for k in ['model','metadata','weights','config_dtype','quantization_config']})
        identities[-1]['max_model_len']=identity['service']['max_model_len']
        all_gens.extend(gens);all_matches.extend(matches);audits.append(done)
    if any(i!=identities[0] for i in identities): raise ValueError('Node model metadata/context mismatch')
    all_gens=validate_cache(tasks,'task_id',all_gens)
    mt=old.matching_tasks(tasks,index(all_gens,'task_id'),patients)
    all_matches=validate_cache(mt,'id',all_matches)
    dump(a.out/'shard_acceptance.json',dict(status='passed',nodes=NODES,generations=len(all_gens),
        matching_groups=len(all_matches),duplicate_or_missing_tasks=0,node_audits=audits,
        model_identity='Matching local checkpoint metadata and service context; actual deployment weights confirmed by operator'))
    return all_gens, all_matches


def snapshot(out):
    result=[]
    for i in range(NODES):
        folder=out/'nodes'/f'node_{i}'; p=folder/'status.json'
        s=read(p) if p.exists() else dict(status='not_started',stage='not_started')
        done=(folder/'DONE.json').exists()
        row=dict(node=i,status='complete' if done else s['status'],stage=s.get('stage'),host=s.get('hostname'),
                 heartbeat_age_seconds=round(time.time()-s['updated_at']) if 'updated_at' in s else None)
        if not done and row['status']=='complete':row['status']='missing_done_marker'
        for label in ['generation','matching']:
            p=folder/(label+'_progress.json');r=read(p) if p.exists() else {}
            row[label]=dict(completed=r.get('completed',0),planned=r.get('planned'),
                statuses=r.get('statuses',{}),eta_hours=r.get('eta_hours'))
        result.append(row)
    return result


def wait_nodes(a):
    started=time.monotonic()
    while True:
        state=snapshot(a.out);print(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),nodes=state)),flush=True)
        if all(s['status']=='complete' for s in state): return
        bad=[s['node'] for s in state if s['status'] in ['failed','interrupted']]
        if bad: raise RuntimeError('Nodes stopped: '+str(bad)+'; inspect shared node logs, resume nodes, then restart finalizer')
        p=a.out/'preparation/status.json'
        if p.exists() and read(p)['status'] in ['failed','interrupted']:
            raise RuntimeError('Preparation stopped; inspect node0 log')
        if not a.wait or time.monotonic()-started>a.wait_timeout:
            raise RuntimeError('Not all eight shards complete')
        time.sleep(20)


def finalize(a):
    import run_trace_generator_rq2 as old
    with exclusive(a.out/'finalize.lock'):
        with Heartbeat(a.out/'finalization','waiting_for_nodes') as hb:
            state=dict(status='failed');started=time.time()
            try:
                wait_nodes(a)
                with contextlib.ExitStack() as locks:
                    for i in range(NODES):locks.enter_context(exclusive(a.out/'nodes'/f'node_{i}'/'writer.lock',wait=True))
                    hb.set(stage='acceptance')
                    ready=verify_ready(a);la=legacy_args(a)
                    la.model_dir=Path(read(a.out/'input_binding.json')['model_dir'])
                    _,patients,split,records,tasks,old_ledger,baseline,spec=old.preflight(la)
                    gens,matched=collect(a,ready,tasks,patients)
                    generations=index(gens,'task_id');ledger,matches=old.make_ledger(tasks,generations,matched,patients,split)
                    emit(a.out/'generation_records_PRIVATE.jsonl',gens)
                    emit(a.out/'explanation_ledger_PRIVATE.jsonl',ledger)
                    emit(a.out/'matching_candidates_PRIVATE.jsonl',matches)
                    dump(a.out/'generation_summary.json',dict(planned=len(tasks),admissions=len(records),
                        statuses=dict(collections.Counter(g['status'] for g in gens))))
                    dump(a.out/'mapping_summary.json',dict(planned_explanations=len(tasks),matching_groups=len(matched),
                        group_statuses=dict(collections.Counter(m['status'] for m in matched)),
                        raw_scan_statuses=dict(collections.Counter(e['visible_fact_scan_status'] for e in ledger)),
                        analysis_available=sum(e['analysis_mapping_available'] for e in ledger),
                        reasons=dict(collections.Counter(e['unavailability_reason'] or 'complete' for e in ledger))))
                    from .execution_audit import aggregate_execution,measurement_flow
                    infra=aggregate_execution(a.out)
                    dump(a.out/'measurement_flow.json',measurement_flow(ledger,matches,infra))
                    old.audit_sample(la,tasks,generations,patients,split,matches)
                    hb.set(stage='scoring')
                    from .scoring import parallel_score
                    scored=parallel_score(old.patient_jobs(patients,records,ledger),spec,a.out/'new_scores',a.cpu_workers,
                        dict(input_binding=sha(a.out/'input_binding.json'),ledger=digest(ledger)),'27B scoring')
                    if len(scored)!=2872 or len({(r['explanation_id'],r['probe']) for r in scored})!=2872:
                        raise ValueError('Score row census mismatch')
                    all_rows=[dict(r,generator='Qwen3-4B') for r in baseline]+scored
                    emit(a.out/'scores_PRIVATE.jsonl',all_rows);hb.set(stage='bootstrap')
                    from .statistics import summarize
                    contrasts=summarize(all_rows,a.out,a.summary_workers,5000,261016)
                    state=dict(status='complete',admissions=359,new_explanations=1436,matching_groups=len(matched),
                        contrast_rows=len(contrasts),primary_contrasts=sum(r['primary'] for r in contrasts),
                        nodes=NODES,cpu_workers=a.cpu_workers,summary_workers=a.summary_workers,
                        models_trained=0,independent_mapping_spotcheck='pending; 32 fixed records exported')
            except BaseException:
                state['error']=traceback.format_exc();raise
            finally:
                from .execution_audit import aggregate_execution
                aggregate_execution(a.out)
                state['elapsed_seconds']=time.time()-started;dump(a.out/'run_manifest.json',state)
                hb.set(status=state['status'],stage='complete' if state['status']=='complete' else 'failed')
                old.package(legacy_args(a))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['node','prepare','finalize','status','watch'])
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--config',type=Path,default=ROOT/'configs/trace_generator_rq2_20261027.json')
    p.add_argument('--node-index',type=int)
    p.add_argument('--endpoint',default=os.environ.get('QWEN_ENDPOINT','http://127.0.0.1:8802/v1/chat/completions'))
    p.add_argument('--model',default='Qwen3.8-27B')
    p.add_argument('--model-dir',type=Path,default=Path('models/Qwen3.8-27B'))
    p.add_argument('--cpu-workers',type=int,default=128)
    p.add_argument('--summary-workers',type=int,default=128)
    p.add_argument('--wait',action='store_true',help='Finalizer waits for all eight nodes')
    p.add_argument('--wait-timeout',type=float,default=21600)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    def stop(signum,frame):raise KeyboardInterrupt('Termination requested; preserve cache for resume')
    signal.signal(signal.SIGTERM,stop)
    if a.action=='prepare':prepare(a)
    elif a.action=='node':node(a)
    elif a.action=='finalize':finalize(a)
    else:
        while True:
            print(time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
            for r in snapshot(a.out):
                g=r['generation'];m=r['matching']
                age=str(r['heartbeat_age_seconds'])+'s' if r['heartbeat_age_seconds'] is not None else '-'
                stale=' STALE_CHECK_LOG' if r['status']=='running' and r['heartbeat_age_seconds']>120 else ''
                print(f"node{r['node']} {r['status']:12} {str(r['stage']):20} gen={g['completed']}/{g['planned'] if g['planned'] is not None else '-'} match={m['completed']}/{m['planned'] if m['planned'] is not None else '-'} heartbeat_age={age} host={r['host']}{stale}",flush=True)
            for name in ['preparation','finalization']:
                q=a.out/name/'status.json'
                if q.exists():print(name,read(q),flush=True)
            if a.action=='status':break
            time.sleep(20)

