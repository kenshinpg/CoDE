"""Shared-storage partition and merge contracts; all records here are synthetic."""
from pathlib import Path
import sys
import tempfile
import unittest
import contextlib
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from trace_generator_rq2.common import dump,emit,read,sha,digest,index
from trace_generator_rq2.sharded import plan_tasks,collect,snapshot,Heartbeat,node
from run_trace_generator_rq2 import matching_tasks


def fixture(out):
    patients={f'p{i}':dict(facts=[dict(fact_id=f'f{j}',feature_id=f'e{j}',state='present',display_text='pain present')
                                  for j in range(9+i)]) for i in range(8)}
    tasks=[dict(task_id=f't{i:02}',patient_id=f'p{i%8}',rollout_index=i//8) for i in range(32)]
    plan=plan_tasks(tasks,patients);dump(out/'shard_plan.json',plan)
    ready=dict(identity='synthetic');ti=index(tasks,'task_id')
    for s in plan['shards']:
        folder=out/'nodes'/f"node_{s['node_index']}"
        ts=[ti[t] for t in s['task_ids']];emit(folder/'tasks_PRIVATE.jsonl',ts)
        gens=[dict(task_id=t['task_id'],patient_id=t['patient_id'],status='complete',parsed_explanation='Pain.',
                   parsed_diagnoses=['appendicitis','pancreatitis','cholecystitis'],task_input_digest=digest(t)) for t in ts]
        mt=matching_tasks(ts,index(gens,'task_id'),patients)
        matches=[dict(id=t['id'],task_id=t['task_id'],status='FAILED',parsed=None,task_input_digest=digest(t)) for t in mt]
        emit(folder/'generation_records_PRIVATE.jsonl',gens);emit(folder/'matching_records_PRIVATE.jsonl',matches)
        dump(folder/'service_binding.json',dict(model='same',metadata={},weights=[],config_dtype='bf16',
             quantization_config=None,service=dict(max_model_len=65536)))
        done=dict(node_index=s['node_index'],ready_digest=digest(ready),generations=len(gens),matching_groups=len(matches),
                  files={n:sha(folder/n) for n in ['tasks_PRIVATE.jsonl','generation_records_PRIVATE.jsonl',
                         'matching_records_PRIVATE.jsonl','service_binding.json']})
        dump(folder/'DONE.json',done)
    return ready,tasks,patients


class ShardContracts(unittest.TestCase):
    def test_independent_node_runs_both_stages_and_completed_resume_makes_no_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);ready,tasks,patients=fixture(out);folder=out/'nodes/node_3'
            (folder/'DONE.json').unlink();view=out/'view'
            emit(view/'patients_model_input_PRIVATE.jsonl',[dict(p,patient_id=pid) for pid,p in patients.items()])
            config=out/'config.json';dump(config,dict(paths=dict(view=str(view))))
            a=SimpleNamespace(out=out,node_index=3,config=config,endpoint='unused',model='same',model_dir=out)
            calls=[]
            class Fake:
                def generate(self,t):
                    calls.append(('gen',t['task_id']))
                    return dict(task_id=t['task_id'],patient_id=t['patient_id'],status='complete',
                                parsed_explanation='Pain.',parsed_diagnoses=['appendicitis','pancreatitis','cholecystitis'])
                def match(self,t):
                    calls.append(('match',t['id']))
                    return dict(id=t['id'],task_id=t['task_id'],status='COMPLETE',parsed=dict(decisions=[]))
            with patch('trace_generator_rq2.sharded.exclusive',side_effect=lambda p:contextlib.nullcontext()), \
                 patch('trace_generator_rq2.sharded.wait_ready',return_value=ready), \
                 patch('trace_generator_rq2.service.service',return_value=65536), \
                 patch('trace_generator_rq2.service.Client',return_value=Fake()):
                node(a);initial=len(calls);node(a)
            self.assertEqual(initial,12);self.assertEqual(len(calls),initial)
            self.assertEqual(read(folder/'DONE.json')['generations'],4)
            self.assertEqual(read(folder/'DONE.json')['matching_groups'],8)

    def test_balancing_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);ready,tasks,patients=fixture(out)
            a=plan_tasks(tasks,patients);b=plan_tasks(tasks,patients)
            self.assertEqual(a,b)
            all_ids=[i for s in a['shards'] for i in s['task_ids']]
            self.assertEqual(len(all_ids),32);self.assertEqual(len(set(all_ids)),32)
            costs=[s['planned_matching_groups'] for s in a['shards']]
            self.assertLessEqual(max(costs)-min(costs),2)

    def test_complete_merge_includes_terminal_failures_without_dropping_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);ready,tasks,patients=fixture(out)
            gens,matched=collect(SimpleNamespace(out=out),ready,tasks,patients)
            self.assertEqual(len(gens),32);self.assertEqual(len(matched),64)
            self.assertTrue(all(m['status']=='FAILED' for m in matched))
            self.assertEqual(read(out/'shard_acceptance.json')['duplicate_or_missing_tasks'],0)

    def test_merge_rejects_missing_node_and_tampered_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);ready,tasks,patients=fixture(out)
            path=out/'nodes/node_7/DONE.json';save=read(path);path.unlink()
            with self.assertRaises(FileNotFoundError):collect(SimpleNamespace(out=out),ready,tasks,patients)
            dump(path,save)
            with (out/'nodes/node_7/generation_records_PRIVATE.jsonl').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'output changed'):collect(SimpleNamespace(out=out),ready,tasks,patients)

    def test_merge_rejects_model_identity_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);ready,tasks,patients=fixture(out)
            folder=out/'nodes/node_7';p=folder/'service_binding.json';v=read(p);v['config_dtype']='fp32';dump(p,v)
            d=read(folder/'DONE.json');d['files']['service_binding.json']=sha(p);dump(folder/'DONE.json',d)
            with self.assertRaisesRegex(ValueError,'metadata/context mismatch'):collect(SimpleNamespace(out=out),ready,tasks,patients)

    def test_heartbeat_and_status_use_shared_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with Heartbeat(out/'nodes/node_2','generation') as hb:
                hb.set(planned_generations=3)
                self.assertEqual(snapshot(out)[2]['status'],'running')
            self.assertEqual(snapshot(out)[2]['status'],'missing_done_marker')
            with self.assertRaises(ValueError):
                with Heartbeat(out/'nodes/node_3','matching'):raise ValueError('synthetic')
            self.assertEqual(snapshot(out)[3]['status'],'failed')

    def test_score_resume_recomputes_uncommitted_patient_cache(self):
        import concurrent.futures as cf
        from trace_generator_rq2.scoring import parallel_score
        class Executor:
            def __init__(self,**kw):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def submit(self,fn,task):
                f=cf.Future();f.set_result(('p1',[dict(value='recomputed')]));return f
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            dump(folder/'patient_p1.json',dict(patient_id='p1',rows=[dict(value='uncommitted')]))
            with patch('trace_generator_rq2.scoring.cf.ProcessPoolExecutor',Executor):
                result=parallel_score([(dict(patient_id='p1'),{},[])],{},folder,1,dict(frozen=True),'synthetic')
            self.assertEqual(result,[dict(value='recomputed')]);self.assertTrue((folder/'patient_p1.lock').exists())

if __name__=='__main__':unittest.main()
