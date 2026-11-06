"""Unique changed contracts; synthetic data only, no API calls or training."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from trace_generator_rq2.common import make_tasks, digest, read, LABELS
from trace_generator_rq2 import base, segment, original_generation as gen, scoring
from trace_generator_rq2.service import Client, queue, InfrastructureError
from trace_generator_rq2.multi import Pool, endpoints
from trace_generator_rq2.statistics import panels, block, summarize
from run_trace_generator_rq2 import matching_tasks, make_ledger
from code_eval.proxy_metrics import average_precision


def patient(pid='p1'):
    return dict(patient_id=pid, view_id='frozen', context={}, facts=[
        dict(fact_id='f1', feature_id='a', state='present', display_text='pain present', group_id='a'),
        dict(fact_id='f2', feature_id='b', state='absent', display_text='fever absent', group_id='b')])


class GeneratorContracts(unittest.TestCase):
    def test_multinode_capacity_warmup_and_no_duplicate_dispatch(self):
        guard=threading.Lock(); current=[0,0]; peak=[0,0]; calls=[]
        class Fake:
            def __init__(self,i):self.i=i
            def generate(self,t):
                with guard:
                    calls.append(t['id']);current[self.i]+=1;peak[self.i]=max(peak[self.i],current[self.i])
                time.sleep(.005)
                with guard:current[self.i]-=1
                return dict(id=t['id'],status='complete')
        nodes=[dict(name='n0'),dict(name='n1')]
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'generation'; tasks=[dict(id=str(i)) for i in range(30)]
            pool=Pool(nodes,[Fake(0),Fake(1)],capacity=2,out=Path(temp))
            pool.begin('generate');pool.warmup(tasks,'id',folder)
            result=queue(tasks,'id',pool.execute,folder,8,'generation',pool.progress,verify_first=False)
            self.assertEqual(len(calls),30);self.assertEqual(len(set(calls)),30)
            self.assertTrue(all(0<p<=2 for p in peak))
            self.assertEqual({r['worker_node'] for r in result},{'n0','n1'})
            pool.begin('generate');pool.warmup(tasks,'id',folder)
            again=queue(tasks,'id',pool.execute,folder,8,'generation',pool.progress,verify_first=False)
            self.assertEqual(result,again);self.assertEqual(len(calls),30)

    def test_multinode_transport_failover_and_deadline_estimate(self):
        class Offline:
            def match(self,t):raise InfrastructureError('offline')
        class Healthy:
            def match(self,t):return dict(id=t['id'],status='COMPLETE')
        with tempfile.TemporaryDirectory() as temp:
            pool=Pool([dict(name='bad'),dict(name='good')],[Offline(),Healthy()],capacity=2,out=Path(temp),
                      budget_start=time.monotonic()-3600,target_minutes=120,local_reserve_minutes=30)
            pool.begin('match');r=pool.execute(dict(id='t1'))
            self.assertEqual(r['worker_node'],'good');self.assertEqual(r['transport_failover_nodes'],['bad'])
            self.assertEqual(pool.active,[0,0]);self.assertEqual(pool.live,[False,True])
            report=pool.progress(1000,600,1000)
            self.assertFalse(report['deadline_estimate']['target_on_track'])
            self.assertGreaterEqual(report['deadline_estimate']['estimated_nodes_needed_at_observed_rate'],2)
            self.assertEqual(read(Path(temp)/'node_failover.json')[0]['node'],'bad')

    def test_multinode_configuration_rejects_placeholders_and_duplicate_endpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'nodes.json'
            path.write_text(json.dumps(dict(nodes=[dict(name='n1',endpoint='http://NODE_01:8802/v1/chat/completions')])))
            with self.assertRaisesRegex(ValueError,'actual node'):endpoints(path)
            path.write_text(json.dumps(dict(nodes=[dict(name=n,endpoint='http://127.0.0.1:8802/v1/chat/completions') for n in ['a','b']])))
            with self.assertRaisesRegex(ValueError,'Duplicate'):endpoints(path)

    def test_fixed_four_tasks_reuses_prompt_without_answer_and_identity(self):
        pts = {p: patient(p) for p in ['p1','p2']}
        split = {'p1':dict(partition='test',reference_class='appendicitis',subject_group='s1'),
                 'p2':dict(partition='train',reference_class='pancreatitis',subject_group='s2')}
        tasks = make_tasks(pts, split, 'Qwen3.8-27B')
        self.assertEqual(len(tasks),4)
        self.assertEqual([t['rollout_index'] for t in tasks],list(range(4)))
        for t in tasks:
            self.assertEqual(t['messages'],gen.messages(pts['p1']))
            self.assertEqual(set(json.loads(t['messages'][1]['content'])), {'allowed_diagnoses','patient_facts'})
            self.assertEqual(t['baseline_task_id'],'TRACE_'+gen.digest(['frozen','p1',t['rollout_index']])[:24])
            self.assertNotEqual(t['task_id'],t['baseline_task_id'])

    def test_match_fallback_preserves_verbatim_negative_evidence(self):
        t = dict(id='t_g0000',task_id='t',patient_id='p1',explanation='Pain is present. No fever.',facts=patient()['facts'])
        invalid = {'decisions':[dict(index=i,status='exact',quotes=[dict(text='invented',occurrence=0)],overflow=False) for i in range(2)]}
        segments = segment.segments(t['explanation'])
        valid = {'decisions':[dict(index=0,status='exact',segment_ids=[segments[0]['id']],overflow=False),
                              dict(index=1,status='exact',segment_ids=[segments[1]['id']],overflow=False)]}
        raw = lambda x: {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(x)}}]}
        with tempfile.TemporaryDirectory() as temp:
            client = Client('unused','Qwen3.8-27B',65536,Path(temp)); requests=[]
            def request(payload):
                requests.append(payload)
                return raw(invalid if len(requests)<3 else valid)
            with patch.object(client,'tokens',return_value=100), patch.object(client,'request',side_effect=request):
                r = client.match(t)
            self.assertEqual(r['status'],'COMPLETE'); self.assertEqual(r['attempts'],3)
            self.assertEqual(r['annotation_protocol'],'full_v1_segment')
            self.assertEqual(requests[0]['messages'][0]['content'],base.PROMPT)
            self.assertEqual([x['temperature'] for x in requests],[0,0,0])
            self.assertEqual(r['parsed']['decisions'][1]['fact_id'],'f2')
            q=r['parsed']['decisions'][1]['quotes'][0]
            self.assertEqual(q['text'],t['explanation'][q['start']:q['end']])

    def test_unavailable_is_not_empty_and_ambiguity_is_excluded(self):
        pts={'p1':patient()};split={'p1':dict(partition='test',reference_class='appendicitis',subject_group='s1')}
        tasks=make_tasks(pts,split,'Qwen3.8-27B')
        gens={t['task_id']:dict(task_id=t['task_id'],status='complete',parsed_explanation='No fever.',parsed_diagnoses=LABELS[:3]) for t in tasks}
        gens[tasks[0]['task_id']].update(status='failed',parsed_explanation=None,parsed_diagnoses=None)
        mt=matching_tasks(tasks,gens,pts);self.assertEqual(len(mt),3)
        results=[dict(id=t['id'],task_id=t['task_id'],status='FAILED',parsed=None) for t in mt]
        decisions=[dict(fact_id='f1',status='ambiguous',quotes=[]),dict(fact_id='f2',status='not_referenced',quotes=[])]
        results[0].update(status='COMPLETE',parsed=dict(decisions=decisions))
        ledger,_=make_ledger(tasks,gens,results,pts,split)
        self.assertIsNone(ledger[0]['cited_fact_ids']);self.assertFalse(ledger[0]['reviewed_available'])
        self.assertEqual(ledger[1]['repair_status'],'PARTIAL');self.assertTrue(ledger[1]['reviewed_available'])
        self.assertIsNone(ledger[2]['cited_fact_ids']);self.assertFalse(ledger[2]['reviewed_available'])

    def test_resume_keeps_completed_tasks_and_rejects_input_drift(self):
        tasks=[dict(id=str(i),payload=i) for i in range(3)];called=[]
        def work(t):called.append(t['id']);return dict(id=t['id'],status='COMPLETE')
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'queue'
            queue(tasks,'id',work,folder,2,'test');queue(tasks,'id',work,folder,2,'test')
            self.assertEqual(len(called),3)
            changed=[dict(t,payload=99) for t in tasks]
            with self.assertRaisesRegex(ValueError,'input changed'):queue(changed,'id',work,folder,2,'test')

    def test_generation_transport_resume_keeps_completed_attempt_budget(self):
        t=dict(task_id='new1',patient_id='p1',messages=gen.messages(patient()))
        raw=lambda content:dict(choices=[dict(finish_reason='stop',message=dict(content=content))],usage=dict(prompt_tokens=100))
        valid=raw(json.dumps(dict(explanation='Pain is present.',ranked_diagnoses=LABELS[:3])))
        with tempfile.TemporaryDirectory() as temp:
            client=Client('unused','Qwen3.8-27B',8192,Path(temp))
            with patch.object(client,'tokens',return_value=100),patch.object(client,'request',side_effect=[raw('{}'),InfrastructureError('offline')]):
                with self.assertRaises(InfrastructureError):client.generate(t)
            with patch.object(client,'tokens',return_value=100),patch.object(client,'request',return_value=valid) as request:
                result=client.generate(t)
            self.assertEqual(request.call_count,1)
            self.assertEqual(result['attempts'],2)
            self.assertEqual(result['status'],'complete')

    def test_cpu_score_uses_proposal_for_scores_and_reference_for_endpoint(self):
        class Enc:
            columns=['a','b']
            def groups(self,tokens):return {'a':['a_@_present'],'b':['b_@_absent']}
            def token(self,t):return (0 if t.startswith('a_') else 1,1.)
            def encode(self,t,c):return np.array([1.,1.])
        class Bank:
            encoding=Enc();tree=object()
            def predict(self,sets,context,sources):
                result={n:[] for n in sources}
                for ts in sets:
                    prob=np.array([.8,.1,.05,.05]) if len(ts)==2 else np.array([.1,.7,.1,.1])
                    for n in result:result[n].append(prob)
                return result
        p=patient();r=dict(patient_id='p1',tokens=['a_@_present','b_@_absent'],label='appendicitis')
        e=dict(explanation_id='e',reviewed_available=True,cited_fact_ids=['f1'],diagnosis='cholecystitis',
               target_supported=True,diagnosis_correct=False,subject_group='s1',repair_status='COMPLETE')
        shap=np.zeros((1,4,3));shap[0,1,:2]=[2.,-1.]
        with patch.object(scoring,'_STATE',Bank()),patch('code_eval.tree_subset.checked_contributions',return_value=shap):
            _,rs=scoring.score_patient((p,r,[e]))
        self.assertTrue(rs[0]['reference_failure']);self.assertFalse(rs[0]['proposal_failure'])
        self.assertEqual(rs[0]['scores']['lightgbm_absolute'],.7)
        self.assertEqual(rs[0]['scores']['tree_positive_shap'],1.)
        self.assertEqual(rs[0]['scores']['group_coverage'],.5)
        self.assertIsNone(rs[0]['scores']['cohd_coverage'])

    def test_paired_bootstrap_matches_duplicate_subjects_and_reports_no_failure(self):
        rs=[]
        for i,(subject,y,s) in enumerate([('s1',True,.1),('s1',False,.1),('s2',False,.8)]):
            rs.append(dict(explanation_id=str(i),patient_id=subject,subject_group=subject,reference_failure=y,
                           scores=dict(cohd_coverage=s,group_coverage=.5,tree_positive_shap=s,lightgbm_absolute=1-s,fact_coverage=.5),
                           generator='Qwen3.8-27B',probe='encoder',mapping_available=True,scan_status='COMPLETE',
                           target_supported=True,both_full_correct=True,probe_available=True,diagnosis_correct=True))
        pp,_=panels(rs);p=next(p for p in pp if p['primary'])
        pid,start,values=block((p,0,np.array([[2,1],[0,3]]),['s1','s2']))
        duplicate=rs[:2]*2+rs[2:]
        expected=average_precision([r['reference_failure'] for r in duplicate],[-r['scores']['cohd_coverage'] for r in duplicate])-average_precision([r['reference_failure'] for r in duplicate],[-r['scores']['group_coverage'] for r in duplicate])
        self.assertAlmostEqual(values[0,0],expected)
        self.assertTrue(np.isnan(values[1]).all())
        rs[0]['scan_status']='PARTIAL'
        pp,_=panels(rs);p=next(p for p in pp if p['primary'])
        self.assertEqual(len(p['comparisons'][0]['records']),2)

    def test_parallel_summary_exports_all_strata_and_resumes_blocks(self):
        rs=[]
        for generator in ['Qwen3-4B','Qwen3.8-27B']:
            for probe in ['encoder','lightgbm']:
                for i in range(6):
                    rs.append(dict(explanation_id=str(i),patient_id='p'+str(i),subject_group='s'+str(i),
                                   generator=generator,probe=probe,reference_failure=i%3==0,
                                   reference_diagnosis=LABELS[i%4],mapping_available=True,scan_status='COMPLETE',
                                   both_full_correct=True,probe_available=True,target_supported=True,diagnosis_correct=i%2==0,
                                   scores=dict(cohd_coverage=i/6,group_coverage=.5,tree_positive_shap=(5-i)/6,
                                               lightgbm_absolute=.4,fact_coverage=.5)))
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            first=summarize(rs,out,2,repetitions=12,seed=261016)
            again=summarize(rs,out,2,repetitions=12,seed=261016)
            self.assertEqual(first,again)
            self.assertEqual(len(first),36)
            self.assertEqual(sum(r['primary'] for r in first),3)
            self.assertTrue((out/'three_comparisons_PRIMARY.csv').is_file())

if __name__=='__main__':unittest.main()
