"""M1/M2 regression contracts from LANTERN's production consistency review."""
from pathlib import Path
import sys,json,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from trace_generator_rq2.common import dump,read,digest
from trace_generator_rq2.service import Client,InfrastructureError,queue
from trace_generator_rq2.execution_audit import execution_summary
from run_trace_generator_rq2 import make_ledger


class ReviewContracts(unittest.TestCase):
    def test_nine_facts_partial_group_preserves_observed_exact_but_never_scores_it(self):
        facts=[dict(fact_id='f'+str(i)) for i in range(9)]
        tasks=[dict(task_id='t',patient_id='p')]
        gens=dict(t=dict(status='complete',parsed_explanation='Pain.',parsed_diagnoses=['appendicitis']))
        split=dict(p=dict(partition='test',subject_group='s',reference_class='appendicitis'))
        decisions=[dict(fact_id='f'+str(i),status='exact' if i==0 else 'not_referenced',
                        quotes=[dict(text='Pain',start=0,end=4)] if i==0 else []) for i in range(8)]
        first=dict(id='t_g0',task_id='t',status='COMPLETE',parsed=dict(decisions=decisions))
        failed=dict(id='t_g1',task_id='t',status='FAILED',parsed=None)
        for results in [[first,failed],[first]]:
            ledger,matching=make_ledger(tasks,gens,results,dict(p=dict(facts=facts)),split)
            e,m=ledger[0],matching[0]
            self.assertEqual(e['visible_fact_scan_status'],'PARTIAL');self.assertEqual(e['generation_status'],'complete')
            self.assertFalse(e['scan_complete']);self.assertFalse(e['reviewed_available']);self.assertFalse(e['analysis_mapping_available'])
            self.assertIsNone(e['cited_fact_ids']);self.assertEqual(m['observed_exact_fact_ids'],['f0'])
            self.assertEqual(len(m['decisions']),8);self.assertFalse(m['observed_exact_is_complete'])
            self.assertEqual(m['failed_groups']+m['missing_groups'],1)

    def test_generation_infrastructure_event_and_resume_keep_response_budget(self):
        t=dict(task_id='t',patient_id='p',messages=[])
        def raw(x):return dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(x)))],usage=dict(prompt_tokens=100))
        valid=raw(dict(explanation='Pain.',ranked_diagnoses=['appendicitis','pancreatitis','cholecystitis']))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);client=Client('unused','model',65536,out)
            with patch.object(client,'tokens',return_value=100),patch.object(client,'request',side_effect=[raw({}),InfrastructureError('HTTP 503',503,'HTTPError')]):
                with self.assertRaises(InfrastructureError):queue([t],'task_id',client.generate,out/'generation_results_PRIVATE',2,'generation')
            before=execution_summary(out)
            self.assertEqual(before['generation']['returned_response_attempts'],1)
            self.assertEqual(before['generation']['infrastructure_unfinished_tasks'],1)
            event=read(next((out/'infrastructure_events_PRIVATE').glob('*.json')))
            self.assertEqual(event['attempt'],2);self.assertEqual(event['http_status'],503)
            self.assertEqual(event['seed'],int(digest([261013,'t','initial',1])[:8],16)%2147483647)
            self.assertEqual(event['stage'],'generation');self.assertEqual(event['operation'],'chat')
            self.assertEqual(len(list((out/'queue_errors_PRIVATE').glob('*.json'))),1)
            with patch.object(client,'tokens',return_value=100),patch.object(client,'request',return_value=valid) as call:
                result=queue([t],'task_id',client.generate,out/'generation_results_PRIVATE',2,'generation')
                self.assertEqual(call.call_count,1)
            after=execution_summary(out)
            self.assertEqual(result[0]['attempts'],2)
            self.assertEqual(after['generation']['returned_response_attempts'],2)
            self.assertEqual(after['generation']['infrastructure_events'],1)
            self.assertEqual(after['generation']['infrastructure_unfinished_tasks'],0)

    def test_parallel_queue_persists_every_failed_future_task_id(self):
        tasks=[dict(id=str(i),task_id='t'+str(i)) for i in range(2)]
        def fail(t):raise InfrastructureError('offline')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with self.assertRaises(InfrastructureError):
                queue(tasks,'id',fail,out/'matching_results_PRIVATE',2,'matching',verify_first=False)
            errors=[read(p) for p in (out/'queue_errors_PRIVATE').glob('*.json')]
            self.assertEqual({e['task_key'] for e in errors},{'0','1'})

    def test_interrupted_matching_replay_does_not_erase_later_saved_attempts(self):
        t=dict(id='g',task_id='t',explanation='Pain.',facts=[dict(fact_id='f',feature_id='f',state='present',display_text='pain')])
        raw=dict(choices=[dict(finish_reason='stop',message=dict(content='{}'))])
        history=[dict(mode=mode,attempt=i,prompt_tokens=100,raw_response=raw) for mode,i in [('free',1),('free',2),('segment',1)]]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);journal=out/'matching_attempts_PRIVATE/g.json';dump(journal,history)
            client=Client('unused','model',65536,out)
            def interrupt_after_write(path,value):
                dump(path,value)
                if path==journal:raise KeyboardInterrupt('synthetic stop')
            with patch('trace_generator_rq2.service.dump',side_effect=interrupt_after_write):
                with self.assertRaises(KeyboardInterrupt):client.match(t)
            saved=read(journal)
            self.assertEqual([(x['mode'],x['attempt']) for x in saved],[('free',1),('free',2),('segment',1)])

if __name__=='__main__':unittest.main()
