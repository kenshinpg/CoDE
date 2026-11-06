import sys,tempfile,unittest,zipfile,contextlib,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from export_trace_lantern_handoff import export,sha,dump,emit,read
class Handoff(unittest.TestCase):
    def fixture(self,d):
        view=d/'view';view.mkdir();rules=ROOT/'configs/trace_lantern_handoff';vid='fixture'
        dictionary={'features':{'HP_1':{'label':'Pain','values':{'present':'present'},'group_id':'HP_1'}}}
        dump(view/'evidence_dictionary.json',dictionary);dump(view/'view_config.json',{'view_id':vid});dump(view/'view_report.json',{})
        split=[{'admission_id':'a','subject_group':'s','partition':'development','reference_class':'disease_a'}]
        emit(view/'split_PRIVATE.jsonl',split)
        emit(view/'patients_model_input_PRIVATE.jsonl',[{'patient_id':'a','view_id':vid,'context':{},'facts':[{'fact_id':'f','feature_id':'HP_1','state':'present','display_text':'Record mentions: Pain','group_id':'HP_1'}]}])
        emit(view/'fact_provenance_PRIVATE.jsonl',[{'fact_id':'f','source_observation_ids':['o'],'sources':['history']}])
        for part in ['train','development','test']:emit(view/(part+'_records_PRIVATE.jsonl'),[{'patient_id':'a','tokens':['HP_1_@_present'],'context':{},'label':'disease_a','partition':part}] if part=='development' else [])
        dump(view/'manifest.json',{'output_sha256':{p.name:sha(p) for p in view.iterdir()}})
        return view,rules,{'view_id':vid,'input_sha256':{'view_config.json':sha(view/'view_config.json')}}
    def test_export_keeps_facts_and_separates_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            d=Path(tmp);v,r,a=self.fixture(d);report=export(v,d/'out',r,a,expected=1)
            self.assertEqual(report['logical_tasks'],4)
            with zipfile.ZipFile(d/'out/TRACE2400_frozen_rollout_input_PRIVATE.zip') as z:
                self.assertFalse(any('outcomes' in n and n!='split_without_outcomes_PRIVATE.jsonl' for n in z.namelist()))
                self.assertNotIn('reference_class',z.read('split_without_outcomes_PRIVATE.jsonl').decode())
                self.assertNotIn('disease_a',z.read('patients_model_input_PRIVATE.jsonl').decode())
                self.assertIn('disease_a',z.read('diagnosis_vocabulary.json').decode())
            self.assertTrue((d/'out/evaluation_PRIVATE/reference_outcomes_PRIVATE.jsonl').exists())
    def test_rehashed_visible_input_drift_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);v,r,a=self.fixture(d)
            p=v/'development_records_PRIVATE.jsonl';emit(p,[{'patient_id':'a','tokens':[],'context':{},'label':'disease_a','partition':'development'}])
            m=read(v/'manifest.json');m['output_sha256'][p.name]=sha(p);dump(v/'manifest.json',m)
            with self.assertRaisesRegex(ValueError,'LLM visible facts differ'):export(v,d/'out',r,a,expected=1)
if __name__=='__main__':unittest.main()
