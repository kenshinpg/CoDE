import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import run_trace2400_sources as m

def fixture(base):
    root=base/'view';root.mkdir();old=base/'old';old.mkdir();parts={'train':32,'development':12,'test':8};split=[];records={}
    for part,n in parts.items():
        records[part]=[]
        for i in range(n):
            aid=part+str(i);label=str(i%4)
            split.append({'admission_id':aid,'subject_group':aid,'partition':part,'reference_class':label})
            # Development-only legal state produces explicit unavailability in neural and association sources.
            tokens=['X'+label+'_@_'+('absent' if part=='development' and i==0 else 'present')]
            records[part].append({'patient_id':aid,'input_group_id':aid,'partition':part,'label':label,'tokens':tokens,'context':{},'pattern_id':aid})
    def emit(p,rs):p.write_text(''.join(json.dumps(r)+'\n' for r in rs),encoding='utf-8')
    emit(root/'split_PRIVATE.jsonl',split);(old/'split_PRIVATE.jsonl').write_bytes((root/'split_PRIVATE.jsonl').read_bytes())
    for part in ['train','development']:emit(root/(part+'_records_PRIVATE.jsonl'),records[part])
    # A training pipeline that opens test records will fail parsing this sentinel.
    (root/'test_records_PRIVATE.jsonl').write_text('TEST FILE MUST NOT BE OPENED',encoding='utf-8')
    m.dump(root/'view_report.json',{'view_id':'trace2400_record_evidence_v2_names','missing_display_names':[],'common_llm_input_ready':True})
    m.dump(root/'view_config.json',{})
    m.dump(root/'evidence_dictionary.json',{'features':{'X'+str(i):{'type':'C','values':{'present':'present','absent':'absent'},'group_id':'X'+str(i)} for i in range(4)}})
    refresh(root);return root,old

def refresh(root):m.dump(root/'manifest.json',{'output_sha256':{p.name:m.sha(p) for p in root.iterdir() if p.is_file() and p.name!='manifest.json'}})

class Inputs(unittest.TestCase):
    def test_no_test_features_read(self):
        with tempfile.TemporaryDirectory() as td:
            root,old=fixture(Path(td));data,labels,enc,audit=m.load_inputs(root,old,False)
            self.assertEqual(set(data),{'train','development'});self.assertTrue(audit['previous_split_identical']);self.assertEqual(enc.export()['fixed_context'],[])
    def test_subject_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root,old=fixture(Path(td));s=m.rows(root/'split_PRIVATE.jsonl');s[-1]['subject_group']=s[0]['subject_group']
            (root/'split_PRIVATE.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in s));refresh(root)
            with self.assertRaisesRegex(ValueError,'Subject crosses'):m.load_inputs(root,None,False)
    def test_target_concept_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root,old=fixture(Path(td));d=m.read(root/'evidence_dictionary.json');d['features']['HP_0001733']={'type':'C'};m.dump(root/'evidence_dictionary.json',d);refresh(root)
            with self.assertRaisesRegex(ValueError,'Direct target'):m.load_inputs(root,old,False)
    def test_manifest_drift_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root,old=fixture(Path(td));(root/'train_records_PRIVATE.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):m.load_inputs(root,old,False)

@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('lightgbm'),'torch/lightgbm not installed locally')
class FitSmoke(unittest.TestCase):
    def test_reused_models_tiny_cpu_fit(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root,old=fixture(base);out=base/'out';out.mkdir()
            r=m.run(SimpleNamespace(view=root,previous_view=old,out=out,threads=1,device='cpu',seed=261013,rounds=2,epochs=2,synthetic_smoke=True))
            self.assertEqual(r['status'],'complete');self.assertEqual(r['partition'],'development')
            self.assertEqual(r['results']['1.0']['common_population_n'],11)
            self.assertEqual(r['results']['1.0']['methods']['tree_full_only']['available'],12)
            self.assertEqual(r['results']['0']['common_population_n'] if '0' in r['results'] else r['results']['0.0']['common_population_n'],12)
            self.assertTrue((out/'encoder.pt').exists())
if __name__=='__main__':unittest.main()
