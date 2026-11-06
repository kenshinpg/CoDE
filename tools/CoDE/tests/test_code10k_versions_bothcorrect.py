import json,hashlib,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class BothCorrectContract(unittest.TestCase):
 def test_reference_correct_filter_not_proposal_correct(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tests') as td:
   p=Path(td);cfg=json.loads((ROOT/'configs/code10k_versions_bothcorrect_20261007.json').read_text());cfg['bootstrap']=10;cfgpath=p/'config.json';cfgpath.write_text(json.dumps(cfg));ms=cfg['baselines']+cfg['candidates'];rs=[]
   for patient,other in [('p1','y'),('p2','d')]:
    for vid,fs,q,pred in [('original',['a'],.1,'y'),('add|g',['a','b'],.3,'y'),('add|h',['a','c'],.1,other)]:
     scores={m:q for m in ms};scores.update(B0_cohd=.8,B0_lightgbm=.8,B1_fact=len(fs)/3)
     rs.append(dict(patient_id=patient,pattern_id=patient,explanation_id=patient,d='d',y='y',variant=vid,fact_ids=fs,actual_fact_count=len(fs),scores=scores,probe={pr:dict(q_d=q,q_y=1-q,prediction=pred) for pr in ['encoder','lightgbm']}))
   raw=''.join(json.dumps(r)+'\n' for r in rs).encode();(p/'variant_scores.jsonl').write_bytes(raw);(p/'scoring_manifest.json').write_text(json.dumps(dict(output_sha256={'variant_scores.jsonl':hashlib.sha256(raw).hexdigest()})))
   subprocess.run([sys.executable,str(ROOT/'scripts/analyze_code10k_versions_bothcorrect.py'),'--source-dir',str(p),'--config',str(cfgpath),'--out',str(p/'out')],check=True,capture_output=True)
   panels=json.loads((p/'out/g2_panels.json').read_text());r=panels['all|wrong|add|encoder|support_d_gap_001_bothcorrect'];self.assertEqual(r['counts']['all_pairs'],2);self.assertEqual(r['counts']['eligible_pairs'],1);self.assertEqual(r['metrics']['score:B2_A_lightgbm']['patients'],1)
if __name__=='__main__':unittest.main()
