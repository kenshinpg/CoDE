import sys,unittest,tempfile,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_code10k_interventions import legal_variants
from code_eval.fixed_diagnosis_g0 import select

class InterventionContracts(unittest.TestCase):
    def test_partial_group_change_and_direction_matching(self):
        full={'a','b','c','d'};groups={'g':['a','b'],'h':['c'],'i':['d']};E={'a','c'}
        ge=legal_variants(full,groups,E)
        self.assertEqual(ge['add']['g']['changed_fact_ids'],['b'])
        self.assertEqual(ge['delete']['g']['changed_fact_ids'],['a'])
        self.assertEqual(ge['delete']['g']['fact_ids'],['c'])
        s=select(ge['add'],{'g':2.,'h':-1.,'i':1.},'support','seed')
        self.assertTrue(s['available']);self.assertEqual(s['high'],'g');self.assertEqual(set(s['random']),{'i'})
        self.assertFalse(select(ge['add'],{'g':2.,'h':-1.,'i':-1.},'counter','seed')['available'])
        with self.assertRaises(ValueError):legal_variants(full,groups,{'nonpatient'})

    def test_both_legacy_estimators_accept_new_protocol_and_schema(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as td:
            p=Path(td);cfg=json.loads((ROOT/'configs/code10k_interventions_20261007.json').read_text());cfg['selectors']={'cohd':['support']};cfg['operations']=['add']
            attempts=[];interventions=[];variants=[]
            cfg2=json.loads((ROOT/'configs/code10k_versions_20261007.json').read_text());methods=cfg2['baselines']+cfg2['candidates']
            for patient in ['p1','p2']:
                base=dict(patient_id=patient,pattern_id=patient,explanation_id=patient,d='d',selector='cohd',operation='add',direction='support')
                attempts.append(dict(base,available=True,actual_fact_count=1))
                for vid,fs,q in [('original',['a'],.4),('add|g',['a','b'],.8),('add|h',['a','c'],.5)]:
                    scores={m:q for m in methods};scores.update(B0_cohd=.9,B0_lightgbm=.9,B0_tabr=.9,B0_no_retrieval=.9,B1_fact=len(fs)/3,B1_group=len(fs)/3)
                    variants.append(dict(patient_id=patient,pattern_id=patient,explanation_id=patient,d='d',y='d',variant=vid,fact_ids=fs,actual_fact_count=len(fs),scores=scores,probe={pr:dict(q_d=q,q_y=q,prediction='d' if q>.6 else 'other') for pr in ['encoder','lightgbm']}))
                for c,rep,q in [('high',None,.8),('low',None,.5)]+[('random',i,.5) for i in range(10)]:
                    interventions.append(dict(base,condition=c,repeat=rep,y='d',probe={pr:dict(available=True,q_d_before=.4,q_d_after=q,q_y_before=.4,q_y_after=q,prediction_before='other',prediction_after='d' if q>.6 else 'other') for pr in ['encoder','lightgbm']}))
            for name,rs in [('attempts',attempts),('interventions',interventions),('variant_scores',variants)]:
                (p/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rs))
            (p/'protocol_lock.json').write_text(json.dumps(dict(config=cfg,protocol_sha256='test')))
            (p/'scoring_manifest.json').write_text(json.dumps(dict(status='complete',output_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir()})))
            for script,args in [('analyze_code10k_intervention_effects.py',[]),('analyze_code10k_versions.py',['--config',str(ROOT/'configs/code10k_versions_20261007.json')])]:
                dest=p/script.replace('.py','');subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--source-dir',str(p),'--out',str(dest)]+args,check=True,capture_output=True)
                if 'effects' in script:
                    panels=json.loads((dest/'g1_all_panels.json').read_text());r=panels['cohd|support|add|correct|encoder'];self.assertAlmostEqual(r['metrics']['high_minus_random_d']['mean'],.3)
                else:
                    panels=json.loads((dest/'g2_panels.json').read_text());self.assertTrue(panels);self.assertTrue(all(k.startswith('all|') for k in panels));r=panels['all|correct|add|encoder|support_d'];self.assertEqual(r['metrics']['score:B2_A_tabr']['mean'],1.)

if __name__=='__main__':unittest.main()

