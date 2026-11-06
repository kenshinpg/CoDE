import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from code_eval import role_sensitivity as role
from code_eval.adjudication_sensitivity import contrasts,bootstrap_block
from code_eval.round1_followup import resample
from test_adjudication_sensitivity import record


def sample():
    rr=[]
    for i in range(8):
        for v in ['all','first','support']:
            r=record(i,v);r['targeted_sample']=False
            if v!='all':r['scores']['tree_positive_shap']+=.15 if i%2 else -.1
            if i==1 and v=='support':
                r.update(reference_failure=True,proposal_failure=True,prediction='X',cited_probabilities=[.1,.9],fact_ids=[])
            rr.append(r)
    return rr


class RoleContracts(unittest.TestCase):
    @unittest.skipUnless((ROOT/'inputs/round1_roles_20261023/role_sets_PRIVATE.jsonl').exists(), 'Requires separately authorized audit input')
    def test_actual_roles_resolve_without_reference_drift(self):
        b=ROOT/'inputs/round1_roles_20261023'
        roles=[json.loads(l) for l in (b/'role_sets_PRIVATE.jsonl').read_text(encoding='utf-8').splitlines()]
        keys=json.loads((b/'analysis_key_PRIVATE.json').read_text(encoding='utf-8'))
        rr=role.resolve(roles,keys); cc=role.census(rr)
        self.assertEqual(len(rr),100);self.assertEqual(sum(x['all_citations'] for x in cc),1315)
        self.assertEqual(sum(x['support_empty'] for x in cc),13)
        self.assertTrue(all(r['sets']['first']==r['sets']['support'] for r in rr if r['dataset']=='DDXPlus'))
        bad=copy.deepcopy(roles);bad[0]['E_all']=bad[0]['E_all'][:-1]
        with self.assertRaises(ValueError):role.resolve(bad,keys)
        # The real frozen first diagnosis (not an invented default) is preserved in all four variants.
        r=rr[0];original={r['explanation_id']:dict(patient_id=r['patient_id'],partition='test',reviewed_available=True,
                      diagnosis=r['role_first_diagnosis'],cited_fact_ids=r['sets']['all'])}
        ledger,lookup=role.make_ledger([r],original,{r['patient_id']:set(r['sets']['all'])})
        self.assertEqual(len(ledger),4);self.assertEqual(len(lookup),4)
        self.assertTrue(all(x['diagnosis']==r['role_first_diagnosis'] for x in ledger))

    def test_three_way_common_population_and_empty_support(self):
        rr=sample();next(r for r in rr if r['explanation_id']=='e0' and r['version']=='support')['scores']['tree_positive_shap']=None
        panels,flows=role.make_panels(rr)
        self.assertEqual(len(panels),48)
        for p in [panels[0],panels[24]]:
            self.assertEqual(len(p['comparisons'][0]['pairs']),8)
            self.assertEqual(len(p['comparisons'][1]['pairs']),7)
            self.assertFalse(any(a['explanation_id']=='e0' for a,b in p['comparisons'][1]['pairs']))
            self.assertTrue(any(a['explanation_id']=='e1' for a,b in p['comparisons'][1]['pairs']))
        self.assertEqual(panels[0]['restricted_set'],'first')
        self.assertEqual(panels[24]['restricted_set'],'support')

    def test_replay_and_updated_vs_fixed_all_endpoint(self):
        r=dict(dataset='DDXPlus',explanation_id='e1',audit_id='N1',patient_id='p1',status='complete',
               random_sample=True,targeted_sample=False,automatic_fact_ids=['f'],sets={'all':['f'],'first':['f'],'support':[]})
        lookup={v:(r,v) for v in ['automatic','all','first','support']}; scored=[];old={}
        for probe in ['encoder','lightgbm']:
            a=record(1);a['probe']=probe;old['e1',probe]=copy.deepcopy(a)
            for v in lookup:
                b=copy.deepcopy(a);b['explanation_id']=v
                if v=='support':b.update(reference_failure=True,proposal_failure=True,cited_probabilities=[.1,.9],cited_fact_count=0)
                scored.append(b)
        cooked,check=role.cook(scored,lookup,old,['D','X'])
        self.assertEqual(len(cooked),6);self.assertEqual(check['support']['automatic_rows'],2)
        self.assertTrue(all(r['prediction']=='X' for r in cooked if r['version']=='support'))
        p=role.make_panels(sample())[0][24];draws=resample(contrasts(p),30,261023)
        rr,tt=role.finish_panel(p,draws,30,261023)
        self.assertTrue(all(t['success_to_failure']==1 for t in tt))
        self.assertTrue(all(t['restricted_failures']==t['all_failures']+1 for t in tt))
        self.assertTrue(all('restricted_fixed_all_endpoint_difference' in r for r in rr))
        self.assertFalse(any(k.startswith('automatic_') for r in rr for k in r))

    def test_parallel_summary_matches_serial_and_resume(self):
        from run_round1_role_sensitivity import summarize,emit,read
        # One populated panel is sufficient to exercise process spawn, paired bootstrap and checkpoint reuse.
        import run_round1_role_sensitivity as runner
        rr=sample();p=role.make_panels(rr)[0][24];p['panel_id']=0
        original=runner.make_panels;runner.make_panels=lambda records:([p],[])
        try:
            with tempfile.TemporaryDirectory() as temp:
                a=SimpleNamespace(out=Path(temp),bootstrap=300,seed=261023,summary_workers=2)
                emit(a.out/'scored_records_PRIVATE.jsonl',rr);summarize(a,rr)
                draws=np.concatenate([np.load(f)['draws'] for f in sorted((a.out/'bootstrap').glob('*.npz'))])
                np.testing.assert_array_equal(draws,resample(contrasts(p),300,261023))
                first=read(a.out/'statistics.json');summarize(a,rr);second=read(a.out/'statistics.json')
                self.assertEqual(first['results'],second['results'])
                a.seed+=1
                with self.assertRaises(ValueError):summarize(a,rr)
        finally:runner.make_panels=original

if __name__=='__main__':unittest.main()
