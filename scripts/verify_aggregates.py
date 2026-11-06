"""Check paper exports against full-precision accepted aggregate inputs."""
from pathlib import Path
import csv,json,math,re

ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig',newline='')))
def one(rs,**kw):
    found=[r for r in rs if all(r.get(k)==v for k,v in kw.items())]
    assert len(found)==1,(kw,len(found))
    return found[0]
def verify():
    panels=load('results/figure_inputs/accepted_panels.json')
    methods={'Association':'cohd','Representation':'encoder','Group-enhanced':'grasp',
             'Tree removal':'lightgbm','TabR-S':'tabr','No-retrieval':'no_retrieval',
             'Global gain':'tree_gain','Local absolute SHAP':'tree_local_abs_shap',
             'Class SHAP':'tree_class_abs_shap','Voting':'voting'}
    count=0
    for ds,filename in [('DDXPlus','ddx_selection.csv'),('TRACE','trace_selection.csv')]:
        for row in rows('results/aggregate/'+filename):
            method=methods[row['method']];probe='encoder' if row['probe'].startswith('R') else 'lightgbm'
            family='tree_rules' if method.startswith('tree_') else 'retrieval' if method in ['no_retrieval','voting'] else 'sources'
            for budget in [25,50,75]:
                g=one(panels['selection'][ds],probe=probe,selector=method,family=family,population='both_full_correct',fraction=budget/100)['gain']
                expected=f"{100*g['mean']:.2f} [{100*g['ci95'][0]:.2f}, {100*g['ci95'][1]:.2f}]"
                assert row[f'budget{budget}_estimate_CI']==expected,(ds,row['method'],budget)
                count+=1
    reference=load('results/figure_inputs/figure1_expected.json')
    for row in reference:
        if row['panel'] not in ['0a','1a']:continue
        ds='DDXPlus' if row['panel']=='0a' else 'TRACE';method=row['method']
        family='tree_rules' if method.startswith('tree_') else 'sources'
        g=one(panels['selection'][ds],probe='encoder',selector=method,family=family,population='both_full_correct',fraction=.25)['gain']
        assert abs(row['mean']-g['mean']*100)<1e-10
        assert all(abs(x-y*100)<1e-10 for x,y in zip(row['ci'],g['ci95']))
    generated=ROOT/'build/figures/fig01_source_data.json'
    if generated.exists():
        assert json.loads(generated.read_text(encoding='utf-8'))==reference,'Overview plot differs from final-proof reference'
    table_index=load('docs/table_index.json')
    assert len(table_index)==34
    for item in table_index:assert rows(item['file']),item['file']
    artifact_map=rows('docs/paper_artifact_map.csv');assert len(artifact_map)==40
    for item in artifact_map:
        for source in item['results'].split('; '):assert (ROOT/source).exists(),source
        for source in item['implementation'].split('; '):
            if source.startswith(('tools/','scripts/','docs/')):assert (ROOT/source).exists(),source
    arrays=load('results/figure_inputs/ddx_intervention_effects.json');ecdf=0
    for row in rows('results/aggregate/round1_cached_statistics/intervention_patient_distribution.csv'):
        if row['target']!='d':continue
        key='|'.join(row[k] for k in ['selector','direction','operation','proposal_stratum','probe'])
        if key in arrays:
            values=arrays[key]
            assert len(values)==int(row['patients'])
            assert abs(sum(values)/len(values)-float(row['mean_pp']))<1e-10
            ecdf+=1
    result=dict(selection_estimates_and_intervals=count,overview_source_points=len(reference),
                accepted_tables=len(table_index),mapped_items=len(artifact_map),ecdf_summary_checks=ecdf,
                status='passed')
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':verify()
