"""Public DDXPlus fact/coverage example; optional fitting on official CSV inputs."""
import argparse,ast,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/CoDE/src'))
from code_eval.explanation_scoring import score_coverage,validate_mapping
from code_eval.formal_sources import AssociationSource

def read_csv(path,limit):
    out=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            out.append(dict(tokens=sorted(set(ast.literal_eval(r['EVIDENCES']))),
                            context=dict(age=float(r['AGE']),sex=r['SEX']),label=r['PATHOLOGY']))
            if len(out)==limit:break
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,help='Official DDXPlus release_train_patients.csv')
    p.add_argument('--test',type=Path,help='Official DDXPlus release_test_patients.csv')
    p.add_argument('--limit',type=int,default=2048)
    a=p.parse_args()
    if bool(a.train)!=bool(a.test):p.error('Supply --train and --test together')
    if a.limit<2:p.error('--limit must be at least two')
    if a.train:
        train=read_csv(a.train,a.limit);test=read_csv(a.test,a.limit)
        labels=sorted({r['label'] for r in train});source=AssociationSource(labels);source.fit(train)
        # Demonstration selection is fixed and outcome-independent.
        results=[]
        excluded=dict(reference_label_not_in_training=0, empty_fact_set=0,
                      unseen_fact_tokens=0)
        for r in test[:8]:
            if r['label'] not in labels:
                excluded['reference_label_not_in_training']+=1
                continue
            if not r['tokens']:
                excluded['empty_fact_set']+=1
                continue
            groups={}
            for token in r['tokens']:groups.setdefault(token.split('_@_')[0],[]).append(token)
            selected={x for g in sorted(groups)[::2] for x in groups[g]}
            # Keep the proposed diagnosis distinct from the reference label.
            full=source.distribution(r['tokens'],r['context'])
            if full is None:
                excluded['unseen_fact_tokens']+=1
                continue
            diagnosis=labels[max(range(len(labels)),key=lambda i:full[i])]
            def q(fs,d):
                probs=source.distribution(sorted(fs),r['context'])
                if probs is None:raise ValueError('Unavailable source input')
                return probs[labels.index(d)]
            result=score_coverage(r['tokens'],sorted(selected),list(groups.values()),diagnosis,q,empty_supported=True)
            result['uniform_group_coverage']=sum(result['group_coverage'])/len(result['group_coverage'])
            results.append(result)
        output=dict(mode='small association fit on supplied public DDXPlus CSVs',
                    training_records=len(train),labels=len(labels),examples=results,
                    considered_records=len(test[:8]), scored_records=len(results),
                    excluded_records=excluded,
                    status='complete' if results else 'no_evaluable_records',
                    interpretation='Interface demonstration; separate from the paper cohort and effect estimates')
    else:
        examples=json.loads((ROOT/'examples/ddxplus_facts.json').read_text())
        results=[]
        for r in examples:
            groups={}
            for f in r['facts']:groups.setdefault(f['group_id'],[]).append(f['fact_id'])
            facts=[f['fact_id'] for f in r['facts']]
            cited=[x for g in sorted(groups)[::2] for x in groups[g]]
            f,e,b=validate_mapping(facts,cited,list(groups.values()))
            fractions=[len(e&g)/len(g) for g in b]
            results.append(dict(example=r['example'],facts=len(f),cited_facts=len(e),groups=len(b),
                                uniform_group_coverage=sum(fractions)/len(fractions),
                                citation_rule='every second dependency group, by sorted group ID'))
        output=dict(mode='actual DDXPlus-derived input and coverage calculation',examples=results,
                    interpretation='Deterministic demonstration citations; no generated-rationale accuracy or learned support result',
                    fitted_support='Supply official --train and --test CSVs to run the association example')
    dest=ROOT/'build/ddxplus_example.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(output,indent=2));print('Saved:',dest)
    if not results:
        print('No evaluable records. See the output JSON for record counts and exclusion reasons.',
              file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':sys.exit(main())
