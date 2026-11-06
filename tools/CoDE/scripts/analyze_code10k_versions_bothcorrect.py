"""Fixed-diagnosis equal-count version ranking; no training or API calls."""
import argparse, hashlib, itertools, json, tarfile
from collections import defaultdict
from pathlib import Path
import numpy as np

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')

def agreement(score_delta, outcome_delta, eps):
    return np.where(np.abs(score_delta) <= eps, .5, (score_delta*outcome_delta > 0).astype(float))

def finite(x):
    return x is not None and np.isfinite(x)

def estimate(values, cfg):
    x=np.asarray(values, dtype=float)
    if not len(x): return None
    ci=None
    if len(x)>1:
        rng=np.random.default_rng(cfg['seed'])
        means=np.mean(x[rng.integers(0,len(x),(cfg['bootstrap'],len(x)))],axis=1)
        ci=np.quantile(means,[.025,.975]).tolist()
    return {'mean':float(x.mean()), 'ci95':ci, 'patients':len(x)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text(encoding='utf8'));eps=cfg['numerical_tolerance'];a.out.mkdir(parents=True,exist_ok=True)
    raw=(a.source_dir/'variant_scores.jsonl').read_bytes();manifest=json.loads((a.source_dir/'scoring_manifest.json').read_text());pilot=set()
    assert hashlib.sha256(raw).hexdigest()==manifest['output_sha256']['variant_scores.jsonl']
    grouped=defaultdict(list)
    for line in raw.splitlines():
        r=json.loads(line);grouped[r['explanation_id']].append(r)
    methods=cfg['baselines']+cfg['candidates']
    # Entries are explanation averages; repeated variants never become independent patients.
    accum=defaultdict(lambda:defaultdict(list));counts=defaultdict(lambda:defaultdict(int));audit=defaultdict(int)
    for index,(eid,rows) in enumerate(grouped.items(),1):
        if index%100==0:print("Version summaries",index,"/",len(grouped),flush=True)
        original=[r for r in rows if r['variant']=='original'];assert len(original)==1
        origin=original[0];pid=origin['patient_id'];stratum='correct' if origin['d']==origin['y'] else 'wrong'
        buckets=defaultdict(dict)
        for r in rows:
            assert all(r[k]==origin[k] for k in ['patient_id','d','y','pattern_id'])
            assert r['actual_fact_count']==len(set(r['fact_ids']))
            for k in ['B0_cohd','B0_lightgbm']:
                assert r['scores'][k]==origin['scores'][k]
            if r['variant']=='original':continue
            op=r['variant'].split('|')[0];assert op in ['add','delete']
            f=frozenset(r['fact_ids']);o=frozenset(origin['fact_ids'])
            assert (o<f if op=='add' else f<o)
            key=(op,len(f))
            if f in buckets[key]:
                assert r['scores']==buckets[key][f]['scores'];audit['duplicate_versions']+=1
            else:buckets[key][f]=r
        for op in ['add','delete']:
            pairs=[]
            for (operation,n),variants in buckets.items():
                if operation==op:pairs.extend(itertools.combinations(variants.values(),2))
            if not pairs:continue
            audit['all_pairs']+=len(pairs)
            deltas=np.asarray([[float(l['scores'][m])-float(r['scores'][m]) if finite(l['scores'][m]) and finite(r['scores'][m]) else np.nan for m in methods] for l,r in pairs])
            assert np.all(np.abs(deltas[:,methods.index('B1_fact')])<=eps)
            for probe in ['encoder','lightgbm']:
                for endpoint in cfg['endpoints']:
                    if endpoint.startswith('support'):
                        dy=np.asarray([l['probe'][probe]['q_d']-r['probe'][probe]['q_d'] for l,r in pairs],dtype=float)
                        threshold=cfg['support_gap_sensitivity'] if 'gap_001' in endpoint else eps
                        eligible=np.isfinite(dy)&(np.abs(dy)>eps)&(np.abs(dy)>=threshold)
                        if endpoint.endswith('bothcorrect'):
                            eligible &= np.asarray([l['probe'][probe]['prediction']==l['y'] and r['probe'][probe]['prediction']==r['y'] for l,r in pairs],dtype=bool)
                    else:
                        target=endpoint[-1]
                        dy=np.asarray([int(l['probe'][probe]['prediction']==l[target])-int(r['probe'][probe]['prediction']==r[target]) for l,r in pairs],dtype=float)
                        eligible=dy!=0
                    for pop in ['all']:
                        panel='|'.join([pop,stratum,op,probe,endpoint]);counts[panel]['all_pairs']+=len(pairs);counts[panel]['eligible_pairs']+=int(eligible.sum());counts[panel]['all_explanations']+=1
                        for j,m in enumerate(methods):
                            mask=eligible&np.isfinite(deltas[:,j]);v=agreement(deltas[mask,j],dy[mask],eps)
                            metric='score:'+m
                            counts[panel][metric+':pairs']+=int(mask.sum())
                            if len(v):accum[(panel,metric)][pid].append(float(v.mean()))
                        for m in cfg['candidates']:
                            j=methods.index(m)
                            for base in cfg['comparators']:
                                k=methods.index(base);mask=eligible&np.isfinite(deltas[:,j])&np.isfinite(deltas[:,k]);metric='difference:'+m+'-'+base
                                counts[panel][metric+':pairs']+=int(mask.sum())
                                if mask.any():
                                    v=agreement(deltas[mask,j],dy[mask],eps)-agreement(deltas[mask,k],dy[mask],eps)
                                    accum[(panel,metric)][pid].append(float(v.mean()))
    results={};patient_rows=[]
    for panel,c in sorted(counts.items()):
        metrics={}
        for (pn,metric),by in accum.items():
            if pn!=panel:continue
            values=[]
            for pid,vs in sorted(by.items()):
                v=float(np.mean(vs));values.append(v);patient_rows.append({'panel':panel,'metric':metric,'patient_id':pid,'value':v,'explanations':len(vs)})
            metrics[metric]=estimate(values,cfg);metrics[metric]['explanations']=sum(len(vs) for vs in by.values());metrics[metric]['pairs']=c[metric+':pairs']
        results[panel]={'counts':dict(c),'metrics':metrics}
    dump(a.out/'g2_panels.json',results);dump(a.out/'patient_aggregates.json',patient_rows);dump(a.out/'protocol.json',cfg)
    dump(a.out/'provenance.json',{'input_archive_sha256':None,'variant_sha256':hashlib.sha256(raw).hexdigest(),'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'patients':len({r['patient_id'] for rs in grouped.values() for r in rs}),'explanations':len(grouped),'panels':len(results),'audit':dict(audit),'status':'complete; retrospectively frozen test; no training or API'})
    lines=['# G2 全部固定诊断、等事实数版本比较','','分数越大是否对应探针对提议诊断的支持越大；错误提议不代表临床质量。top1_y为另列的目标错位敏感性分析。排序并列计0.5，终点并列排除；患者等权，区间未校正多重比较。差值均在共同可评分配对上计算。','']
    for panel,result in results.items():
        lines += ['## '+panel,'',f"候选配对 {result['counts']['all_pairs']}；终点可区分配对 {result['counts']['eligible_pairs']}。",'', '| 指标 | 患者/解释/配对 | 均值% [95% CI] |','|---|---:|---|']
        for m,z in result['metrics'].items():
            ci=z['ci95'];interval=f'[{100*ci[0]:.2f}, {100*ci[1]:.2f}]' if ci else 'NA'
            lines.append(f"| {m} | {z['patients']}/{z['explanations']}/{z['pairs']} | {100*z['mean']:.2f} {interval} |")
        lines.append('')
    (a.out/'G2全部结果.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'panels':len(results),'pairs':audit['all_pairs'],'out':str(a.out)},ensure_ascii=False))

if __name__=='__main__':main()
