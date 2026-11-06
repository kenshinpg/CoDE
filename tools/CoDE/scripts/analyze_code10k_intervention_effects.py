"""All prespecified G1 contrasts on the frozen G0 sample; patient-level inference."""
import argparse,hashlib,json,tarfile
from collections import defaultdict
from pathlib import Path
import numpy as np
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def effect(r,probe,target):
 p=r['probe'][probe];delta=p['q_'+target+'_after']-p['q_'+target+'_before']
 return delta if r['operation']=='add' else -delta
def main():
 p=argparse.ArgumentParser();src=p.add_mutually_exclusive_group(required=True);src.add_argument('--archive',type=Path);src.add_argument('--source-dir',type=Path);p.add_argument('--exclude-patients',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.archive:
  with tarfile.open(a.archive) as t:f={m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
 else:f={p.name:p.read_bytes() for p in a.source_dir.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')}
 m=json.loads(f['scoring_manifest.json']);assert m['status']=='complete'
 for n,h in m['output_sha256'].items():assert hashlib.sha256(f[n]).hexdigest()==h
 cfg=json.loads(f['protocol_lock.json'])['config'];eps=cfg['zero_tolerance']
 rows=[json.loads(l) for l in f['interventions.jsonl'].splitlines()];attempts=[json.loads(l) for l in f['attempts.jsonl'].splitlines()];vs=[json.loads(l) for l in f['variant_scores.jsonl'].splitlines()];orig={r['explanation_id']:r for r in vs if r['variant']=='original'}
 groups=defaultdict(list)
 for r in rows:groups[(r['explanation_id'],r['selector'],r['direction'],r['operation'])].append(r)
 assert len(groups)==sum(r['available'] for r in attempts)
 excluded=set(json.loads(a.exclude_patients.read_text(encoding='utf8'))) if a.exclude_patients else set()
 assert excluded<={r['patient_id'] for r in attempts},'Excluded patient not found'
 attempts=[r for r in attempts if r['patient_id'] not in excluded]
 panels={};patient_records=[]
 for sel,dirs in cfg['selectors'].items():
  for direction in dirs:
   for op in cfg['operations']:
    for st in ['correct','wrong','all']:
     eligible=[r for r in attempts if (r['selector'],r['direction'],r['operation'])==(sel,direction,op) and (st=='all' or (orig[r['explanation_id']]['d']==orig[r['explanation_id']]['y'])==(st=='correct'))]
     for probe in ['encoder','lightgbm']:
      key='|'.join([sel,direction,op,st,probe]);by=defaultdict(list);nexp=0;unavailable=0;flip_events={c:0 for c in ['high','low','random']}
      for at in eligible:
       if not at['available']:continue
       rr=groups[(at['explanation_id'],sel,direction,op)]
       assert len(rr)==12
       if not all(r['probe'][probe]['available'] for r in rr):unavailable+=1;continue
       cond={'high':[r for r in rr if r['condition']=='high'],'low':[r for r in rr if r['condition']=='low'],'random':sorted([r for r in rr if r['condition']=='random'],key=lambda r:r['repeat'])}
       assert [len(cond[c]) for c in ['high','low','random']]==[1,1,10]
       values={}
       for c,cr in cond.items():
        for target in ['d','y']:
         values[c+'_effect_'+target]=float(np.mean([effect(r,probe,target) for r in cr]))
        for name,fn in [('top1_flip',lambda r:r['probe'][probe]['prediction_after']!=r['probe'][probe]['prediction_before']),('reference_correct_change',lambda r:int(r['probe'][probe]['prediction_after']==r['y'])-int(r['probe'][probe]['prediction_before']==r['y'])),('proposal_support_change',lambda r:int(r['probe'][probe]['prediction_after']==r['d'])-int(r['probe'][probe]['prediction_before']==r['d']))]:
         values[c+'_'+name]=float(np.mean([fn(r) for r in cr]))
        if direction!='unsigned':
         sign=1 if direction=='support' else -1
         values[c+'_direction_agreement_d']=float(np.mean([sign*effect(r,probe,'d')>eps for r in cr]))
        values[c+'_numerical_zero_d']=float(np.mean([abs(effect(r,probe,'d'))<=eps for r in cr]))
        flip_events[c]+=sum(r['probe'][probe]['prediction_after']!=r['probe'][probe]['prediction_before'] for r in cr)
       for t in ['d','y']:
        values['high_minus_random_'+t]=values['high_effect_'+t]-values['random_effect_'+t]
        values['high_minus_low_'+t]=values['high_effect_'+t]-values['low_effect_'+t]
       by[at['patient_id']].append(values);nexp+=1
      patient_ids=sorted(by);metrics={}
      if patient_ids:
       cols=list(next(iter(by.values()))[0]);arr=np.array([[np.mean([v[c] for v in by[pid]]) for c in cols] for pid in patient_ids]);rng=np.random.default_rng(cfg['seed']);boot=np.array([arr[rng.integers(0,len(arr),len(arr))].mean(axis=0) for _ in range(500)])
       for i,c in enumerate(cols):metrics[c]={'mean':float(arr[:,i].mean()),'ci95':np.quantile(boot[:,i],[.025,.975]).tolist() if len(arr)>1 else None}
       patient_records.extend({'panel':key,'patient_id':pid,'explanations':len(by[pid]),'metrics':dict(zip(cols,map(float,arr[i])))} for i,pid in enumerate(patient_ids))
      panels[key]={'attempted_explanations':len(eligible),'matched_explanations':sum(r['available'] for r in eligible),'probe_unavailable_explanations':unavailable,'analyzed_explanations':nexp,'patients':len(by),'comparison_unit':'average within patient before equal-weight patient bootstrap','metrics':metrics,'raw_flip_counts':flip_events,'random_rows_not_independent':True}
 a.out.mkdir(parents=True,exist_ok=True);dump(a.out/'g1_all_panels.json',panels);dump(a.out/'g1_patient_aggregates.json',patient_records)
 dump(a.out/'provenance.json',{'archive_sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest() if a.archive else None,'source_manifest_sha256':hashlib.sha256(f['scoring_manifest.json']).hexdigest(),'excluded_patients':sorted(excluded),'analyzed_input_patients':len({r['patient_id'] for r in attempts}),'bootstrap':500,'seed':cfg['seed'],'protocol_sha256':json.loads(f['protocol_lock.json'])['protocol_sha256'],'status':'retrospectively frozen test G1; no multiplicity adjustment; no new fitting','empty_panels':sum(v['patients']==0 for v in panels.values()),'all_change_counts':sorted({r['actual_fact_count'] for r in attempts if r['available']}),'positive_delta_definition':'addition after-minus-before; deletion before-minus-after; counter expected negative, unsigned no directional hypothesis','strata':'correct/wrong primary; pooled all descriptive only','zero_threshold':'1e-10 numerical only, not clinical threshold','CI_warning':'Patient bootstrap does not establish fit stability or clinical significance.'})
 names={'cohd':'COHD','local_shap':'局部SHAP','class_shap':'类SHAP'}
 lines=['# G1 单组增删：全部预设条件','','Δ：补回为after−before，删除为before−after。主对比为高重要性Δ−10次随机平均Δ。先患者内平均，再500次患者bootstrap。以下乘100表示支持值百分点；不是诊断准确率百分点。反证预期负向，无符号方法不预设方向。','']
 for probe in ['encoder','lightgbm']:
  for st in ['correct','wrong','all']:
   lines += [f'## {probe} / {st}','','| 方法/方向/操作 | 解释/患者 | 高Δd | 随机Δd | 差值及95% CI（百分点） | 高干预top1翻转条数 |','|---|---:|---:|---:|---|---:|']
   for sel,dirs in cfg['selectors'].items():
    for direction in dirs:
     for op in cfg['operations']:
      v=panels['|'.join([sel,direction,op,st,probe])];label=f'{names[sel]}/{direction}/{op}'
      if not v['metrics']:lines.append(f'| {label} | 0/0 | NA | NA | 不可比较 | NA |');continue
      z=v['metrics'];c=z['high_minus_random_d'];ci=c['ci95'];ct=f'[{100*ci[0]:.4f}, {100*ci[1]:.4f}]' if ci else '不足2位患者'
      lines.append(f'| {label} | {v["analyzed_explanations"]}/{v["patients"]} | {100*z["high_effect_d"]["mean"]:.4f} | {100*z["random_effect_d"]["mean"]:.4f} | {100*c["mean"]:+.4f} {ct} | {v["raw_flip_counts"]["high"]} |')
 (a.out/'01_G1全部预设条件结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print('Panels',len(panels),'empty',sum(v['patients']==0 for v in panels.values()))
if __name__=='__main__':main()
