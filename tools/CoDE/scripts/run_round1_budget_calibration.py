"""Read-only fact-count matching and uncalibrated probability diagnostics.

No training, model prediction or temperature fitting. Existing selected sets are
retained; matching conditions on their actual fact counts. Test-only diagnostics
cannot replace development-fitted temperature calibration.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,json,hashlib,time,tarfile,traceback,multiprocessing
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def records(p):
 with Path(p).open(encoding='utf-8-sig') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(p,rows):
 if not rows:return
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def matched_job(task):
 meta,aa,bb=task
 common=sorted(set(aa)&set(bb));valid=[i for i in common if aa[i]['available'] and bb[i]['available']]
 if meta['population']=='both_full_correct':valid=[i for i in valid if aa[i]['both_full_correct'] and bb[i]['both_full_correct']]
 eq=[i for i in valid if aa[i]['fact_count']==bb[i]['fact_count']]
 out=[]
 for mode,ids in [('common_available',valid),('equal_actual_fact_count',eq)]:
  dif=np.array([aa[i]['correct']-bb[i]['correct'] for i in ids],float)
  mean=float(dif.mean()) if len(dif) else None;ci=[None,None]
  seed=int.from_bytes(hashlib.sha256(json.dumps([meta,mode],sort_keys=True).encode()).digest()[:4],'little')
  if len(dif)>1:
   rng=np.random.default_rng(seed);boots=[]
   for j in range(0,5000,100):boots.extend(dif[rng.integers(len(dif),size=(100,len(dif)))].mean(1))
   ci=np.quantile(boots,[.025,.975]).tolist()
  out.append(dict(meta,matching=mode,common_available=len(valid),matched=len(eq),match_fraction=len(eq)/len(valid) if valid else None,patients_or_admissions=len(ids),left_accuracy=float(np.mean([aa[i]['correct'] for i in ids])) if ids else None,right_accuracy=float(np.mean([bb[i]['correct'] for i in ids])) if ids else None,difference=mean,ci_low=ci[0],ci_high=ci[1],seed=seed,interpretation='post-review conditional paired comparison; fixed selections and fits; marginal patient/admission bootstrap'))
 return out
def calibration(p,y,bins=10):
 p=np.asarray(p,dtype=float);y=np.asarray(y,dtype=int)
 if len(p)==0:return None
 assert p.ndim==2 and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5)
 assert ((y>=0)&(y<p.shape[1])).all()
 conf=p.max(1);correct=(p.argmax(1)==y).astype(float);index=np.minimum((conf*bins).astype(int),bins-1)
 ece=0.;out=[]
 for i in range(bins):
  q=index==i;n=int(q.sum());acc=float(correct[q].mean()) if n else None;c=float(conf[q].mean()) if n else None
  if n:ece+=n/len(p)*abs(acc-c)
  out.append(dict(bin=i,lower=i/bins,upper=(i+1)/bins,n=n,accuracy=acc,confidence=c))
 brier=np.sum(p*p,axis=1)-2*p[np.arange(len(y)),y]+1
 return dict(n=len(y),accuracy=float(correct.mean()),ece=float(ece),nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),brier=float(brier.mean())),out
def main(a):
 cfg=read(ROOT/'configs/round1_followup_20261016.json');tasks=[];audit={};cal=[];binrows=[];inventory=[]
 for ds,source in [('DDX','artifacts/code10k_rq1_v1_retry1'),('TRACE','artifacts/trace2400_bce_audit_v1')]:
  source=ROOT/source;manifest=read(source/'run_manifest.json')
  assert manifest['status']==('complete' if ds=='DDX' else 'complete_gates_passed')
  files=sorted(source.glob('shard_*/selections.jsonl')) if ds=='DDX' else [source/'rq1/selections_PRIVATE.jsonl']
  assert files
  data={};by=defaultdict(dict)
  for f in files:
   if ds=='DDX':assert read(f.parent/'run_manifest.json')['status']=='complete'
   expected=manifest.get('output_sha256',{}).get(f.relative_to(source).as_posix())
   if expected:assert sha(f)==expected
   audit[str(f.relative_to(ROOT))]=sha(f)
   for r in records(f):
    key=(r['patient_id'],r['selector'],r['probe'],r['fraction']);assert key not in data;data[key]=r
    if r['available']:assert r['fact_count']==len(r['selected_tokens']) and r['correct'] in [0,1]
    by[(r['selector'],r['probe'],r['fraction'])][r['patient_id']]=r
  assert len(data)==(1506*10*3*2 if ds=='DDX' else 359*9*3*2),(ds,len(data))
  print('LOADED selections',ds,len(data),'selectors',sorted({k[0] for k in by}),flush=True)
  for left,right in [('tree_local_abs_shap','tree_class_abs_shap'),('tabr','no_retrieval')]:
   names={k[0] for k in by}
   assert left in names and right in names,(ds,left,right,sorted(names))
   for probe in ['encoder','lightgbm']:
    for frac in [.25,.5,.75]:
     for pop in ['all_available','both_full_correct']:
      tasks.append((dict(dataset=ds,left=left,right=right,probe=probe,budget=frac,population=pop),by[left,probe,frac],by[right,probe,frac]))
  label_file=ROOT/cfg['datasets'][ds]['models']/'fitted_sources.json' if ds=='DDX' else source/'class_shap.json'
  labels=read(label_file)['labels'];assert len(labels)==(46 if ds=='DDX' else 4)
  audit[str(label_file.relative_to(ROOT))]=sha(label_file)
  if ds=='TRACE':
   subjects={r['patient_id']:r['subject_group'] for r in data.values()}
   assert len(set(subjects.values()))==len(subjects),'Repeated test subjects require cluster resampling'
  for rq in ['rq2','rq3']:
   path=ROOT/cfg['datasets'][ds][rq];expected=cfg['accepted_score_sha256'].get(ds+'_'+rq.upper());actual=sha(path)
   if expected:assert actual==expected,(ds,rq,'score cache changed')
   audit[str(path.relative_to(ROOT))]=actual
   groups=defaultdict(dict)
   for r in records(path):
    if not r.get('probe_available',False) or not r.get('mapping_available',True):continue
    if ds=='TRACE' and rq=='rq2' and r.get('scan_status')!='COMPLETE':continue
    y=labels.index(r['reference_diagnosis']);probe=r['probe'];frac=r.get('fraction','natural')
    for population in ['all_cached_available']+(['both_full_correct'] if r.get('both_full_correct') else []):
     for view,field,key in [('full','full_probabilities',r['patient_id']),('cited','cited_probabilities',r['explanation_id'])]:
      values=r.get(field)
      if values is None:continue
      g=(ds,rq,probe,str(frac) if view=='cited' else 'full',view,population)
      previous=groups[g].get(key)
      if previous is not None:assert previous[1]==y and np.allclose(previous[0],values,atol=1e-7)
      groups[g][key]=(values,y)
   for g,rr in groups.items():
    vals=list(rr.values());stats,bs=calibration([x[0] for x in vals],[x[1] for x in vals]);meta=dict(zip(['dataset','analysis','probe','budget','view','population'],g))
    cal.append(dict(meta,**stats,temperature=1.,bins=10,interpretation='uncalibrated cached-population diagnostic; conditional sampling; no development fit'))
    binrows.extend(dict(meta,**b) for b in bs)
   print('CALIBRATION original',ds,rq,len(groups),'panels',flush=True)
  modelroot=ROOT/cfg['datasets'][ds]['models']
  for f in modelroot.rglob('*'):
   if not f.is_file() or f.suffix not in ['.json','.jsonl','.npz'] or not any(x in f.name.lower() for x in ['dev','valid','pred','logit']):continue
   inventory.append(dict(dataset=ds,path=f.relative_to(ROOT).as_posix(),bytes=f.stat().st_size))
 workers=min(a.workers,len(tasks),os.cpu_count() or 1);print('BOOTSTRAP',len(tasks),'tasks;',workers,'processes',flush=True);out=[]
 with ProcessPoolExecutor(max_workers=workers) as pool:
  pending={pool.submit(matched_job,t) for t in tasks};n=0
  while pending:
   done,pending=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
   for f in done:out+=f.result();n+=1
   if not done or n%5==0 or not pending:print('PROGRESS',n,'/',len(tasks),flush=True)
 table(a.out/'fact_count_matched_comparisons.csv',sorted(out,key=lambda r:json.dumps(r,sort_keys=True)))
 table(a.out/'uncalibrated_cache_metrics.csv',cal);table(a.out/'calibration_bins.csv',binrows);dump(a.out/'development_cache_inventory.json',inventory);dump(a.out/'input_audit.json',audit)
 return dict(status='complete_readonly_diagnostics',matched_tasks=len(tasks),calibration_panels=len(cal),temperature_fit=False,reason='Development prediction availability must be established before temperature fitting; no test-data fitting.',new_predictions=False,model_training=False,api_calls=0)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=64);p.add_argument('--out',type=Path,default=ROOT/'artifacts/round1_budget_calibration_v1');a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);state={};start=time.time()
 try:state=main(a)
 except Exception:state={'status':'failed','error':traceback.format_exc()};raise
 finally:
  state['seconds']=time.time()-start;dump(a.out/'run_manifest.json',state)
  with tarfile.open(str(a.out)+'_reports.tar.gz','w:gz') as z:
   for f in a.out.iterdir():
    if f.is_file():z.add(f,arcname=f.name)
  print('Return report:',str(a.out)+'_reports.tar.gz',flush=True)
