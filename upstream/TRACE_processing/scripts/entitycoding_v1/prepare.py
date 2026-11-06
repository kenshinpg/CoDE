"""Prepare an engineering pilot. No discharge text or reference labels in model input."""
import argparse, collections, hashlib, json, pickle
from pathlib import Path
import numpy as np
from common import sha,write

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('scalar','_reconstruct'):
            core=getattr(np,'_core',None)
            if core is None: core=np.core
            return getattr(core.multiarray,name)
        if module=='numpy' and name in ('dtype','ndarray'):return getattr(np,name)
        if (module,name)==('collections','defaultdict'):return collections.defaultdict
        if module=='builtins' and name in ('set','dict','list','tuple','int','str','float'):
            return getattr(__import__('builtins'),name)
        raise pickle.UnpicklingError(f'Unsupported global: {module}.{name}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--per-disease',type=int,default=8);p.add_argument('--seed',type=int,default=261010);a=p.parse_args()
    if a.per_disease<0:raise ValueError('per-disease must be nonnegative; 0 means all')
    files=sorted(a.data.glob('*_hadm_info_first_diag.pkl'))
    if len(files)!=4:raise ValueError(f'Expected four disease pickle files, found {len(files)}')
    a.out.mkdir(parents=True,exist_ok=False)
    docs=[];private=[];counts={};seen=set()
    for f in files:
        with f.open('rb') as h: data=RestrictedUnpickler(h).load()
        ordered=sorted(data,key=lambda k:hashlib.sha256(f'{a.seed}:{f.name}:{k}'.encode()).hexdigest())
        label=f.name.split('_hadm')[0]; counts[label]=min(a.per_disease,len(ordered)) if a.per_disease else len(ordered)
        for key in ordered[:counts[label]]:
            pid='cdm_'+str(key)
            if pid in seen:raise ValueError('Repeated admission across source files')
            seen.add(pid);record=data[key]
            private.append({'patient_id':pid,'reference_disease':label,'source_file':f.name,'source_key':str(key)})
            for field in ('Patient History','Physical Examination'):
                text=record.get(field)
                if not isinstance(text,str) or not text.strip():raise ValueError(f'Missing text: {pid}/{field}')
                docs.append({'document_id':pid+':'+field,'patient_id':pid,'source_field':field,'text':text})
    for name,records in [('documents.jsonl',docs),('selection_PRIVATE.jsonl',private)]:
        with (a.out/name).open('w',encoding='utf-8') as h:
            for r in records:h.write(json.dumps(r,ensure_ascii=False)+'\n')
    report={'patients':len(private),'documents':len(docs),'per_disease':counts,'seed':a.seed,
        'purpose':'Extraction coverage audit; not an independent diagnostic test or clinical validation',
        'selection':'all source cases' if a.per_disease==0 else 'hash-ordered per-disease sample',
        'model_fields':['Patient History','Physical Examination'],'inputs':{f.name:sha(f) for f in files},
        'documents_sha256':sha(a.out/'documents.jsonl'),'code_sha256':sha(Path(__file__))}
    write(a.out/'manifest.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
