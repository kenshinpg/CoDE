"""Shared patient/evidence contract. No diagnosis labels in public records."""
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

VERSION = 'lantern_neutral50k_v1'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(p):
    with open(p,encoding='utf-8-sig') as f:
        for l in f:
            if l.strip():yield json.loads(l)
def write(p,rs):
    with open(p,'w',encoding='utf-8',newline='\n') as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def dictionary(raw):
    result={}
    for key,m in raw.items():
        code=m.get('name',key)
        if code in result or m['data_type'] not in ['B','C','M']:raise ValueError('Invalid dictionary '+code)
        meanings={}
        for v in m.get('possible-values',[]):
            val=str(v);meaning=m.get('value_meaning',{}).get(val)
            if isinstance(meaning,dict):meaning=meaning.get('en')
            if meaning is None:
                try:float(val);meaning=val
                except ValueError:raise ValueError('Untranslated value '+code+'/'+val)
            meanings[val]=str(meaning)
        result[code]=dict(code=code,type=m['data_type'],question=m['question_en'],
            values=meanings,default=m.get('default_value'),code_question=m.get('code_question',code),
            is_antecedent=m.get('is_antecedent'))
    # Undirected association components, NOT proof of causal parent/child order.
    links=defaultdict(set)
    for code,m in result.items():
        q=m['code_question'];links[code].add(q);links[q].add(code)
    groups={};seen=set()
    for node in sorted(links):
        if node in seen:continue
        stack=[node];component=set()
        while stack:
            n=stack.pop()
            if n in component:continue
            component.add(n);stack.extend(links[n]-component)
        seen|=component;gid='group_'+digest(sorted(component))[:16]
        for n in component:
            if n in result:result[n]['group_id']=gid
        groups[gid]=dict(codes=sorted(component & result.keys()),unresolved_nodes=sorted(component-result.keys()))
    result.update({'DEM_AGE':dict(code='DEM_AGE',type='N',question='Age in years?',values={},default=None,code_question='DEM_AGE',is_antecedent=False,group_id='demographics_age'), 'DEM_SEX':dict(code='DEM_SEX',type='C',question='Recorded sex?',values={'M':'male','F':'female'},default=None,code_question='DEM_SEX',is_antecedent=False,group_id='demographics_sex')})
    for g in groups.values():g['edit_eligibility']='REQUIRES_GROUP_REVIEW' if not g['unresolved_nodes'] else 'BLOCKED_UNRESOLVED_ASSOCIATION'
    for code in ['DEM_AGE','DEM_SEX']:
        groups[result[code]['group_id']]=dict(codes=[code],unresolved_nodes=[],edit_eligibility='EXCLUDED_FROM_SYMPTOM_EDITS')
    return dict(version=VERSION,features=result,groups=groups,
        dependency_semantics='code_question association components; directional clinical dependencies not inferred')
def parse_tokens(raw):
    ts=ast.literal_eval(raw) if isinstance(raw,str) else raw
    if not isinstance(ts,list) or not all(isinstance(t,str) for t in ts):raise ValueError('Bad tokens')
    return sorted(set(ts))
def facts(tokens,contract):
    by=defaultdict(list);meta=contract['features']
    for token in tokens:
        code,sep,value=token.partition('_@_')
        if code not in meta:raise ValueError('Unknown feature '+code)
        m=meta[code]
        if (m['type']=='B' and sep) or (m['type']!='B' and (not sep or value not in m['values'])):raise ValueError('Invalid value '+token)
        by[code].append('true' if m['type']=='B' else value)
    out=[];issues=[]
    for code,vs in sorted(by.items()):
        m=meta[code];vs=sorted(set(vs))
        if m['type']=='C' and len(vs)!=1:raise ValueError('Conflicting categorical values')
        if m['type']=='M' and len(vs)>1 and str(m['default']) in vs:issues.append('default_with_other_values:'+code)
        for v in vs:
            out.append(dict(fact_id=code+':'+v,feature_id=code,value=v,
                question=m['question'],value_text='Yes' if m['type']=='B' else m['values'][v],
                group_id=m['group_id'],observation_status='recorded',
                default_value_recorded=m['type']!='B' and v==str(m['default']),
                provenance={'source_field':'EVIDENCES','source_token':code if m['type']=='B' else code+'_@_'+v},
                polarity='affirmed' if m['type']=='B' else 'encoded_value',
                is_antecedent=m['is_antecedent']))
    return out,issues
def public_patient(tokens,contract,age=None,sex=None,view='symptoms_only'):
    fs,issues=facts(tokens,contract);pattern=digest(tokens)
    demographics=dict(age=age,sex=sex)
    pid=digest([pattern,demographics]) if age is not None or sex is not None else pattern
    text='\n'.join(f"{f['question']} Answer: {f['value_text']}" for f in fs)
    for f in fs:f.update(visible=True,coverage_scope='symptoms_antecedents')
    for code,value in [('DEM_AGE',age),('DEM_SEX',sex)]:
        if value is not None and str(value).strip():
            val=str(value)
            if code=='DEM_AGE':
                import math
                n=float(value)
                if not math.isfinite(n) or n<0:raise ValueError('Invalid recorded age')
                val=str(int(n)) if n.is_integer() else str(n)
            elif val not in contract['features'][code]['values']:raise ValueError('Invalid recorded sex')
            fs.append(dict(fact_id=code+':'+val,feature_id=code,value=val,question=contract['features'][code]['question'],value_text=val,group_id=contract['features'][code]['group_id'],observation_status='recorded',polarity='affirmed',is_antecedent=False,default_value_recorded=False,visible=view=='full',coverage_scope='demographics',provenance={'source_field':'AGE' if code=='DEM_AGE' else 'SEX','source_value':value}))
    if view=='full':text=f'Age: {age} years. Sex: {sex}.\n'+text
    return dict(patient_id=pid,pattern_id=pattern,input_view=view,demographics=demographics,facts=fs,semantic_issues=issues,
        input_view_id=digest([pid,view]),visible_input_sha256=digest(text),
        input_text=text,
        missing_policy='Unlisted findings are unknown; do not infer a negative finding.')

def align(claim,patient,contract):
    """Conservative value alignment; semantic mapper correctness is not certified."""
    code=claim['feature_id'];v=claim['value'];pol=claim['polarity']
    base=dict(claim,matched_fact_ids=[],status='UNCERTAIN',modifier_validation='NOT_VERIFIED',polarity_semantics='polarity applies to proposition feature=value, not to surface negative words')
    if code is None:return dict(base,status='UNMAPPED')
    m=contract['features'][code]
    if v is None or pol=='uncertain':return base
    allowed={'true','false'} if m['type']=='B' else set(m['values'])
    if m['type']=='N':
        try:
            import math
            if not math.isfinite(float(v)) or float(v)<0:raise ValueError('Invalid numeric age')
        except (TypeError,ValueError):raise ValueError('Invalid numeric age')
        allowed={v}
    if v not in allowed:raise ValueError('Illegal extracted value '+code+'/'+v)
    observed=[f for f in patient['facts'] if f['feature_id']==code and f.get('visible',True)]
    if not observed:return dict(base,status='NOT_IN_RECORD')
    if m['type']=='B':
        if pol=='negated':v='false' if v=='true' else 'true'
        matches=[f for f in observed if f['value']==v]
        return dict(base,status='MATCHED' if matches else 'CONTRADICTS_RECORD',matched_fact_ids=[f['fact_id'] for f in matches])
    matches=[f for f in observed if (float(f['value'])==float(v) if m['type']=='N' else f['value']==v)]
    if pol=='negated':return dict(base,status='CONTRADICTS_RECORD' if matches else 'UNCERTAIN')
    if matches:return dict(base,status='MATCHED',matched_fact_ids=[f['fact_id'] for f in matches])
    return dict(base,status='CONTRADICTS_RECORD' if m['type'] in ['C','N'] else 'NOT_IN_RECORD')
