"""Dictionary-derived feature/value contracts; no feature-specific rules."""
from typing import Literal,Union
from functools import lru_cache
import json
from pydantic import ConfigDict,Field,create_model,model_validator,BaseModel

class ClaimBase(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    @model_validator(mode='after')
    def resolution_consistent(self):
        if (self.value_resolution=='exact') != (self.value is not None):
            raise ValueError('Only exact claims carry a coded value')
        return self

def extraction_dictionary(contract,codes):
    out={}
    for c in codes:
        m=contract['features'][c];vals=({'true':'present','false':'absent'} if m['type']=='B' else m['values'])
        numeric=m['type']=='N'
        if vals and m['type']!='B':
            try:
                for v in vals:float(v)
                numeric=True
            except ValueError:pass
        out[c]=dict(question=m['question'],type=m['type'],values=vals,numeric_values=numeric)
    return out

def feature_chunks(contract,limit):
    # Preserve associated symptoms/attributes where they fit; oversized groups split explicitly.
    groups={}
    for c,m in sorted(contract['features'].items()):groups.setdefault(m['group_id'],[]).append(c)
    chunks=[];current=[]
    for group in groups.values():
        for start in range(0,len(group),limit):
            part=group[start:start+limit]
            if current and len(current)+len(part)>limit:chunks.append(current);current=[]
            current+=part
    if current:chunks.append(current)
    return chunks

def extraction_model(dictionary):return _model(json.dumps(dictionary,sort_keys=True))

@lru_cache(maxsize=256)
def _model(serialized):
    dictionary=json.loads(serialized);variants=[]
    for code,m in dictionary.items():
        values=list(m['values'])
        value_type=Literal[tuple(values)] if values else str
        fields=dict(feature_id=(Literal[code],...),value=(value_type,...),
            quote=(str,Field(min_length=1)),surface_value=(str,...),
            value_resolution=(Literal['exact','qualitative','unresolved'],...),
            polarity=(Literal['affirmed','negated','uncertain'],...),
            interpretation=(str,Field(max_length=240)),
            relation=(Literal['supports','opposes','describes','uncertain'],...),diagnosis=(str|None,...))
        for resolution in ['exact','qualitative','unresolved']:
            branch=dict(fields)
            branch['value_resolution']=(Literal[resolution],...)
            branch['value']=(value_type if resolution=='exact' else type(None),...)
            variants.append(create_model('Claim_'+code+'_'+resolution,__base__=ClaimBase,**branch))
    union=variants[0] if len(variants)==1 else Union[tuple(variants)]
    return create_model('Extraction',__config__=ConfigDict(extra='forbid',strict=True),claims=(list[union],...))
