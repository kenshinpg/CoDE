"""Export patient-free prompts and actual strict schemas; performs no API call."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/CoDE/scripts'))
from trace_generator_rq2 import base,segment,original_generation

def main():
    data=dict(generation=dict(system=original_generation.SYSTEM,schema=original_generation.Answer.model_json_schema()),
              patient_fact_matching=dict(system=base.PROMPT,schema=base.Answer.model_json_schema()),
              span_fallback=dict(system=segment.PROMPT,schema=segment.Answer.model_json_schema()))
    p=ROOT/'build/patient_free_contracts.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Exported generation, matching and fallback contracts to',p)
if __name__=='__main__':main()
