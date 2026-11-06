"""Strict Pydantic output checks plus dynamic dictionary validation in caller."""
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
class Explanation(Strict):
    ranked_diagnoses:list[str]=Field(min_length=3,max_length=3)
    explanation:str=Field(min_length=1)
class Claim(Strict):
    quote:str=Field(min_length=1)
    feature_id:str|None
    value:str|None
    polarity:Literal['affirmed','negated','uncertain']
    interpretation:str
    relation:Literal['supports','opposes','describes','uncertain']
    diagnosis:str|None
class Extraction(Strict):
    claims:list[Claim]
