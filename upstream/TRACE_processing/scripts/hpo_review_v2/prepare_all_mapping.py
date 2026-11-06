"""Freeze all extracted admissions without resampling or reference labels."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rag_hpo_native_fix_v1'))
import retrieval_core as core

p=argparse.ArgumentParser()
p.add_argument('--entity-dir',type=Path,required=True)
p.add_argument('--obo',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.patients=0
core.prepare(a)
