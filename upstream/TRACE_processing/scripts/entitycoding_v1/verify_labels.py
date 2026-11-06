"""Read only the trusted upstream model's own label encoder; verify alias mapping."""
import argparse,json
from pathlib import Path
import joblib
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
classes=[str(x) for x in joblib.load(a.repo/'data/models/ac_model/label_encoder.joblib').classes_]
expected=['absent','associated_with_someone_else','hypothetical','possible','present']
print(json.dumps({'checkpoint_assertion_classes':classes,'expected':expected,'match':classes==expected},indent=2),flush=True)
if classes!=expected:raise ValueError('Checkpoint label encoder differs from upstream inference mapping; stop and inspect')
