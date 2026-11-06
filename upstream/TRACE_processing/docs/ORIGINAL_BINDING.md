# Historical production bindings and the public adaptation

Historical generation used trace_rollouts_full_v1/pipeline.py, source SHA-256
09045edc5d144ec0f944790214c9eef9701179ff5c53736679a2200b3b17244a.
Final full matcher source hashes were:

- pipeline.py: c9e2d1c4caaf5ae90736db3832b5b320c6606897aa6142e3436e9c9eda7de9e7
- base.py: 7f4bff206e405396e27eb2bf8a0447a4402b59b7bc84f6c3e670170a4800eefe
- segment.py: 3031e14936fb4511d1ac602a3dc597be39f71249aa5153a0a28b1fb507c07fa5

The public files are not falsely labelled byte-identical historical executables:
paths and endpoint defaults are portable, and unused development CLI functions
were removed from matching helper modules. Prompts, Pydantic schemas, validators,
fact grouping and bounded fallback logic are retained. The copied upstream NER
and native-tokenizer retrieval files were checked against existing identity manifests.

Original generation: temperature .7, top_p .8, output 2048, thinking=false, strict
JSON schema, base seed 261013, four requests/admission. Per invocation it tries at
most three calls; transport failures count. Ordinary full launcher uses workers=64
and does not pass retry-failed. Explicit retry-failed is a distinct recorded invocation
and changes the seed using its invocation identity. There is no truncation repair,
score-conditioned generation or implicit manual replacement. All terminal failures stay.

Original final matching: eight facts/group; free output 3072 and segment 2048;
at most two attempts/mode; regular temperature=0, whitespace retry or second segment
attempt=.2; thinking=false, strict schema and local Pydantic plus verbatim span checks.
Free length truncation without whitespace stall falls back rather than identical retry.
Whole-text matching uses every visible F, not NER/HPO/meaning-selected candidates.
Original matching timeout is 150 seconds, tokenization timeout 30 seconds; transport
exceptions are caught in the attempt loop and may enter segment fallback. Three
serial requests precede workers=128, with the retained failure circuit. Existing
terminal group results are not silently retried. Original free/segment API requests
and response validation logic are retained in the public driver.

Generated-but-incomplete matching is PARTIAL in the raw scan ledger; unavailable
generation is UNAVAILABLE; complete mapping with ambiguous decisions is also PARTIAL.
Do not treat a partial exact set as an exhaustive E. CoDE's adapter and complete-only
analysis gate determine eligibility. The historical final run completed all 85,813
groups; 9,171 explanation scans were COMPLETE, 376 PARTIAL and 53 UNAVAILABLE.

The separate CoDE second-generator release implements generation/matching in a new
driver. It may use a 240-second timeout and infrastructure stop/resume separately
from returned-response validation budgets. Those execution differences require their
own records; this supplement does not overwrite that code or claim token-equivalent
responses. It adds the actual original entry points missing from the main release.

Public prepare_match_source.py only supplies the validated final input envelope
from frozen facts and original parsed generation text. It omits intermediate mention
inventories that did not feed E. Its inputs/outputs are private runtime artifacts.
For already-produced history, prefer the authorized historical release and manifest.

The collaborator-provided derivative is the received starting point; neither these
bindings nor its filenames certify its one-to-one identity with official MIMIC-CDM.
Original main F and later 85-action source sensitivity remain distinct.
