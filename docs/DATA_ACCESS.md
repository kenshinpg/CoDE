# Data access and release contents

| Resource | Included | Access route |
| --- | --- | --- |
| Accepted aggregate statistics and table cells | Yes | `results/` |
| DDXPlus small synthetic example | Yes | `examples/`; original dataset CC BY 4.0 |
| Full DDXPlus study input and generated explanations | No | Obtain the original DDXPlus release and the study's frozen selection/mapping inputs described in INPUT_CONTRACTS |
| TRACE clinical records, generated explanations, exact citations and per-admission predictions | No | Obtain personal project access through PhysioNet; derivative sharing requires the applicable DUA/authorized route |
| Fitted checkpoints and clinical embeddings | No | Refit from authorized inputs; no learned patient-level artifacts are redistributed here |
| Audit workbooks and linked reviewer decisions | No | Human review code, aggregate quality and sensitivity results are included |

DDXPlus: Fansi Tchango et al., *DDXPlus: A New Dataset for Automatic Medical Diagnosis* (2022), https://github.com/mila-iqia/ddxplus; English release https://doi.org/10.6084/m9.figshare.22687585.

Clinical source: Hager, Jungmann and Rueckert (2024), *MIMIC-IV-Ext Clinical Decision Making*, version 1.0, https://doi.org/10.13026/2pfq-5b68; access page https://physionet.org/content/mimic-iv-ext-cdm/1.0/. Each user follows the project's credentialing, training and DUA requirements. Parent MIMIC access alone should not be represented as project-specific access.

The study processed a collaborator-supplied four-file clinical derivative. Exact byte-level equivalence with an official release and the reported radiology-count discrepancy were not independently established. Downloading a current official dataset is therefore insufficient, by itself, to assert reproduction of the frozen TRACE inputs. The processing specification and accepted aggregate results describe the study actually performed.

Clinical-source fields that could identify admissions, source spans, clinical narratives, endpoint logs, service inventories, reviewer identities, raw workbooks, credentials and local machine paths are omitted. Source code uses symbolic patient/record fields and expected restricted filenames; these are interfaces rather than supplied patient records.
