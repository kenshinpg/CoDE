# Ownership and third-party resources

Team-authored TRACE adapters, clients, validators and release helpers are licensed
under the MIT License in this directory and the CoDE repository root. Third-party
code, model resources and clinical data retain their separate terms below.

EntityCoding executable dependencies under vendor/entity-coding/ are copied from
jd4501/entity-coding (recorded code revision fdf610ca1bee62be7b75d499c1d34eecfc4dada2).
Keep vendor/entity-coding/LICENSE (MIT, James Douglas). Model weights have additional
upstream restrictions and are not bundled. No MDACE dataset or other upstream dataset is copied.

The actual HPO mapping prompt is the team's adaptation of the retrieval-then-contextual
selection strategy and RAG-HPO prompt. The upstream JSON is obtained separately;
this package does not redistribute that repository, notebook or upstream model weights.
Use https://github.com/PoseyPod/RAG-HPO and retain its applicable notices when adding
upstream material. HPO ontology terms, SapBERT/Qwen checkpoints and clinical source
data each have independent access/use terms; no model license is assigned to the data.

Method references and implementation sources:
- EntityCoding: https://aclanthology.org/2025.acl-long.1489/ ; https://github.com/jd4501/entity-coding
- HPO: https://github.com/obophenotype/human-phenotype-ontology
- SapBERT: https://aclanthology.org/2021.naacl-main.334/ ; https://huggingface.co/pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb
- RAG-HPO: https://doi.org/10.1186/s13073-025-01521-w ; https://github.com/PoseyPod/RAG-HPO
- Reference-interval idea (not a LOINC mapping reproduction): https://pmc.ncbi.nlm.nih.gov/articles/PMC6527418/
- Received-source upstream study/resource: https://doi.org/10.1038/s41591-024-03097-1 ; https://physionet.org/content/mimic-iv-ext-cdm/1.0/
- 27B download source: https://www.modelscope.cn/models/Qwen/Qwen3.8-27B

No clinical expertise, semantic accuracy certification, prospective availability or
blanket permission to redistribute clinical source/derived records is implied.
