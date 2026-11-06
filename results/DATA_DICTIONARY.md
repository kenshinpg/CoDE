# Result formats

`aggregate/README.md` and subdirectory READMEs document comparison matrices. `figure_inputs/accepted_panels.json` stores full-precision selection gains, natural AP differences and controlled concordance panels. `figure1_expected.json` is the final-proof numerical reference for its 34 plotted values.

Selection gains are accuracy differences. Raw panel values are proportions; figures multiply them by 100 to show percentage points. Budget fractions are 0.25/0.5/0.75. Natural AP differences stay in AP units unless a column explicitly specifies multiplication by 100. Controlled concordance differences are displayed in percentage points. Confidence intervals preserve the original analysis method and cluster unit.

Internal method keys: `cohd`/`cohd_coverage` = Association; `encoder` = Representation; `grasp` = Group-enhanced; `lightgbm` = Tree; `tree_local_abs_shap` = local absolute SHAP for selection; `tree_positive_shap` = local positive-SHAP coverage; `tree_class_abs_shap` = class SHAP; `tabr` = TabR-S; `no_retrieval` = matched no-retrieval source. A is direct support and R is full-record-referenced support.

`ddx_intervention_effects.json` maps selector|direction|operation|proposal-stratum|probe to sorted anonymous patient-average effect values, measured in predicted-probability percentage points relative to matched random edits. No patient linkage is retained. These arrays derive exclusively from synthetic DDXPlus and reproduce the accepted distribution summaries. Positive values are model-support effects.

`paper_tables/*.csv` supplies numbered cell columns (`cell_1`, ...) in original row order. Multi-panel headings, footnotes and mathematical cell content retain LaTeX notation. Captions and labels are in `docs/table_index.json`; these cell matrices are display exports, while the aggregate statistical CSVs provide structured comparison fields.

Empty cells and explicit NA/unavailable values retain their meaning. Random and failure-targeted audit strata remain separate. Self-evaluation markers, conditional populations and simultaneous versus marginal intervals follow the associated table captions.
