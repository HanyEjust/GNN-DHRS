# Implementation decisions for details not uniquely specified in the manuscript

The manuscript is the normative source. The following low-level choices are fixed here so that the public implementation is deterministic and auditable rather than underspecified.

- Borderline-SMOTE: `kind=borderline-1`, `k_neighbors=5`, `m_neighbors=10`.
- SMOTEENN: SMOTE `k_neighbors=5`; ENN `n_neighbors=3`, `sampling_strategy=all`.
- One-hot synthetic values are projected blockwise by argmax after each resampling stage.
- ECE uses 10 equal-width probability bins.
- Kernel SHAP uses `nsamples=256` unless changed in YAML.
- FT-Transformer reference implementation uses token dimension 64, 3 Transformer layers, 8 heads, FFN multiplier 4, dropout 0.20, AdamW 1e-3, weight decay 5e-4, and early stopping patience 20.
- Conventional baseline defaults are explicitly encoded in `src/gnn_dhrs/baselines.py`.
- Cross-dataset transfer uses only raw predictor columns common to source and target; all transformations are fit on source data.
- Quantitative figures are generated from experiment CSV files; manuscript values are never hard-coded.

If an original experiment log establishes a different low-level value, update this document, `configs/paper.yaml`, and the corresponding implementation together.
