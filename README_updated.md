# GNN-DHRS

Official reproducibility implementation for **“An Explainable Graph Attention Framework with Dual Hybrid Resampling for Imbalanced Brain Stroke Prediction.”**

GNN-DHRS is an explainable graph-attention framework for imbalanced stroke prediction. The repository implements leakage-safe preprocessing, the Dual Hybrid Resampling Strategy (DHRS), patient-similarity graph construction, similarity-aware multi-head graph attention, weighted optimization, nested cross-validation, strictly inductive test inference, conventional and graph-learning baselines, FT-Transformer comparisons, cross-dataset evaluation, ablation studies, calibration, selective prediction, robustness analysis, statistical validation, computational profiling, scalability analysis, and feature- and graph-level explainability.
![System architecture]<img width="1570" height="1002" alt="gnn_dhrs_framework_" src="https://github.com/user-attachments/assets/2ed09253-4e6b-447f-9ef8-83fd69c004c4" />

The repository is designed to **calculate experimental results from the supplied datasets**. Manuscript performance values are not hard-coded into the implementation.

> **Reproducibility principle:** if a fresh execution differs from a manuscript table or figure, the discrepancy should be investigated and reported. Repository code must not be modified merely to force agreement with previously reported numerical values.

---

## 1. Framework overview

The principal workflow is:

```text
Raw stroke benchmark
        |
        v
Outer stratified cross-validation
        |
        +------------------------------+
        |                              |
        v                              |
Outer-training partition               |
        |                              |
        v                              |
Training-only preprocessing            |
        |
        v
Training-only feature selection
        |
        v
DHRS
Borderline-SMOTE -> SMOTEENN
        |
        v
Patient-similarity graph
        |
        v
Similarity-aware multi-head GAT
        |
        v
Inner validation / model selection
        |
        v
Validation-derived decision threshold
        |
        +------------------------------+
        |
        v
Inductive attachment of each
untouched outer-test patient
        |
        v
Stroke probability
        |
        v
Performance, calibration,
explainability, robustness,
ablation and statistical analyses
```

All preprocessing, feature-selection, resampling, model-selection, and threshold-selection operations are restricted to the appropriate training partitions. Outer-test labels are used only for final scoring.

---

## 2. Benchmark datasets

The manuscript evaluates three public stroke benchmark datasets, referred to as **DF-1**, **DF-2**, and **DF-3**.

| Dataset | Samples | Stroke | Non-stroke | Repository filename |
|---|---:|---:|---:|---|
| DF-1 | 43,400 | 783 | 42,617 | `data/raw/DF-1.csv` |
| DF-2 | 4,981 | 248 | 4,733 | `data/raw/DF-2.csv` |
| DF-3 | 5,110 | 249 | 4,861 | `data/raw/DF-3.csv` |

The original patient-level datasets are **not redistributed** in this repository. Obtain them from the public sources cited in the manuscript and place them under:

```text
data/raw/
├── DF-1.csv
├── DF-2.csv
└── DF-3.csv
```

The expected outcome column is:

```text
stroke
```

Identifier columns such as `id`, `ID`, or `patient_id` are excluded when present.

Public sources currently documented for DF-2 and DF-3 are:

- DF-2: `https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset`
- DF-3: `https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset`

The exact DF-1 source should be verified against the dataset citation in the submitted manuscript before final archival release.

---

## 3. Reproducibility protocol

The principal configuration is stored in `configs/paper.yaml`.

### Cross-validation

- 10 outer stratified folds for performance estimation.
- 5 inner stratified folds for model development.
- Controlled stochastic seeds: `11, 22, 33, 44, 55`.
- Outer-test observations remain untouched during preprocessing, feature selection, DHRS, model selection, threshold selection, and training.

### Preprocessing

- Numeric missing values are imputed using training-partition statistics.
- Categorical missing values are represented using an `Unknown` category.
- Categorical variables are one-hot encoded with unseen-category handling.
- Numeric variables are standardized using training-derived statistics only.
- Highly correlated predictors are filtered using `|r| > 0.90`.
- Training-only ANOVA F ranking is used for candidate retained feature counts of 5, 10, 15, and all available predictors.

### Dual Hybrid Resampling Strategy

DHRS is applied only to training observations and consists of:

```text
Borderline-SMOTE
        |
        v
SMOTEENN
        |
        v
valid categorical-state projection
```

No validation or outer-test observation is used to generate synthetic training patients.

### Patient-similarity graph

The principal graph configuration uses:

- cosine similarity;
- directed top-k neighborhood construction;
- `k = 10`;
- positive similarities only;
- no forced graph symmetrization;
- self-loops introduced by the GAT with weight 1.

Test patients are evaluated inductively. Each test patient is attached to original outer-training reference patients without creating test-to-test edges.

### GNN-DHRS architecture

| Parameter | Principal configuration |
|---|---:|
| Graph layers | 2 |
| Hidden dimension | 128 |
| Attention heads | 8 |
| Dimension per head | 16 |
| Dropout | 0.30 |
| LeakyReLU negative slope | 0.20 |
| Graph neighbors | 10 |
| Loss | Weighted binary cross-entropy |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0005 |
| Maximum epochs | 200 |
| Early-stopping patience | 20 |
| LR scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.50 |
| Scheduler patience | 10 |
| Minimum learning rate | `1e-6` |

Similarity-aware attention implements the graph-similarity contribution directly in the attention normalization.

### Decision threshold

Candidate thresholds are:

```text
0.10  0.20  0.30  0.40  0.50  0.60  0.70  0.80  0.90
```

The principal implementation selects the threshold using training-derived inner-validation predictions, with F1 as the selection objective.

---

## 4. Evaluation measures

The implementation supports:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- Negative Predictive Value
- F1-score
- Balanced Accuracy
- Matthews Correlation Coefficient
- Cohen's Kappa
- ROC-AUC
- PR-AUC
- Brier Score
- Expected Calibration Error

Because the stroke benchmarks are highly imbalanced, particular attention is given to recall, F1, PR-AUC, MCC, and balanced accuracy rather than relying on accuracy alone.

---

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/HanyEjust/GNN-DHRS.git
cd GNN-DHRS
```

Create an isolated Python environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU execution, install PyTorch and PyTorch Geometric builds compatible with the local CUDA runtime when the generic installation is not appropriate.

---

## 6. Software environment

The manuscript reports the following experimental environment:

| Software / hardware | Version or configuration |
|---|---|
| Python | 3.11 |
| CUDA | 12.1 |
| PyTorch | 2.1.0 |
| PyTorch Geometric | 2.4.0 |
| scikit-learn | 1.3.2 |
| NumPy | 1.26.2 |
| Pandas | 2.1.3 |
| imbalanced-learn | 0.11.0 |
| XGBoost | 2.0.3 |
| LightGBM | 4.1.0 |
| CatBoost | 1.2.2 |
| SHAP | 0.44.0 |
| Execution environment | Kaggle Notebook |
| GPU | NVIDIA Tesla T4 |
| GPU memory | 16 GB |
| Approx. system RAM | 30 GB |

---

## 7. Experiment suite

The repository contains dedicated entry points for the manuscript analyses.

| Analysis | Script | Primary output |
|---|---|---|
| Dataset integrity and class distribution | `scripts/run_dataset_audit.py` | `outputs/tables/dataset_audit.csv` |
| Main leakage-safe nested CV | `scripts/run_nested_cv.py` | `outputs/tables/nested_cv_fold_metrics.csv` |
| Conventional/tabular baselines | `scripts/run_baselines.py` | `outputs/tables/tabular_baselines.csv` |
| GCN/GraphSAGE/GAT baselines | `scripts/run_graph_baselines.py` | `outputs/tables/graph_baselines.csv` |
| FT-Transformer comparison | `scripts/run_ft_transformer.py` | `outputs/tables/ft_transformer_baselines.csv` |
| Cross-dataset transfer | `scripts/run_cross_dataset.py` | `outputs/tables/cross_dataset.csv` |
| Resampling/order ablation | `scripts/run_resampling_ablation.py` | `outputs/tables/resampling_ablation.csv` |
| DHRS stage diagnostics | `scripts/run_dhrs_diagnostics.py` | `outputs/tables/dhrs_stage_distribution.csv` |
| Categorical resampling sensitivity | `scripts/run_categorical_resampling.py` | `outputs/tables/categorical_resampling_sensitivity.csv` |
| BCE vs weighted BCE | `scripts/run_loss_ablation.py` | `outputs/tables/class_weight_ablation.csv` |
| Synthetic-edge provenance audit | `scripts/run_synthetic_edge_audit.py` | `outputs/tables/synthetic_edge_audit.csv` |
| Synthetic-node message-passing ablation | `scripts/run_synthetic_node_ablation.py` | `outputs/tables/synthetic_node_ablation.csv` |
| Graph/attention/similarity ablation | `scripts/run_graph_component_ablation.py` | `outputs/tables/graph_component_ablation.csv` |
| Feature-count/topology analysis | `scripts/run_feature_topology.py` | `outputs/tables/feature_topology.csv` |
| Hyperparameter sensitivity | `scripts/run_sensitivity.py` | `outputs/tables/hyperparameter_sensitivity.csv` |
| Five-seed stability | `scripts/run_seed_stability.py` | `outputs/tables/random_seed_stability.csv` |
| Alternative graph definitions | `scripts/run_graph_definition_ablation.py` | `outputs/tables/graph_definition_*.csv` |
| Threshold sensitivity | `scripts/run_threshold_analysis.py` | `outputs/tables/threshold_sensitivity.csv` |
| Calibration | `scripts/run_calibration.py` | `outputs/tables/calibration_*.csv` |
| Selective prediction / risk coverage | `scripts/run_selective_prediction.py` | `outputs/tables/selective_prediction.csv` |
| Kernel SHAP, attention edge ablation and SHAP stability | `scripts/run_explainability.py` | explainability tables/figures |
| GNNExplainer / PGExplainer | `scripts/run_graph_explainers.py` and `src/gnn_dhrs/graph_explainers.py` | graph explanation outputs |
| Missingness/noise/prevalence/edge-corruption robustness | `scripts/run_robustness.py` | `outputs/tables/robustness.csv` |
| Paired statistical validation | `scripts/run_statistical_validation.py` | `outputs/tables/statistical_validation.csv` |
| Corrected Nadeau-Bengio statistics + Holm adjustment | `scripts/run_corrected_statistics.py` | `outputs/tables/corrected_statistical_validation.csv` |
| Computational efficiency and memory | `scripts/run_computational_efficiency.py` | `outputs/tables/computational_efficiency.csv` |
| Scalability | `scripts/run_scalability.py` | `outputs/tables/scalability.csv` |
| Manuscript figures/tables | `scripts/make_outputs.py` | `outputs/figures/`, `outputs/tables/` |

For a detailed manuscript-to-code mapping, see:

```text
docs/MANUSCRIPT_TO_CODE.md
```

---

## 8. Running the experiments

Individual experiments can be executed directly. For example:

```bash
python scripts/run_dataset_audit.py
python scripts/run_nested_cv.py
python scripts/run_baselines.py
python scripts/run_graph_baselines.py
python scripts/run_ft_transformer.py
python scripts/run_cross_dataset.py
python scripts/run_resampling_ablation.py
python scripts/run_dhrs_diagnostics.py
python scripts/run_categorical_resampling.py
python scripts/run_loss_ablation.py
python scripts/run_synthetic_edge_audit.py
python scripts/run_synthetic_node_ablation.py
python scripts/run_graph_component_ablation.py
python scripts/run_feature_topology.py
python scripts/run_sensitivity.py
python scripts/run_seed_stability.py
python scripts/run_graph_definition_ablation.py
python scripts/run_threshold_analysis.py
python scripts/run_calibration.py
python scripts/run_selective_prediction.py
python scripts/run_explainability.py
python scripts/run_graph_explainers.py
python scripts/run_robustness.py
python scripts/run_statistical_validation.py
python scripts/run_corrected_statistics.py
python scripts/run_computational_efficiency.py
python scripts/run_scalability.py
python scripts/make_outputs.py
```

To execute the manuscript experiment suite in dependency order:

```bash
python scripts/run_all.py
```

> **Note:** the complete suite is computationally expensive because it includes nested cross-validation, repeated seeds, baseline comparisons, ablations, explainability, robustness, and sensitivity analyses.

---

## 9. Baseline and comparative experiments

The repository includes conventional machine-learning and ensemble baselines, including implementations for models such as:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- MLP
- SVM
- XGBoost
- LightGBM
- CatBoost

Graph-learning comparisons are provided through dedicated graph-baseline utilities and scripts. FT-Transformer is included as a modern tabular deep-learning reference.

Cross-dataset experiments are separated from within-dataset nested-CV evaluation because they measure transfer under dataset shift rather than ordinary within-benchmark discrimination.

---

## 10. Ablation and sensitivity analyses

The repository includes controlled analyses of:

- alternative resampling strategies and ordering;
- DHRS stage-wise class distributions;
- categorical resampling handling;
- weighted vs unweighted loss;
- synthetic-node and synthetic-edge contributions;
- graph message passing;
- attention and similarity weighting;
- alternative graph definitions;
- retained feature count and graph topology;
- graph neighborhood size;
- attention-head count;
- hidden dimension;
- dropout;
- learning rate;
- weight decay;
- DHRS oversampling ratio;
- stochastic seed sensitivity.

These analyses are intended to isolate the contribution and stability of individual GNN-DHRS components.

---

## 11. Explainability

The repository provides both patient-feature and graph-relational explanation workflows.

### Kernel SHAP

Kernel SHAP is applied to selected patient-level input features while explaining the fitted GNN-DHRS prediction pipeline.

The configured protocol uses training-derived background observations only. Outer-test observations are excluded from background construction.

The explanation workflow supports:

- global mean absolute SHAP importance;
- local patient-level explanations;
- repeated explanation stability analysis;
- final graph-based stroke probability as the explained model output.

SHAP values characterize the behavior of the fitted model and must not be interpreted as causal clinical effects.

### Graph-relational explanations

The repository includes utilities and entry points for:

- learned attention extraction;
- high-importance relation analysis;
- controlled edge ablation;
- random-edge controls;
- fidelity/stability analysis;
- GNNExplainer;
- PGExplainer.

Graph explanations characterize learned patient-to-patient relational dependence; they do not establish clinically causal relationships between patients.

---

## 12. Calibration and selective prediction

Calibration analysis includes:

- Brier score;
- Expected Calibration Error;
- calibration-curve generation.

Selective prediction evaluates model behavior when lower-confidence predictions are withheld. The configured target coverages are:

```text
100%
95%
90%
80%
70%
```

The corresponding scripts operate on saved outer-test probabilities to preserve the evaluation protocol.

---

## 13. Robustness analysis

The robustness workflow evaluates perturbations affecting different aspects of the inference problem.

### Missingness

```text
0%
10%
20%
30%
```

### Gaussian feature noise

```text
sigma = 0.00
sigma = 0.05
sigma = 0.10
sigma = 0.20
```

### Prevalence shift

```text
0.5x
1.0x
2.0x
```

### Graph-edge corruption

```text
0%
10%
20%
30%
```

The implementation is organized so robustness evaluation is performed on fitted experimental models rather than by embedding expected manuscript results.

---

## 14. Statistical validation

The repository includes both conventional paired comparisons and corrected repeated-CV analysis.

The corrected analysis supports:

- Nadeau-Bengio correction;
- paired model comparisons;
- multiple-comparison adjustment using the Holm procedure.

Statistical outputs are generated from experiment results rather than hard-coded manuscript significance values.

---

## 15. Computational efficiency and scalability

Dedicated scripts are provided for:

- training-time profiling;
- inference-time profiling;
- memory measurement;
- graph/model computational assessment;
- scalability experiments.

These analyses are separated from predictive-performance evaluation so that computational cost can be assessed independently of discrimination metrics.

---

## 16. Generated outputs

Generated artifacts are written under `outputs/`:

```text
outputs/
├── tables/
├── figures/
├── predictions/
└── checkpoints/
```

`tables/` contains machine-readable experimental summaries.

`figures/` contains generated quantitative plots.

`predictions/` stores untouched outer-test probabilities used for pooled evaluation, calibration, threshold analysis, selective prediction, and statistical comparisons.

`checkpoints/` stores frozen fitted models and associated fold information required by post-hoc explainability and robustness analyses when generated by the corresponding workflow.

---

## 17. Repository structure

The current source repository is organized as follows:

```text
GNN-DHRS/
├── README.md
├── requirements.txt
├── LICENSE
├── configs/
│   └── paper.yaml
├── docs/
│   ├── ASSUMPTIONS.md
│   ├── MANUSCRIPT_TO_CODE.md
│   └── manuscript_source.tex
├── src/
│   └── gnn_dhrs/
│       ├── __init__.py
│       ├── ablations.py
│       ├── analysis.py
│       ├── baselines.py
│       ├── calibration.py
│       ├── checkpoints.py
│       ├── config.py
│       ├── data.py
│       ├── dhrs.py
│       ├── edge_analysis.py
│       ├── experiment.py
│       ├── explain.py
│       ├── ft_transformer.py
│       ├── graph.py
│       ├── graph_baselines.py
│       ├── graph_explainers.py
│       ├── graph_variants.py
│       ├── inductive.py
│       ├── metrics.py
│       ├── model.py
│       ├── plots.py
│       ├── profiling.py
│       ├── repro.py
│       ├── robustness.py
│       ├── stats.py
│       └── train.py
├── scripts/
│   ├── make_outputs.py
│   ├── run_all.py
│   ├── run_baselines.py
│   ├── run_calibration.py
│   ├── run_categorical_resampling.py
│   ├── run_computational_efficiency.py
│   ├── run_corrected_statistics.py
│   ├── run_cross_dataset.py
│   ├── run_dataset_audit.py
│   ├── run_dhrs_diagnostics.py
│   ├── run_explainability.py
│   ├── run_feature_topology.py
│   ├── run_ft_transformer.py
│   ├── run_graph_baselines.py
│   ├── run_graph_component_ablation.py
│   ├── run_graph_definition_ablation.py
│   ├── run_graph_explainers.py
│   ├── run_loss_ablation.py
│   ├── run_nested_cv.py
│   ├── run_resampling_ablation.py
│   ├── run_robustness.py
│   ├── run_scalability.py
│   ├── run_seed_stability.py
│   ├── run_selective_prediction.py
│   ├── run_sensitivity.py
│   ├── run_statistical_validation.py
│   ├── run_synthetic_edge_audit.py
│   ├── run_synthetic_node_ablation.py
│   └── run_threshold_analysis.py
└── tests/
    ├── test_dhrs.py
    ├── test_graph.py
    └── test_metrics.py
```

The `data/` and `outputs/` directories are created/used locally as required and need not contain redistributed patient data in the public repository.

---

## 18. Reproducibility safeguards

When reproducing or extending the experiments:

1. Never fit preprocessing on the complete dataset before cross-validation.
2. Never apply DHRS before the corresponding training/test split.
3. Never use outer-test labels for feature selection, model selection, threshold selection, or early stopping.
4. Never create test-to-test edges during inductive evaluation.
5. Construct SHAP background samples from training-derived observations only.
6. Preserve common outer folds when comparing competing methods.
7. Record stochastic seeds and software versions.
8. Report imbalance-sensitive metrics in addition to accuracy and ROC-AUC.
9. Use frozen fold models for post-hoc explanation and robustness analyses whenever the protocol requires them.
10. Do not hard-code manuscript values into experiment scripts.

---

## 19. Scientific interpretation

The experiments are conducted on public benchmark stroke datasets. High within-benchmark performance should not be interpreted as evidence that the same performance will necessarily be achieved in prospective clinical deployment.

Further validation using independently collected, multi-center, prospective, and demographically heterogeneous cohorts is required before clinical use.

Likewise, SHAP values, learned attention, GNNExplainer/PGExplainer outputs, and edge-ablation results are model-interpretation tools. They do not establish causal medical relationships.

---

## 20. Reproducibility status and assumptions

This repository is organized from the methodology and experimental analyses described in the manuscript.

Where a software-level implementation choice is not uniquely determined by the manuscript, the explicit choice is documented in:

```text
docs/ASSUMPTIONS.md
```

The complete manuscript-to-code experiment map is documented in:

```text
docs/MANUSCRIPT_TO_CODE.md
```

A successful software run demonstrates execution of the implemented protocol; **bit-for-bit numerical reproduction of manuscript tables should only be claimed after running the exact datasets and reconciling the generated outputs with the submitted results.**

---

## 21. Code availability

The source code and reproducibility materials for GNN-DHRS are publicly available in this repository:

`https://github.com/HanyEjust/GNN-DHRS`

Additional implementation information may be obtained from the corresponding author upon reasonable request.

---

## 22. Citation

If this implementation is used in academic work, please cite the associated manuscript. Full bibliographic information and DOI should be added after publication.

```bibtex
@article{elghaish_gnndhrs_2026,
  author = {El-Ghaish, Hany and others},
  title  = {An Explainable Graph Attention Framework with Dual Hybrid Resampling for Imbalanced Brain Stroke Prediction},
  year   = {2026},
  note   = {Manuscript under review}
}
```

---

## 23. License

The source code is released under the MIT License. Dataset licenses and usage conditions remain with the original data providers.
