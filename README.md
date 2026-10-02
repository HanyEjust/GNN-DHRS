# GNN-DHRS

## Graph Neural Network with Dual Hybrid Resampling Strategy for Imbalanced Stroke Prediction

This repository provides the implementation and reproducibility framework for **GNN-DHRS**, a graph-based machine-learning framework developed for stroke prediction under severe class imbalance.

GNN-DHRS combines a **Dual Hybrid Resampling Strategy (DHRS)** with **patient-similarity graph construction** and a **multi-head Graph Attention Network (GAT)**. The experimental framework is designed around leakage-safe preprocessing, strictly inductive evaluation, nested cross-validation, imbalance-aware performance assessment, explainability analysis, robustness testing, and cross-dataset evaluation.

The repository is provided to support **scientific transparency, reproducibility, and independent validation** of the experiments reported in the associated research paper.

---

## Overview

Stroke prediction from structured clinical data presents several methodological challenges, particularly:

- severe class imbalance;
- heterogeneous clinical and demographic predictors;
- nonlinear relationships among risk factors;
- patient-to-patient similarity information;
- risk of information leakage during preprocessing and resampling;
- limited interpretability of complex predictive models;
- dataset shift and cross-dataset generalization.

GNN-DHRS addresses these issues through an integrated pipeline consisting of:

1. leakage-safe preprocessing;
2. training-only feature selection;
3. Dual Hybrid Resampling Strategy (DHRS);
4. patient-similarity graph construction;
5. graph-attention-based representation learning;
6. imbalance-aware optimization;
7. nested cross-validation;
8. validation-based decision-threshold selection;
9. explainability analysis;
10. robustness and sensitivity experiments.

---

## GNN-DHRS Framework

The main computational workflow is:

```text
Raw Clinical Dataset
        |
        v
Outer Stratified Cross-Validation
        |
        v
Training / Test Separation
        |
        +----------------------------+
        |                            |
        v                            |
Training Data Only                   |
        |                            |
        v                            |
Missing-Value Processing             |
        |                            |
        v                            |
Feature Transformation               |
        |                            |
        v                            |
Feature Selection                    |
        |                            |
        v                            |
Dual Hybrid Resampling (DHRS)        |
        |                            |
        v                            |
Patient-Similarity Graph             |
        |                            |
        v                            |
Multi-Head Graph Attention Network   |
        |                            |
        v                            |
Inner Validation / Model Selection   |
        |                            |
        +----------------------------+
        |
        v
Inductive Test-Patient Attachment
        |
        v
Final Prediction
        |
        v
Performance + Calibration +
Explainability + Robustness Analysis
```

All preprocessing, feature-selection, resampling, and model-selection operations are restricted to the appropriate training partitions.

---

## Dual Hybrid Resampling Strategy

The proposed pipeline employs **DHRS** to address severe class imbalance.

The strategy combines:

- **Borderline-SMOTE** for targeted minority-class oversampling;
- **SMOTEENN** for additional resampling and neighborhood cleaning.

Importantly, DHRS is applied **only to training data** within the corresponding cross-validation partition.

No synthetic samples are generated from validation or outer-test observations.

---

## Patient-Similarity Graph

Patients are represented as nodes in a similarity graph.

For the principal GNN-DHRS configuration:

- similarity metric: **cosine similarity**;
- graph construction: **k-nearest-neighbor graph**;
- number of neighbors: **k = 10**;
- node attributes: selected transformed patient-level predictors;
- test patients are attached **inductively** to training patients.

The evaluation procedure prevents test-to-test graph connectivity from being used to improve predictions.

This design is intended to preserve a realistic inductive evaluation setting.

---

## Graph Attention Network

The principal GNN-DHRS architecture uses a two-layer multi-head graph attention model.

| Parameter | Configuration |
|---|---:|
| GNN layers | 2 |
| Hidden dimension | 128 |
| Attention heads | 8 |
| Dimension per head | 16 |
| Dropout | 0.30 |
| LeakyReLU negative slope | 0.20 |
| Graph neighbors | 10 |
| Loss | Weighted Binary Cross-Entropy |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0005 |
| Maximum epochs | 200 |
| Early-stopping patience | 20 |
| Scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.50 |
| Scheduler patience | 10 |
| Minimum learning rate | 1e-6 |

The stochastic seeds considered in the experimental framework are:

```text
11, 22, 33, 44, 55
```

---

## Leakage-Safe Nested Cross-Validation

The main evaluation follows a nested stratified cross-validation design:

- **10 outer folds** for performance estimation;
- **5 inner folds** for model development and validation.

The outer-test observations remain untouched during:

- preprocessing fitting;
- feature selection;
- resampling;
- hyperparameter selection;
- threshold selection;
- model fitting.

This separation is central to the reproducibility protocol.

---

## Decision-Threshold Selection

Classification thresholds are selected using training-derived validation predictions rather than outer-test labels.

The evaluated threshold candidates are:

```text
0.10
0.20
0.30
0.40
0.50
0.60
0.70
0.80
0.90
```

F1 score is used as the threshold-selection objective in the principal implementation.

---

## Evaluation Metrics

The repository supports a comprehensive set of classification and imbalance-sensitive metrics.

These include:

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

For highly imbalanced stroke datasets, particular attention is given to:

**Recall, F1-score, PR-AUC, MCC, and Balanced Accuracy.**

---

## Benchmark Datasets

The experiments use three public stroke benchmark datasets referred to in the paper as:

```text
DF-1
DF-2
DF-3
```

The repository does not redistribute the original patient-level datasets.

Place the datasets in:

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

Identifier columns such as `id` are excluded from model inputs when present.

Users should obtain the datasets from the original public sources cited in the associated manuscript and comply with their respective licenses and terms of use.

---

## Experiments

The repository is organized to support the major experimental analyses reported in the paper.

### Main GNN-DHRS Evaluation

The principal experiment evaluates GNN-DHRS under leakage-safe nested cross-validation.

The pipeline includes:

```text
Preprocessing
     ↓
Feature Selection
     ↓
DHRS
     ↓
Patient Graph Construction
     ↓
GAT Training
     ↓
Validation-Based Threshold Selection
     ↓
Untouched Outer-Test Evaluation
```

### Baseline Comparisons

The experimental framework supports comparisons with conventional machine-learning, gradient-boosting, deep tabular, and graph-learning approaches.

Examples include:

- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- MLP
- XGBoost
- LightGBM
- CatBoost
- graph-based alternatives
- FT-Transformer

FT-Transformer provides a modern tabular deep-learning reference for examining whether the observed performance can be explained by advanced tabular feature-interaction modeling rather than graph-based relational learning alone.

### Cross-Dataset Evaluation

Cross-benchmark experiments examine transfer between the stroke datasets.

These experiments are intended to assess how models trained on one benchmark behave when evaluated on another dataset with potentially different:

- predictor distributions;
- class prevalence;
- patient characteristics;
- similarity structures.

Cross-dataset results should therefore be interpreted separately from within-dataset cross-validation performance.

### Ablation Analysis

The framework supports controlled ablation experiments examining the contribution of major components, including:

- DHRS;
- graph construction;
- attention-based message passing;
- graph-neighborhood size;
- model architecture;
- resampling configuration.

### Hyperparameter Sensitivity

The repository provides configuration support for evaluating:

```yaml
attention_heads:
  [2, 4, 8, 12]

hidden_dimension:
  [32, 64, 128, 256]

dropout:
  [0.10, 0.30, 0.50, 0.70]

learning_rate:
  [0.0001, 0.0005, 0.001, 0.005]

weight_decay:
  [0.0001, 0.0005, 0.001]

DHRS_oversampling_ratio:
  [0.50, 0.75, 1.00, 1.25]

graph_k:
  [5, 10, 15, 20, 25]
```

---

## Explainability

Interpretability is investigated from both feature-level and graph-relational perspectives.

### Kernel SHAP

Kernel SHAP is applied to the **selected patient-level input features**.

The explained output is the final stroke probability generated by the complete fitted GNN-DHRS inference pipeline.

The SHAP background is constructed from training-derived observations only.

The experimental configuration uses:

- 100 background patients;
- 50 stroke patients;
- 50 non-stroke patients.

Validation and outer-test patients are excluded from background construction.

SHAP explanations should be interpreted as feature contributions within the complete graph-based predictive pipeline rather than as causal effects.

### Graph-Level Interpretation

Graph-related interpretation is complemented by analysis of learned attention and controlled edge perturbation.

This helps investigate whether particular patient-to-patient relations materially affect the fitted predictions.

These analyses provide **model-level interpretability** and should not be interpreted as evidence of causal or clinically verified relationships between patients.

---

## Calibration and Selective Prediction

The framework also evaluates probabilistic prediction quality.

Calibration analyses include:

- Brier score;
- Expected Calibration Error;
- calibration curves.

Selective-prediction experiments evaluate model behavior when predictions with lower confidence are withheld.

Target coverage levels include:

```text
100%
95%
90%
80%
70%
```

---

## Robustness Analysis

The experimental framework includes stress tests for several forms of perturbation.

### Missingness

```text
0%
10%
20%
30%
```

### Gaussian Feature Noise

```text
σ = 0.00
σ = 0.05
σ = 0.10
σ = 0.20
```

### Prevalence Shift

```text
0.5×
1.0×
2.0×
```

### Graph Edge Corruption

```text
0%
10%
20%
30%
```

These experiments are intended to characterize the sensitivity of the trained framework to perturbations affecting patient features, class composition, and graph structure.

---

## Repository Structure

```text
GNN-DHRS/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── configs/
│   └── paper.yaml
│
├── data/
│   ├── raw/
│   │   ├── DF-1.csv
│   │   ├── DF-2.csv
│   │   └── DF-3.csv
│   └── processed/
│
├── src/
│   └── gnn_dhrs/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── dhrs.py
│       ├── graph.py
│       ├── model.py
│       ├── train.py
│       ├── metrics.py
│       ├── experiment.py
│       ├── analysis.py
│       ├── robustness.py
│       ├── stats.py
│       └── repro.py
│
├── scripts/
│   ├── run_dataset_audit.py
│   ├── run_nested_cv.py
│   ├── run_baselines.py
│   ├── run_cross_dataset.py
│   ├── run_ablation.py
│   ├── run_sensitivity.py
│   ├── run_explainability.py
│   ├── run_robustness.py
│   ├── make_outputs.py
│   └── run_all.py
│
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── predictions/
│   └── checkpoints/
│
├── tests/
│   └── test_metrics.py
│
└── docs/
    └── MANUSCRIPT_TO_CODE.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/HanyEjust/GNN-DHRS.git
cd GNN-DHRS
```

Create a Python environment if desired, then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Software Environment

The experimental environment reported for the study includes:

| Software | Version |
|---|---:|
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

The reported computational environment used:

- Kaggle Notebook;
- Ubuntu 20.04.6 LTS;
- NVIDIA Tesla T4 GPU;
- 16 GB GPU memory;
- approximately 30 GB system RAM.

---

## Running the Experiments

### Dataset Audit

```bash
python scripts/run_dataset_audit.py
```

This generates dataset-level integrity and class-distribution information.

### Main Nested Cross-Validation

```bash
python scripts/run_nested_cv.py
```

### Baseline Experiments

```bash
python scripts/run_baselines.py
```

### Cross-Dataset Evaluation

```bash
python scripts/run_cross_dataset.py
```

### Ablation Experiments

```bash
python scripts/run_ablation.py
```

### Hyperparameter Sensitivity

```bash
python scripts/run_sensitivity.py
```

### Explainability Analysis

```bash
python scripts/run_explainability.py
```

### Robustness Experiments

```bash
python scripts/run_robustness.py
```

### Generate Tables and Figures

```bash
python scripts/make_outputs.py
```

### Complete Pipeline

```bash
python scripts/run_all.py
```

---

## Generated Outputs

Experimental outputs are organized under:

```text
outputs/
```

### Tables

```text
outputs/tables/
```

This directory stores machine-readable experiment summaries and manuscript-oriented tables.

### Figures

```text
outputs/figures/
```

This directory contains programmatically generated quantitative figures.

### Predictions

```text
outputs/predictions/
```

Outer-test predictions can be retained for:

- pooled ROC analysis;
- pooled precision-recall analysis;
- calibration;
- threshold analysis;
- statistical comparisons;
- selective prediction.

### Checkpoints

```text
outputs/checkpoints/
```

This directory can be used for trained model states required for subsequent explainability and robustness experiments.

---

## Reproducibility Principles

Several safeguards are important when reproducing the reported experiments.

1. **Never fit preprocessing on the complete dataset before cross-validation.**
2. **Never apply DHRS before splitting the data.**
3. **Never use outer-test labels for model or threshold selection.**
4. **Never construct test-to-test graph connections during inductive evaluation.**
5. **Construct SHAP background samples from training data only.**
6. **Preserve the same outer folds when comparing competing methods.**
7. **Record random seeds and software versions.**
8. **Report imbalance-sensitive measures in addition to accuracy and ROC-AUC.**

These controls are necessary to avoid optimistic performance estimates.

---

## Interpretation of Results

The repository is intended to reproduce experiments on public benchmark datasets.

High within-benchmark performance should **not** be interpreted as evidence that equivalent performance will necessarily be obtained in prospective clinical practice.

The reported results require further validation using:

- independently collected cohorts;
- multi-center datasets;
- prospective clinical data;
- heterogeneous patient populations.

Likewise, SHAP values, graph attention, and edge-ablation analyses characterize the behavior of the fitted model. They do not establish causal clinical relationships.

---

## Reproducibility Note

The implementation is organized according to the methodology and experimental protocol described in the manuscript.

Where a low-level implementation choice is not uniquely specified by the manuscript, it should be reconciled with the original experimental notebook before claiming **bit-for-bit numerical reproduction**.

The repository intentionally calculates experimental results from the supplied data rather than embedding the numerical values reported in the paper.

---

## Citation

If you use this framework or code in academic work, please cite the associated paper.

```bibtex
@article{elghaish_gnndhrs_2026,
  author  = {El-Ghaish, Hany and others},
  title   = {GNN-DHRS},
  year    = {2026},
  note    = {Full bibliographic information will be added after publication}
}
```

The BibTeX entry will be updated when the final publication information and DOI become available.

---

## Code Availability

The implementation is publicly provided to facilitate:

- reproduction of the reported experiments;
- independent validation;
- methodological comparison;
- extension of the proposed framework.

Repository:

https://github.com/HanyEjust/GNN-DHRS

---

## License

This project is released under the **MIT License**.

The source code may be used and modified according to the terms of the license.

The benchmark datasets are subject to the licenses and usage conditions established by their original providers.

---

## Contact

**Assoc. Prof. Hany El-Ghaish**  
Computers and Control Engineering  
Faculty of Engineering  
Tanta University, Egypt

GitHub: https://github.com/HanyEjust

For questions concerning the implementation or reproducibility of the experiments, please use the GitHub repository's Issues section.
