# Manuscript-to-code traceability

This file maps the experimental claims in the manuscript to executable repository entry points. The scripts calculate results from the supplied datasets; manuscript numbers are not embedded in code.

| Manuscript analysis | Entry point | Primary output |
|---|---|---|
| Dataset integrity/class distribution | `scripts/run_dataset_audit.py` | `outputs/tables/dataset_audit.csv` |
| Main leakage-safe nested CV | `scripts/run_nested_cv.py` | `outputs/tables/nested_cv_fold_metrics.csv` |
| Classifier-wise tabular baselines | `scripts/run_baselines.py` | `outputs/tables/tabular_baselines.csv` |
| GCN/GraphSAGE/GAT imbalance-aware baselines | `scripts/run_graph_baselines.py` | `outputs/tables/graph_baselines.csv` |
| FT-Transformer-W / FT-Transformer-DHRS | `scripts/run_ft_transformer.py` | `outputs/tables/ft_transformer_baselines.csv` |
| Cross-benchmark transfer | `scripts/run_cross_dataset.py` | `outputs/tables/cross_dataset.csv` |
| Resampling strategies/order ablation | `scripts/run_resampling_ablation.py` | `outputs/tables/resampling_ablation.csv` |
| DHRS stage-wise distribution | `scripts/run_dhrs_diagnostics.py` | `outputs/tables/dhrs_stage_distribution.csv` |
| Projected one-hot vs SMOTENC-ENN | `scripts/run_categorical_resampling.py` | `outputs/tables/categorical_resampling_sensitivity.csv` |
| BCE vs post-DHRS WBCE | `scripts/run_loss_ablation.py` | `outputs/tables/class_weight_ablation.csv` |
| Synthetic-node/edge provenance | `scripts/run_synthetic_edge_audit.py` | `outputs/tables/synthetic_edge_audit.csv` |
| Synthetic-node message-passing ablation | `scripts/run_synthetic_node_ablation.py` | `outputs/tables/synthetic_node_ablation.csv` |
| Graph/attention/similarity ablation | `scripts/run_graph_component_ablation.py` | `outputs/tables/graph_component_ablation.csv` |
| Feature-count/topology analysis | `scripts/run_feature_topology.py` | `outputs/tables/feature_topology.csv` |
| Hyperparameter sensitivity | `scripts/run_sensitivity.py` | `outputs/tables/hyperparameter_sensitivity.csv` |
| Five-seed stability | `scripts/run_seed_stability.py` | `outputs/tables/random_seed_stability.csv` |
| Graph-definition comparison/topology | `scripts/run_graph_definition_ablation.py` | `outputs/tables/graph_definition_*.csv` |
| Threshold sensitivity | `scripts/run_threshold_analysis.py` | `outputs/tables/threshold_sensitivity.csv` |
| Calibration | `scripts/run_calibration.py` | `outputs/tables/calibration_*.csv` |
| Selective prediction/risk coverage | `scripts/run_selective_prediction.py` | `outputs/tables/selective_prediction.csv` |
| Kernel SHAP + attention edge ablation + SHAP stability | `scripts/run_explainability.py` | `outputs/tables/kernel_shap_importance.csv`, `edge_ablation_results.csv` |
| PyG GNNExplainer/PGExplainer adapters | `src/gnn_dhrs/graph_explainers.py` | callable from frozen fold graphs |
| Corrected Nadeau-Bengio statistics + Holm | `scripts/run_corrected_statistics.py` | `outputs/tables/corrected_statistical_validation.csv` |
| Computational efficiency/memory | `scripts/run_computational_efficiency.py` | `outputs/tables/computational_efficiency.csv` |
| Scalability | `scripts/run_scalability.py` | `outputs/tables/scalability.csv` |
| Missingness/noise/prevalence/edge-corruption robustness | `scripts/run_robustness.py` | `outputs/tables/robustness.csv` |

## Reproduction policy

All preprocessing and resampling must be fitted on training partitions only. Outer-test labels may be used only for final scoring. Test patients are attached inductively and no test-to-test edges are used. If fresh execution differs from a manuscript table, the discrepancy must be reported and the manuscript corrected; output values must never be hard-coded to force agreement.
