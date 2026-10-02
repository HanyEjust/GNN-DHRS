"""Run the complete manuscript experiment suite in dependency order.
Warning: this suite is intentionally computationally expensive (nested CV, five seeds, SHAP, robustness and ablations).
"""
import subprocess,sys
scripts=[
 'run_dataset_audit.py','run_nested_cv.py','run_baselines.py','run_graph_baselines.py','run_ft_transformer.py','run_cross_dataset.py',
 'run_resampling_ablation.py','run_dhrs_diagnostics.py','run_categorical_resampling.py','run_loss_ablation.py',
 'run_synthetic_edge_audit.py','run_synthetic_node_ablation.py','run_graph_component_ablation.py','run_feature_topology.py',
 'run_sensitivity.py','run_seed_stability.py','run_graph_definition_ablation.py','run_threshold_analysis.py','run_calibration.py',
 'run_selective_prediction.py','run_explainability.py','run_graph_explainers.py','run_robustness.py','run_corrected_statistics.py','run_computational_efficiency.py','run_scalability.py','make_outputs.py']
for s in scripts:
 print('\n===',s,'===',flush=True); subprocess.run([sys.executable,'scripts/'+s],check=True)
