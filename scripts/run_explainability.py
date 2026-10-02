"""Kernel-SHAP and graph-explanation primitives are in gnn_dhrs.explain and inductive.py.
For scientific validity, explanations must be generated from the exact frozen fold models used for outer-test predictions. The production workflow should save those checkpoints during the expensive nested-CV run, then invoke these functions on 10+10 outer-test cases/fold and 50+50 original-training background cases/fold.
"""
import sys
sys.path.insert(0,"src")
from gnn_dhrs.explain import kernel_shap,fidelity,stability
print("Explainability primitives loaded. Use frozen outer-fold checkpoints; do not retrain a different model for explanations.")
