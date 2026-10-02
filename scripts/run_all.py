import subprocess,sys
steps=["run_dataset_audit.py","run_nested_cv.py","run_baselines.py","run_ft_transformer.py","run_cross_dataset.py","run_resampling_ablation.py","run_selective_prediction.py","run_statistical_validation.py","make_outputs.py"]
for s in steps:
 print(f"\n=== {s} ===",flush=True);subprocess.run([sys.executable,"scripts/"+s],check=True)
print("Core paper pipeline complete. Run run_sensitivity.py separately because it is substantially more expensive.")
