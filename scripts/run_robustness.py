"""Robustness utilities are implemented in gnn_dhrs.robustness. Full stress tests require saved fold checkpoints; run_nested_cv can be extended with --save-checkpoints for deployment-scale studies. This script validates perturbation generators."""
import sys,numpy as np,pandas as pd
sys.path.insert(0,"src")
from gnn_dhrs.robustness import add_missingness,add_gaussian_noise,prevalence_subsample
X=np.zeros((100,10));y=np.r_[np.zeros(90,int),np.ones(10,int)];rows=[]
for r in [0,.1,.2,.3]:rows.append({"stress":"missingness","level":r,"finite":np.isfinite(add_missingness(X,r)).all()})
for r in [0,.05,.1,.2]:rows.append({"stress":"noise","level":r,"finite":np.isfinite(add_gaussian_noise(X,r)).all()})
for r in [.5,1,2]:rows.append({"stress":"prevalence","level":r,"n":len(prevalence_subsample(y,r))})
pd.DataFrame(rows).to_csv("outputs/tables/robustness_generator_validation.csv",index=False)
