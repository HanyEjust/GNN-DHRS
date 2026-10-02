import numpy as np, shap
from .analysis import jaccard

def stratified_indices(y,n_per_class,seed):
    rng=np.random.default_rng(seed); return np.r_[rng.choice(np.where(y==0)[0],min(n_per_class,np.sum(y==0)),False),rng.choice(np.where(y==1)[0],min(n_per_class,np.sum(y==1)),False)]
def kernel_shap(predict_fn,X_background,X_explain,nsamples=256):
    explainer=shap.KernelExplainer(predict_fn,X_background); return explainer.shap_values(X_explain,nsamples=nsamples,silent=True)
def fidelity(p_full,p_removed,p_only): return {"fid_plus":float(abs(p_full-p_removed)),"fid_minus":float(abs(p_full-p_only))}
def stability(edge_sets):
    vals=[]
    for i in range(len(edge_sets)):
        for j in range(i+1,len(edge_sets)): vals.append(jaccard(edge_sets[i],edge_sets[j]))
    return float(np.mean(vals)) if vals else np.nan
