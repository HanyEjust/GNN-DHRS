import numpy as np,pandas as pd
from scipy.stats import wilcoxon
from .metrics import selective_curve

def summarize_folds(df):
    metrics=[c for c in ["accuracy","precision","recall","specificity","npv","f1","balanced_accuracy","mcc","kappa","roc_auc","pr_auc","brier","ece"] if c in df]
    return df.groupby([c for c in ["dataset","seed"] if c in df.columns])[metrics].agg(["mean","std"])
def paired_wilcoxon(a,b):
    a=np.asarray(a);b=np.asarray(b); stat,p=wilcoxon(a,b,zero_method="wilcox",alternative="two-sided"); d=(a-b); return {"W":float(stat),"p":float(p),"median_difference":float(np.median(d))}
def aurc(y,p,points=101):
    cov=np.linspace(.01,1,points); rows=selective_curve(np.asarray(y),np.asarray(p),cov); return float(np.trapz([r["selective_risk"] for r in rows],[r["coverage"] for r in rows]))
def jaccard(a,b):
    a=set(a);b=set(b);return len(a&b)/max(1,len(a|b))
