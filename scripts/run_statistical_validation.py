import sys,pandas as pd
sys.path.insert(0,"src")
from gnn_dhrs.analysis import paired_wilcoxon
m=pd.read_csv("outputs/tables/nested_cv_fold_metrics.csv");b=pd.read_csv("outputs/tables/tabular_baselines.csv");rows=[]
for d in m.dataset.unique():
 a=m[(m.dataset==d)&(m.seed==m.seed.min())].sort_values("fold")
 for model,g in b[b.dataset==d].groupby("model"):
  g=g.sort_values("fold");
  for metric in ["recall","f1","pr_auc"]:rows.append({"dataset":d,"baseline":model,"metric":metric,**paired_wilcoxon(a[metric],g[metric])})
pd.DataFrame(rows).to_csv("outputs/tables/statistical_validation.csv",index=False)
