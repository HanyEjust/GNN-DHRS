import sys,numpy as np,pandas as pd
sys.path.insert(0,"src")
from sklearn.model_selection import StratifiedKFold
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.baselines import make_tabular,fit_with_imbalance,proba
from gnn_dhrs.metrics import evaluate,select_threshold
cfg=load_config();names=["LR","DT","RF","ExtraTrees","GB","MLP","SVM","XGBoost-CW","LightGBM-CW","CatBoost-CW"];rows=[]
for dn,spec in cfg["datasets"].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(10,shuffle=True,random_state=cfg["cv"]["split_seed"])
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);A=pp.transform(ds.X.iloc[tr]);T=pp.transform(ds.X.iloc[te]);y=ds.y[tr]
  for name in names:
   m=fit_with_imbalance(make_tabular(name,11),A,y);p=proba(m,T);rows.append({"dataset":dn,"fold":fold,"model":name,**evaluate(ds.y[te],p,.5)})
pd.DataFrame(rows).to_csv("outputs/tables/tabular_baselines.csv",index=False)
