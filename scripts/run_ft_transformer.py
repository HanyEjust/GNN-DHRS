import sys,numpy as np,pandas as pd
sys.path.insert(0,"src")
from sklearn.model_selection import StratifiedKFold,train_test_split
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.ft_transformer import train_ft,ft_predict
from gnn_dhrs.metrics import evaluate,select_threshold
cfg=load_config();rows=[]
for dn,spec in cfg["datasets"].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(10,shuffle=True,random_state=11)
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  a,v=train_test_split(tr,test_size=.15,stratify=ds.y[tr],random_state=11);pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[a],ds.y[a]);A,V,T=pp.transform(ds.X.iloc[a]),pp.transform(ds.X.iloc[v]),pp.transform(ds.X.iloc[te])
  for mode in ["FT-T-W","FT-T-DHRS"]:
   X,y=A,ds.y[a]
   if mode.endswith("DHRS"):X,y,_,_=dhrs(A,y,11,1.0,pp.onehot_groups)
   m=train_ft(X,y,V,ds.y[v],11,class_weight=mode.endswith("W"));pv=ft_predict(m,V);th=select_threshold(ds.y[v],pv,cfg["threshold"]["candidates"]);p=ft_predict(m,T);rows.append({"dataset":dn,"fold":fold,"model":mode,**evaluate(ds.y[te],p,th)})
pd.DataFrame(rows).to_csv("outputs/tables/ft_transformer_baselines.csv",index=False)
