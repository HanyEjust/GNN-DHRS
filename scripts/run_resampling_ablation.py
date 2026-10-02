import sys,numpy as np,pandas as pd
sys.path.insert(0,"src")
from sklearn.model_selection import StratifiedKFold,train_test_split
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import resample_variant,dhrs
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.train import train_fixed_epochs
from gnn_dhrs.inductive import build_original_reference_h1,predict_queries
from gnn_dhrs.metrics import evaluate
cfg=load_config();rows=[];variants=["none","smote","borderline","smoteenn","bsm_enn","reverse_dhrs","dhrs"]
for dn,spec in cfg["datasets"].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(10,shuffle=True,random_state=11)
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);A,T=pp.transform(ds.X.iloc[tr]),pp.transform(ds.X.iloc[te])
  for v in variants:
   if v=="dhrs":R,ry,_,_=dhrs(A,ds.y[tr],11,1.0,pp.onehot_groups)
   else:R,ry=resample_variant(v,A,ds.y[tr],11,1.0,pp.onehot_groups)
   ei,ew=directed_cosine_knn(R,10);m=train_fixed_epochs(R,ry,ei,ew,cfg,11,120);H=build_original_reference_h1(m,A,R,10);p=predict_queries(m,A,H,T,10);rows.append({"dataset":dn,"fold":fold,"strategy":v,**evaluate(ds.y[te],p,.5)})
pd.DataFrame(rows).to_csv("outputs/tables/resampling_ablation.csv",index=False)
