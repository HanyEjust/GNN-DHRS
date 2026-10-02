"""One-factor-at-a-time sensitivity. This is computationally expensive."""
import sys,copy,pandas as pd
sys.path.insert(0,"src")
from gnn_dhrs.config import load_config
from gnn_dhrs.experiment import run_nested_dataset
cfg=load_config();rows=[]
mapkey={"heads":("model","heads"),"hidden_dim":("model","hidden_dim"),"dropout":("model","dropout"),"lr":("training","lr"),"weight_decay":("training","weight_decay"),"oversampling_ratio":("dhrs","oversampling_ratio"),"graph_k":("graph","k")}
for par,vals in cfg["sensitivity"].items():
 for val in vals:
  c=copy.deepcopy(cfg);sec,key=mapkey[par];c[sec][key]=val
  for dn,spec in c["datasets"].items():
   m,_=run_nested_dataset(dn,spec,c,11);rows.append({"parameter":par,"value":val,"dataset":dn,"recall":m.recall.mean(),"f1":m.f1.mean(),"pr_auc":m.pr_auc.mean()})
pd.DataFrame(rows).to_csv("outputs/tables/hyperparameter_sensitivity.csv",index=False)
