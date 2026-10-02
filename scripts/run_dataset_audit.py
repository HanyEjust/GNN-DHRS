import sys,pandas as pd
sys.path.insert(0,"src")
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,integrity_audit
cfg=load_config();rows=[]
for n,s in cfg["datasets"].items():rows.append(integrity_audit(load_dataset(n,s)))
pd.DataFrame(rows).to_csv("outputs/tables/dataset_audit.csv",index=False);print(pd.DataFrame(rows).to_string(index=False))
