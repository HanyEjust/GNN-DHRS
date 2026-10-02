import sys,pandas as pd
sys.path.insert(0,"src")
from gnn_dhrs.config import load_config
from gnn_dhrs.metrics import selective_curve
from gnn_dhrs.analysis import aurc
cfg=load_config();p=pd.read_csv("outputs/predictions/outer_test_predictions.csv");rows=[]
for (d,s),g in p.groupby(["dataset","seed"]):
 for r in selective_curve(g.y.to_numpy(),g.p.to_numpy(),cfg["selective_prediction"]["coverages"]):rows.append({"dataset":d,"seed":s,**r})
 rows[-1]["aurc"]=aurc(g.y.to_numpy(),g.p.to_numpy())
pd.DataFrame(rows).to_csv("outputs/tables/selective_prediction.csv",index=False)
