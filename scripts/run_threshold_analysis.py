import sys,pandas as pd,numpy as np
sys.path.insert(0,'src')
from gnn_dhrs.config import load_config
from gnn_dhrs.metrics import evaluate
cfg=load_config(); p=pd.read_csv('outputs/predictions/outer_test_predictions.csv'); rows=[]
for (d,s,f),g in p.groupby(['dataset','seed','fold']):
 for t in cfg['threshold']['candidates']:
  z=evaluate(g.y.to_numpy(),g.p.to_numpy(),t); rows.append({'dataset':d,'seed':s,'fold':f,'threshold':t,'precision':z['precision'],'recall':z['recall'],'f1':z['f1']})
pd.DataFrame(rows).to_csv('outputs/tables/threshold_sensitivity.csv',index=False)
