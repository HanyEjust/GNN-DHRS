import sys,pandas as pd
sys.path.insert(0,'src')
from gnn_dhrs.metrics import evaluate
from gnn_dhrs.calibration import reliability_bins
p=pd.read_csv('outputs/predictions/outer_test_predictions.csv'); rows=[]; bins=[]
for (d,s),g in p.groupby(['dataset','seed']):
 z=evaluate(g.y.to_numpy(),g.p.to_numpy(),.5); rows.append({'dataset':d,'seed':s,'brier':z['brier'],'ece':z['ece']})
 for r in reliability_bins(g.y.to_numpy(),g.p.to_numpy(),10): bins.append({'dataset':d,'seed':s,**r})
pd.DataFrame(rows).to_csv('outputs/tables/calibration_metrics.csv',index=False);pd.DataFrame(bins).to_csv('outputs/tables/calibration_bins.csv',index=False)
