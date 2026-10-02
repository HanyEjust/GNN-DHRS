import sys,os,pandas as pd
sys.path.insert(0,'src')
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset
from gnn_dhrs.plots import class_distribution,roc_pr_curves,calibration_plot,grouped_line
cfg=load_config();os.makedirs('outputs/figures',exist_ok=True);ds={n:load_dataset(n,s) for n,s in cfg['datasets'].items()};class_distribution(ds,'outputs/figures/class_distribution.png')
p=pd.read_csv('outputs/predictions/outer_test_predictions.csv');ref=p[p.seed==cfg['cv']['seeds'][0]];roc_pr_curves(ref,'outputs/figures/pooled');calibration_plot(ref,'outputs/figures/calibration_curves.png')
for path,x,y,group,out,title in [
 ('outputs/tables/threshold_sensitivity.csv','threshold','f1','dataset','outputs/figures/threshold_sensitivity.png','Threshold sensitivity'),
 ('outputs/tables/selective_prediction.csv','coverage','selective_risk','dataset','outputs/figures/selective_risk_coverage.png','Selective prediction'),
 ('outputs/tables/robustness.csv','level','f1','dataset','outputs/figures/robustness_f1.png','Robustness stress tests')]:
 if os.path.exists(path): grouped_line(pd.read_csv(path),x,y,group,out,title)
