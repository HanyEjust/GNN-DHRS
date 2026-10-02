import sys,pandas as pd
sys.path.insert(0,"src")
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset
from gnn_dhrs.plots import class_distribution,roc_pr_curves,calibration_plot
cfg=load_config();ds={n:load_dataset(n,s) for n,s in cfg["datasets"].items()};class_distribution(ds,"outputs/figures/class_distribution.png")
p=pd.read_csv("outputs/predictions/outer_test_predictions.csv");ref=p[p.seed==cfg["cv"]["seeds"][0]];roc_pr_curves(ref,"outputs/figures/pooled");calibration_plot(ref,"outputs/figures/calibration_curves.png")
