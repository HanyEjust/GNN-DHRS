import sys
sys.path.insert(0,"src")
from gnn_dhrs.config import load_config
from gnn_dhrs.experiment import run_all
cfg=load_config();m,p=run_all(cfg);print(m.groupby("dataset")[["recall","f1","pr_auc","roc_auc"]].agg(["mean","std"]))
