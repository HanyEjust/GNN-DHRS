import sys
sys.path.insert(0,"src")
from gnn_dhrs.config import load_config
from gnn_dhrs.experiment import run_all
from gnn_dhrs.analysis import summarize_folds
cfg=load_config();m,p=run_all(cfg,cfg["cv"]["seeds"]);summarize_folds(m).to_csv("outputs/tables/random_seed_stability.csv")
