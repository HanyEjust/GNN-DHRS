import sys,numpy as np
sys.path.insert(0,"src")
from gnn_dhrs.metrics import evaluate,select_threshold,ece

def test_metrics_perfect():
 y=np.array([0,0,1,1]);p=np.array([.01,.1,.9,.99]);m=evaluate(y,p,.5);assert m["f1"]==1 and m["roc_auc"]==1
def test_threshold():
 y=np.array([0,1,1]);p=np.array([.2,.4,.8]);assert select_threshold(y,p,[.3,.5])==.3
def test_ece_bounds():
 assert 0<=ece(np.array([0,1]),np.array([.1,.9]))<=1
