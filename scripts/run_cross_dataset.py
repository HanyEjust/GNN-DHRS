"""Cross-benchmark transfer. Fits source preprocessing/model only on source; aligns target raw columns; no target labels used in fitting."""
import sys,itertools,numpy as np,pandas as pd
sys.path.insert(0,"src")
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.train import train_fixed_epochs
from gnn_dhrs.inductive import build_original_reference_h1,predict_queries
from gnn_dhrs.metrics import evaluate
cfg=load_config();D={n:load_dataset(n,s) for n,s in cfg["datasets"].items()};rows=[]
for src,tgt in itertools.permutations(D,2):
 a,b=D[src],D[tgt];common=[c for c in a.X.columns if c in b.X.columns];Xa,Xb=a.X[common],b.X[common];pp=FoldPreprocessor(.9,10).fit(Xa,a.y);A,T=pp.transform(Xa),pp.transform(Xb);R,ry,_,_=dhrs(A,a.y,11,1.0,pp.onehot_groups);ei,ew=directed_cosine_knn(R,10);m=train_fixed_epochs(R,ry,ei,ew,cfg,11,120);H=build_original_reference_h1(m,A,R,10);p=predict_queries(m,A,H,T,10);rows.append({"source":src,"target":tgt,**evaluate(b.y,p,.5)})
pd.DataFrame(rows).to_csv("outputs/tables/cross_dataset_transfer.csv",index=False)
