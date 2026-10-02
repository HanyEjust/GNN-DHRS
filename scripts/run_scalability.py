import sys,time,numpy as np,pandas as pd
sys.path.insert(0,'src')
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.graph import directed_cosine_knn
cfg=load_config();rows=[]
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec);pp=FoldPreprocessor(.9,10).fit(ds.X,ds.y);X=pp.transform(ds.X)
 for frac in [.25,.5,.75,1.0]:
  n=max(20,int(len(X)*frac));t=time.perf_counter();ei,ew=directed_cosine_knn(X[:n],10);rows.append({'dataset':dn,'fraction':frac,'nodes':n,'edges':ei.shape[1],'graph_seconds':time.perf_counter()-t})
pd.DataFrame(rows).to_csv('outputs/tables/scalability.csv',index=False)
