import sys,pandas as pd
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.graph_variants import graph_stats
cfg=load_config();rows=[]
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(10,shuffle=True,random_state=11)
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);A=pp.transform(ds.X.iloc[tr]);D,y,syn,a=dhrs(A,ds.y[tr],11,1.0,pp.onehot_groups);ei,ew=directed_cosine_knn(D,10);rows.append({'dataset':dn,'fold':fold,'synthetic_nodes':int(syn.sum()),'real_nodes':int((~syn).sum()),**a,**graph_stats(ei,y,syn)})
pd.DataFrame(rows).to_csv('outputs/tables/synthetic_edge_audit.csv',index=False)
