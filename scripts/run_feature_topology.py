import sys,pandas as pd,numpy as np
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph_variants import graph_from_definition,graph_stats
cfg=load_config();rows=[]
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec); cv=StratifiedKFold(cfg['cv']['outer_folds'],shuffle=True,random_state=cfg['cv']['split_seed'])
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  for fc in cfg['preprocessing']['feature_counts']:
   pp=FoldPreprocessor(cfg['preprocessing']['corr_threshold'],fc).fit(ds.X.iloc[tr],ds.y[tr]);A=pp.transform(ds.X.iloc[tr]);D,y,sy,_=dhrs(A,ds.y[tr],11,cfg['dhrs']['oversampling_ratio'],pp.onehot_groups);ei,ew=graph_from_definition(D,'cosine_knn',cfg['graph']['k']); rows.append({'dataset':dn,'fold':fold,'features':fc,**graph_stats(ei,y,sy)})
pd.DataFrame(rows).to_csv('outputs/tables/feature_topology.csv',index=False)
