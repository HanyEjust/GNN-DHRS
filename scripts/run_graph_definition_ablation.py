import sys,pandas as pd,numpy as np
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold,train_test_split
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph_variants import graph_from_definition,graph_stats
from gnn_dhrs.train import train_gnn,train_fixed_epochs,predict
from gnn_dhrs.metrics import evaluate,select_threshold
cfg=load_config(); defs=['euclidean_knn','gower_knn','mutual_cosine_knn','cosine_threshold','cosine_knn']; rows=[];top=[]
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec); cv=StratifiedKFold(cfg['cv']['outer_folds'],shuffle=True,random_state=cfg['cv']['split_seed'])
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  a,v=train_test_split(np.arange(len(tr)),test_size=.15,stratify=ds.y[tr],random_state=11); tri=tr[a]; vi=tr[v]
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tri],ds.y[tri]);A=pp.transform(ds.X.iloc[tri]);V=pp.transform(ds.X.iloc[vi]);D,y,sy,_=dhrs(A,ds.y[tri],11,1.0,pp.onehot_groups)
  for gd in defs:
   ei,ew=graph_from_definition(D,gd,10); C=np.vstack([D,V]); # validation nodes are scored without labels entering training loss
   # train graph only; use simple validation split within D for epoch selection, then fixed refit
   idx=np.arange(len(D)); it,iv=train_test_split(idx,test_size=.12,stratify=y,random_state=11); vm=np.zeros(len(D),bool);vm[iv]=1;m,ep=train_gnn(D,y,ei,ew,vm,cfg,11,return_best_epoch=True); pv=predict(m,D,ei,ew)[iv]; th=select_threshold(y[iv],pv,cfg['threshold']['candidates']);
   pp2=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);R=pp2.transform(ds.X.iloc[tr]);T=pp2.transform(ds.X.iloc[te]);G,gy,gsy,_=dhrs(R,ds.y[tr],11,1.0,pp2.onehot_groups);gei,gew=graph_from_definition(G,gd,10);fm=train_fixed_epochs(G,gy,gei,gew,cfg,11,ep)
   # transductive-free query graph for graph variants: attach query to reference and run a local star through standard cosine inductive path for primary; for alternatives evaluate frozen full graph + one query
   from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,manhattan_distances
   probs=[]
   for q in T:
    if gd in ('cosine_knn','mutual_cosine_knn','cosine_threshold'): sim=cosine_similarity(q.reshape(1,-1),R)[0]
    elif gd=='euclidean_knn': sim=1/(1+euclidean_distances(q.reshape(1,-1),R)[0])
    else: sim=1/(1+manhattan_distances(q.reshape(1,-1),R)[0]/R.shape[1])
    nbr=np.argsort(-sim)[:10]; import torch; qx=np.vstack([G,q]); qei=torch.cat([gei,torch.tensor([nbr,[len(G)]*len(nbr)],dtype=torch.long)],1); qew=torch.cat([gew,torch.tensor(sim[nbr],dtype=torch.float32)]); probs.append(predict(fm,qx,qei,qew)[-1])
   rows.append({'dataset':dn,'fold':fold,'graph_definition':gd,**evaluate(ds.y[te],np.asarray(probs),th)}); top.append({'dataset':dn,'fold':fold,'graph_definition':gd,'stage':'after_dhrs',**graph_stats(gei,gy,gsy)})
pd.DataFrame(rows).to_csv('outputs/tables/graph_definition_performance.csv',index=False);pd.DataFrame(top).to_csv('outputs/tables/graph_definition_topology.csv',index=False)
