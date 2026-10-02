"""Frozen-model comparison of random, GNNExplainer, PGExplainer and attention edge rankings.
For tractability, the configured SHAP-sized stratified outer-test subset is explained per fold. All methods receive the same five incoming query-edge budget.
"""
import sys,numpy as np,pandas as pd,torch
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.train import train_gnn,train_fixed_epochs,predict
from gnn_dhrs.explain import stratified_indices
from gnn_dhrs.graph_explainers import gnnexplainer_edges,pgexplainer_edges
cfg=load_config();rows=[]
def eval_subset(model,X,ei,ew,qidx,selected):
 base=float(predict(model,X,ei,ew)[qidx]); allq=np.where(ei[1].numpy()==qidx)[0];keep=np.ones(ei.shape[1],bool);keep[selected]=False; prem=float(predict(model,X,ei[:,keep],ew[keep])[qidx]);only=np.zeros(ei.shape[1],bool);only[selected]=True; # preserve training graph plus selected query edges
 only |= (ei[1].numpy()!=qidx);ponly=float(predict(model,X,ei[:,only],ew[only])[qidx]);return abs(base-prem),abs(base-ponly)
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(10,shuffle=True,random_state=11)
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);R,T=pp.transform(ds.X.iloc[tr]),pp.transform(ds.X.iloc[te]);G,gy,_,_=dhrs(R,ds.y[tr],11,1.0,pp.onehot_groups);gei,gew=directed_cosine_knn(G,10);idx=np.arange(len(G));it,iv=train_test_split(idx,test_size=.12,stratify=gy,random_state=11);vm=np.zeros(len(G),bool);vm[iv]=1;m0,ep=train_gnn(G,gy,gei,gew,vm,cfg,11,return_best_epoch=True);m=train_fixed_epochs(G,gy,gei,gew,cfg,11,ep);dev=next(m.parameters()).device
  sel=stratified_indices(ds.y[te],cfg['explainability']['shap_test_per_class'],fold+100)
  for qi in sel:
   q=T[qi];s=cosine_similarity(q.reshape(1,-1),G)[0];nbr=np.where(s>0)[0];nbr=nbr[np.argsort(-s[nbr])[:10]];qe=torch.tensor([nbr,[len(G)]*len(nbr)],dtype=torch.long);qw=torch.tensor(s[nbr],dtype=torch.float32);E=torch.cat([gei,qe],1);W=torch.cat([gew,qw]);X=torch.tensor(np.vstack([G,q]),dtype=torch.float32,device=dev);E=E.to(dev);W=W.to(dev);qnode=len(G);cand=np.arange(gei.shape[1],E.shape[1]);budget=min(cfg['explainability']['edge_budget'],len(cand))
   # attention scores from frozen model
   _,(_,att)=predict(m,X.cpu().numpy(),E.cpu(),W.cpu(),attention=True);score=att.mean(1).numpy();attention=cand[np.argsort(-score[cand])[:budget]];rng=np.random.default_rng(fold+qi);random=rng.choice(cand,budget,False)
   methods={'attention':attention,'random':random}
   try: methods['gnnexplainer']=gnnexplainer_edges(m,X,E,W,qnode,budget,100,cand)[0]
   except Exception as e: print('GNNExplainer skipped:',dn,fold,qi,e)
   try: methods['pgexplainer']=pgexplainer_edges(m,X,E,W,list(range(min(20,len(G))))+[qnode],budget,30,cand)[0]
   except Exception as e: print('PGExplainer skipped:',dn,fold,qi,e)
   for name,edges in methods.items():
    fp,fm=eval_subset(m,X.cpu().numpy(),E.cpu(),W.cpu(),qnode,np.asarray(edges));rows.append({'dataset':dn,'fold':fold,'test_local_index':int(qi),'method':name,'fid_plus':fp,'fid_minus':fm,'edges':' '.join(map(str,np.asarray(edges).tolist()))})
pd.DataFrame(rows).to_csv('outputs/tables/graph_explanation_comparison.csv',index=False)
