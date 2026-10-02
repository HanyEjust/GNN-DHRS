"""Full outer-test robustness stress tests with fold-specific frozen GNN-DHRS models."""
import sys,os,numpy as np,pandas as pd,torch
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold,train_test_split
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.train import train_gnn,train_fixed_epochs,predict
from gnn_dhrs.inductive import build_original_reference_h1,predict_queries
from gnn_dhrs.metrics import evaluate,select_threshold
from gnn_dhrs.robustness import add_missingness,add_gaussian_noise,prevalence_subsample
cfg=load_config();rows=[]
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(10,shuffle=True,random_state=11)
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);R,T=pp.transform(ds.X.iloc[tr]),pp.transform(ds.X.iloc[te]);G,gy,_,_=dhrs(R,ds.y[tr],11,1.0,pp.onehot_groups);ei,ew=directed_cosine_knn(G,10);idx=np.arange(len(G));it,iv=train_test_split(idx,test_size=.12,stratify=gy,random_state=11);vm=np.zeros(len(G),bool);vm[iv]=1;m0,ep=train_gnn(G,gy,ei,ew,vm,cfg,11,return_best_epoch=True);th=select_threshold(gy[iv],predict(m0,G,ei,ew)[iv],cfg['threshold']['candidates']);m=train_fixed_epochs(G,gy,ei,ew,cfg,11,ep);H=build_original_reference_h1(m,R,G,10)
  def rec(kind,level,Xq,yq): rows.append({'dataset':dn,'fold':fold,'stress':kind,'level':level,**evaluate(yq,predict_queries(m,R,H,Xq,10),th)})
  for r in cfg['robustness']['missingness']: rec('missingness',r,add_missingness(T,r,fold),ds.y[te])
  for r in cfg['robustness']['gaussian_noise_sigma']: rec('noise',r,add_gaussian_noise(T,r,fold),ds.y[te])
  for r in cfg['robustness']['prevalence_multiplier']:
   ix=prevalence_subsample(ds.y[te],r,fold);rec('prevalence',r,T[ix],ds.y[te][ix])
  # edge corruption: replace a fraction of each query's selected references before inference
  from sklearn.metrics.pairwise import cosine_similarity
  dev=next(m.parameters()).device;rng=np.random.default_rng(fold)
  for rho in cfg['robustness']['edge_corruption']:
   ps=[]
   for q in T:
    s=cosine_similarity(q.reshape(1,-1),R)[0];nbr=np.where(s>0)[0];nbr=nbr[np.argsort(-s[nbr])[:10]].copy();nrep=int(round(rho*len(nbr)))
    if nrep: nbr[rng.choice(len(nbr),nrep,False)]=rng.integers(0,len(R),nrep)
    w=torch.tensor(np.maximum(cosine_similarity(q.reshape(1,-1),R[nbr])[0],1e-8),dtype=torch.float32,device=dev);p,_=m.inductive_probability(torch.tensor(q,dtype=torch.float32,device=dev),torch.tensor(R[nbr],dtype=torch.float32,device=dev),torch.tensor(H[nbr],dtype=torch.float32,device=dev),w);ps.append(float(p.detach().cpu()))
   rows.append({'dataset':dn,'fold':fold,'stress':'edge_corruption','level':rho,**evaluate(ds.y[te],np.array(ps),th)})
pd.DataFrame(rows).to_csv('outputs/tables/robustness.csv',index=False)
