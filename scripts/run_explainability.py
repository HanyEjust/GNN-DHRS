"""End-to-end Kernel-SHAP and attention/edge-ablation experiment.
Runs fold-specific models from raw data, uses training-only SHAP backgrounds, and never uses outer-test labels to rank edges.
"""
import sys,os,numpy as np,pandas as pd,torch
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.train import train_gnn,train_fixed_epochs,predict
from gnn_dhrs.inductive import build_original_reference_h1,predict_queries
from gnn_dhrs.metrics import evaluate,select_threshold
from gnn_dhrs.explain import kernel_shap,stratified_indices,stability
cfg=load_config();os.makedirs('outputs/tables',exist_ok=True);os.makedirs('outputs/explanations',exist_ok=True)
shap_rows=[];edge_rows=[];rank_sets={}
for dn,spec in cfg['datasets'].items():
 ds=load_dataset(dn,spec);cv=StratifiedKFold(cfg['cv']['outer_folds'],shuffle=True,random_state=cfg['cv']['split_seed']);rank_sets[dn]=[]
 for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
  # validation-derived epoch/threshold
  a,v=train_test_split(np.arange(len(tr)),test_size=.15,stratify=ds.y[tr],random_state=11); ppv=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr[a]],ds.y[tr[a]]);A=ppv.transform(ds.X.iloc[tr[a]]);V=ppv.transform(ds.X.iloc[tr[v]]);D,dy,_,_=dhrs(A,ds.y[tr[a]],11,1.0,ppv.onehot_groups);ei,ew=directed_cosine_knn(D,10);idx=np.arange(len(D));it,iv=train_test_split(idx,test_size=.12,stratify=dy,random_state=11);vm=np.zeros(len(D),bool);vm[iv]=1;m0,ep=train_gnn(D,dy,ei,ew,vm,cfg,11,return_best_epoch=True);th=select_threshold(dy[iv],predict(m0,D,ei,ew)[iv],cfg['threshold']['candidates'])
  pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]);R=pp.transform(ds.X.iloc[tr]);T=pp.transform(ds.X.iloc[te]);G,gy,_,_=dhrs(R,ds.y[tr],11,1.0,pp.onehot_groups);gei,gew=directed_cosine_knn(G,10);m=train_fixed_epochs(G,gy,gei,gew,cfg,11,ep);H=build_original_reference_h1(m,R,G,10)
  # Kernel SHAP: background 50/class, explain 10/class (or available counts)
  bgidx=stratified_indices(ds.y[tr],cfg['explainability']['shap_background_per_class'],11+fold); exlocal=stratified_indices(ds.y[te],cfg['explainability']['shap_test_per_class'],101+fold);B=R[bgidx];Q=T[exlocal]
  pred_fn=lambda Z: predict_queries(m,R,H,np.asarray(Z),10)
  sv=np.asarray(kernel_shap(pred_fn,B,Q,cfg['explainability']['shap_nsamples']));
  if sv.ndim==3: sv=sv[-1]
  imp=np.mean(np.abs(sv),axis=0);order=np.argsort(-imp);rank_sets[dn].append(set(order[:5].tolist()))
  for j,name in enumerate(pp.feature_names_):shap_rows.append({'dataset':dn,'fold':fold,'feature':str(name),'mean_abs_shap':float(imp[j])})
  np.savez_compressed(f'outputs/explanations/{dn}_fold{fold}_shap.npz',shap=sv,X=Q,features=np.asarray(pp.feature_names_,dtype=str))
  # patient-wise attention edge ablation: high, low, random, equal budget=5
  probs,atts=predict_queries(m,R,H,T,10,True); strategies={'original':probs.copy(),'high':[],'low':[],'random':[]};rng=np.random.default_rng(11+fold)
  for qi,q in enumerate(T):
   nbr,alpha=atts[qi]; s=cosine_similarity(q.reshape(1,-1),R)[0][nbr]; budget=min(cfg['explainability']['edge_budget'],len(nbr));ordr=np.argsort(-alpha);sets={'high':ordr[:budget],'low':ordr[-budget:]}
   for mode,rm in sets.items():
    keep=np.ones(len(nbr),bool);keep[rm]=False
    if keep.any():
     p,_=m.inductive_probability(torch.tensor(q,dtype=torch.float32,device=next(m.parameters()).device),torch.tensor(R[nbr[keep]],dtype=torch.float32,device=next(m.parameters()).device),torch.tensor(H[nbr[keep]],dtype=torch.float32,device=next(m.parameters()).device),torch.tensor(s[keep],dtype=torch.float32,device=next(m.parameters()).device));strategies[mode].append(float(p.detach().cpu()))
    else: strategies[mode].append(.5)
   vals=[]
   for _ in range(cfg['explainability']['random_edge_repeats']):
    rm=rng.choice(len(nbr),budget,False);keep=np.ones(len(nbr),bool);keep[rm]=False
    if keep.any():
     p,_=m.inductive_probability(torch.tensor(q,dtype=torch.float32,device=next(m.parameters()).device),torch.tensor(R[nbr[keep]],dtype=torch.float32,device=next(m.parameters()).device),torch.tensor(H[nbr[keep]],dtype=torch.float32,device=next(m.parameters()).device),torch.tensor(s[keep],dtype=torch.float32,device=next(m.parameters()).device));vals.append(float(p.detach().cpu()))
   strategies['random'].append(float(np.mean(vals)) if vals else .5)
  for mode,pv in strategies.items(): edge_rows.append({'dataset':dn,'fold':fold,'strategy':mode,**evaluate(ds.y[te],np.asarray(pv),th)})
pd.DataFrame(shap_rows).to_csv('outputs/tables/kernel_shap_importance.csv',index=False);pd.DataFrame(edge_rows).to_csv('outputs/tables/edge_ablation_results.csv',index=False)
st=[]
for d,sets in rank_sets.items(): st.append({'dataset':d,'top5_jaccard':stability(sets)})
pd.DataFrame(st).to_csv('outputs/tables/shap_ranking_stability.csv',index=False)
