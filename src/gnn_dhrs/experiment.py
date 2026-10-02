from pathlib import Path
import copy, numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from .data import load_dataset,FoldPreprocessor
from .dhrs import dhrs
from .graph import directed_cosine_knn,inductive_edges
from .train import train_gnn,train_fixed_epochs,predict
from .metrics import evaluate,select_threshold
from .inductive import build_original_reference_h1,predict_queries

def _combined_train_val_graph(Xtr,Xv,k):
    ei,ew=directed_cosine_knn(Xtr,k); n=len(Xtr); src=[];dst=[];ww=[]
    from sklearn.metrics.pairwise import cosine_similarity
    S=cosine_similarity(Xv,Xtr)
    for q in range(len(Xv)):
        cand=np.where(S[q]>0)[0]; nbr=cand[np.argsort(-S[q,cand])[:k]]; src+=nbr.tolist(); dst += [n+q]*len(nbr); ww += S[q,nbr].tolist()
    if src:
        ei=torch.cat([ei,torch.tensor([src,dst],dtype=torch.long)],1); ew=torch.cat([ew,torch.tensor(ww,dtype=torch.float32)])
    return ei,ew

def predict_inductive(model,Xtrain_real,Xtest,train_graph_ei,train_graph_ew,k):
    ps=[]
    for q in range(len(Xtest)):
        qei,qew=inductive_edges(Xtrain_real,Xtest[q],k); n=len(Xtrain_real)
        # model was trained on DHRS graph, but test reference is original training only. Caller supplies model training graph and original rows separately is not index-compatible.
        # For exact manuscript protocol, final training array places original rows first; synthetic rows follow. q edges point to first n original rows.
        ei=torch.cat([train_graph_ei,qei],1); ew=torch.cat([train_graph_ew,qew]); X=np.vstack([model._train_X,Xtest[q:q+1]])
        ps.append(predict(model,X,ei,ew)[-1])
    return np.asarray(ps)

def run_nested_dataset(name,spec,cfg,seed=11):
    ds=load_dataset(name,spec); outer=StratifiedKFold(cfg["cv"]["outer_folds"],shuffle=True,random_state=cfg["cv"]["split_seed"]); rows=[]; preds=[]
    for fold,(otr,ote) in enumerate(outer.split(ds.X,ds.y),1):
        Xo=ds.X.iloc[otr].reset_index(drop=True); yo=ds.y[otr]; inner=StratifiedKFold(cfg["cv"]["inner_folds"],shuffle=True,random_state=cfg["cv"]["split_seed"])
        # select feature count using pooled inner validation F1; default architecture fixed to manuscript-selected values
        fc_scores={}; fc_payload={}
        for fc in cfg["preprocessing"]["feature_counts"]:
            yv=[];pv=[]; epoch_list=[]
            for itr,iva in inner.split(Xo,yo):
                pp=FoldPreprocessor(cfg["preprocessing"]["corr_threshold"],fc).fit(Xo.iloc[itr],yo[itr]); A=pp.transform(Xo.iloc[itr]); V=pp.transform(Xo.iloc[iva]); D,dy,_,_=dhrs(A,yo[itr],seed,cfg["dhrs"]["oversampling_ratio"],pp.onehot_groups,cfg["dhrs"]["borderline_k_neighbors"],cfg["dhrs"]["borderline_m_neighbors"],cfg["dhrs"]["enn_neighbors"])
                ei,ew=_combined_train_val_graph(D,V,cfg["graph"]["k"]); C=np.vstack([D,V]); cy=np.r_[dy,yo[iva]]; vm=np.zeros(len(cy),bool); vm[len(D):]=True
                m=train_gnn(C,cy,ei,ew,vm,cfg,seed); p=predict(m,C,ei,ew)[len(D):]; yv.append(yo[iva]);pv.append(p); epoch_list.append(cfg["training"]["max_epochs"]-cfg["training"]["early_stopping_patience"])
            yy=np.concatenate(yv);ppp=np.concatenate(pv); th=select_threshold(yy,ppp,cfg["threshold"]["candidates"]); fc_scores[str(fc)]=f1_score(yy,ppp>=th,zero_division=0); fc_payload[str(fc)]=(th,int(np.median(epoch_list)))
        best=max(fc_scores,key=fc_scores.get); fc="all" if best=="all" else int(best); threshold,epochs=fc_payload[best]
        prep=FoldPreprocessor(cfg["preprocessing"]["corr_threshold"],fc).fit(Xo,yo); A=prep.transform(Xo); T=prep.transform(ds.X.iloc[ote]); D,dy,syn,audit=dhrs(A,yo,seed,cfg["dhrs"]["oversampling_ratio"],prep.onehot_groups,cfg["dhrs"]["borderline_k_neighbors"],cfg["dhrs"]["borderline_m_neighbors"],cfg["dhrs"]["enn_neighbors"])
        ei,ew=directed_cosine_knn(D,cfg["graph"]["k"]); m=train_fixed_epochs(D,dy,ei,ew,cfg,seed,epochs)
        H1=build_original_reference_h1(m,A,D,cfg["graph"]["k"])
        ptest=predict_queries(m,A,H1,T,cfg["graph"]["k"])
        met=evaluate(ds.y[ote],ptest,threshold,cfg["calibration"]["ece_bins"]); rows.append({"dataset":name,"seed":seed,"fold":fold,"feature_count":fc,"threshold":threshold,**audit,**met})
        preds.extend({"dataset":name,"seed":seed,"fold":fold,"row_index":int(idx),"y":int(y),"p":float(p),"threshold":threshold} for idx,y,p in zip(ote,ds.y[ote],ptest))
    return pd.DataFrame(rows),pd.DataFrame(preds)

def run_all(cfg,seeds=None):
    out=Path(cfg["project"]["output_dir"]); (out/"tables").mkdir(parents=True,exist_ok=True);(out/"predictions").mkdir(parents=True,exist_ok=True); M=[];P=[]
    for seed in (seeds or cfg["cv"]["seeds"]):
        for n,s in cfg["datasets"].items():
            m,p=run_nested_dataset(n,s,cfg,seed);M.append(m);P.append(p)
    M=pd.concat(M,ignore_index=True);P=pd.concat(P,ignore_index=True);M.to_csv(out/"tables/nested_cv_fold_metrics.csv",index=False);P.to_csv(out/"predictions/outer_test_predictions.csv",index=False);return M,P
