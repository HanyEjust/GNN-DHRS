import numpy as np, torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def _topk(S,k,positive=False):
    S=np.asarray(S,float).copy(); np.fill_diagonal(S,-np.inf); src=[];dst=[];w=[]
    for t in range(len(S)):
        cand=np.arange(len(S)); cand=cand[cand!=t]
        if positive: cand=cand[S[t,cand]>0]
        nbr=cand[np.argsort(-S[t,cand])[:k]]
        src+=nbr.tolist(); dst += [t]*len(nbr); w += np.maximum(S[t,nbr],1e-8).tolist()
    return torch.tensor([src,dst],dtype=torch.long),torch.tensor(w,dtype=torch.float32)

def graph_from_definition(X,definition='cosine_knn',k=10,threshold=None):
    X=np.asarray(X,float); definition=definition.lower()
    if definition=='cosine_knn': return _topk(cosine_similarity(X),k,True)
    if definition=='euclidean_knn':
        D=euclidean_distances(X); S=1/(1+D); return _topk(S,k)
    if definition=='gower_knn':
        # On the fully transformed fold-specific matrix, scaled Manhattan distance is the
        # continuous analogue of Gower distance; one-hot categorical blocks retain 0/1 mismatch costs.
        D=manhattan_distances(X)/max(1,X.shape[1]); return _topk(1/(1+D),k)
    if definition=='mutual_cosine_knn':
        S=cosine_similarity(X); ei,ew=_topk(S,k,True); s,d=ei.numpy(); pairs={(int(a),int(b)):float(c) for a,b,c in zip(s,d,ew.numpy())}; keep=[i for i,(a,b) in enumerate(zip(s,d)) if (int(b),int(a)) in pairs]
        return ei[:,keep],ew[keep]
    if definition=='cosine_threshold':
        S=cosine_similarity(X); np.fill_diagonal(S,-np.inf)
        if threshold is None:
            vals=[]
            for i in range(len(X)):
                row=S[i][S[i]>0]
                if len(row): vals.extend(np.sort(row)[-min(k,len(row)):].tolist())
            threshold=float(np.min(vals)) if vals else 0.0
        d,s=np.where(S>=threshold); # row is target, col is source
        return torch.tensor([s,d],dtype=torch.long),torch.tensor(S[d,s],dtype=torch.float32)
    raise ValueError(definition)

def graph_stats(edge_index,y,synthetic=None):
    y=np.asarray(y); s=edge_index[0].cpu().numpy(); d=edge_index[1].cpu().numpy(); n=len(y)
    hom=float(np.mean(y[s]==y[d])) if len(s) else np.nan
    minority=np.where(y==1)[0]; mh=float(np.mean([np.sum((d==i)&(y[s]==1)) for i in minority])) if len(minority) else np.nan
    if len(s):
        A=coo_matrix((np.ones(len(s)*2),(np.r_[s,d],np.r_[d,s])),shape=(n,n)); comps=int(connected_components(A,directed=False)[0])
    else: comps=n
    out={'nodes':n,'edges':len(s),'minority_homophilic_degree':mh,'homophily':hom,'components':comps}
    if synthetic is not None:
        z=np.asarray(synthetic,bool); out.update(real_real=int(np.sum(~z[s]&~z[d])),real_synthetic=int(np.sum(z[s]^z[d])),synthetic_synthetic=int(np.sum(z[s]&z[d])))
    return out
