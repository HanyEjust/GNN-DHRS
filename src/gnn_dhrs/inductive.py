import numpy as np, torch
from sklearn.metrics.pairwise import cosine_similarity

@torch.no_grad()
def build_original_reference_h1(model,X_original,X_graph,k=10):
    """Compute first-layer representations for original outer-training patients without altering the trained graph."""
    dev=next(model.parameters()).device; model.eval(); Xg=torch.as_tensor(X_graph,dtype=torch.float32,device=dev); refs=[]
    for q in X_original:
        s=cosine_similarity(q.reshape(1,-1),X_graph)[0]; cand=np.where(s>0)[0]; nbr=cand[np.argsort(-s[cand])[:k]]
        q0=torch.as_tensor(q,dtype=torch.float32,device=dev); sx=Xg[nbr]; w=torch.as_tensor(s[nbr],dtype=torch.float32,device=dev)
        h,_=model.g1.forward_query(q0,sx,w); h=torch.relu(model.b1(h.reshape(1,-1))).reshape(-1); refs.append(h.cpu().numpy())
    return np.asarray(refs)

@torch.no_grad()
def predict_queries(model,X_original,H1_original,X_query,k=10,return_attention=False):
    dev=next(model.parameters()).device; model.eval(); Xo=torch.as_tensor(X_original,dtype=torch.float32,device=dev); H=torch.as_tensor(H1_original,dtype=torch.float32,device=dev); probs=[]; atts=[]
    for q in X_query:
        s=cosine_similarity(q.reshape(1,-1),X_original)[0]; cand=np.where(s>0)[0]; nbr=cand[np.argsort(-s[cand])[:k]]; w=torch.as_tensor(s[nbr],dtype=torch.float32,device=dev); q0=torch.as_tensor(q,dtype=torch.float32,device=dev)
        p,a=model.inductive_probability(q0,Xo[nbr],H[nbr],w); probs.append(float(p.cpu())); atts.append((nbr,a[:-1].mean(1).cpu().numpy()))
    return (np.asarray(probs),atts) if return_attention else np.asarray(probs)
