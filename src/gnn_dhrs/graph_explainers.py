"""Adapters for PyG GNNExplainer/PGExplainer on frozen GNN-DHRS graphs.
These functions operate on a standard graph containing a query node. They never use the query label for edge ranking.
"""
import torch, numpy as np
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer

class ProbabilityWrapper(torch.nn.Module):
    def __init__(self,model,edge_weight): super().__init__(); self.model=model; self.register_buffer('base_edge_weight',edge_weight)
    def forward(self,x,edge_index): return torch.sigmoid(self.model(x,edge_index,self.base_edge_weight))

def gnnexplainer_edges(model,x,edge_index,edge_weight,node_index,budget=5,epochs=100,candidate_edges=None):
    wrap=ProbabilityWrapper(model,edge_weight)
    ex=Explainer(model=wrap,algorithm=GNNExplainer(epochs=epochs),explanation_type='model',node_mask_type=None,edge_mask_type='object',model_config=dict(mode='binary_classification',task_level='node',return_type='probs'))
    out=ex(x=x,edge_index=edge_index,index=int(node_index)); score=out.edge_mask.detach().cpu().numpy(); cand=np.arange(len(score)) if candidate_edges is None else np.asarray(candidate_edges,int); return cand[np.argsort(-score[cand])[:budget]],score

def pgexplainer_edges(model,x,edge_index,edge_weight,node_indices,budget=5,epochs=30,candidate_edges=None):
    """Train PGExplainer only on training/reference nodes, then explain requested nodes."""
    wrap=ProbabilityWrapper(model,edge_weight); alg=PGExplainer(epochs=epochs,lr=0.003)
    ex=Explainer(model=wrap,algorithm=alg,explanation_type='phenomenon',edge_mask_type='object',model_config=dict(mode='binary_classification',task_level='node',return_type='probs'))
    wrap.eval();
    with torch.no_grad(): target=wrap(x,edge_index).detach()
    train_nodes=[int(i) for i in node_indices[:-1]] if len(node_indices)>1 else [int(node_indices[0])]
    for epoch in range(epochs):
        for i in train_nodes: alg.train(epoch,wrap,x,edge_index,target=target,index=i)
    q=int(node_indices[-1]); out=ex(x=x,edge_index=edge_index,target=target,index=q);score=out.edge_mask.detach().cpu().numpy();cand=np.arange(len(score)) if candidate_edges is None else np.asarray(candidate_edges,int);return cand[np.argsort(-score[cand])[:budget]],score
