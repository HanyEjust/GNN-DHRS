import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

class SimilarityGATLayer(nn.Module):
    """Multi-head GAT implementing alpha ∝ A_ij exp(LeakyReLU(a^T[Wh_i||Wh_j]))."""
    def __init__(self,in_dim,out_per_head,heads=8,dropout=.3,slope=.2):
        super().__init__(); self.h=heads; self.d=out_per_head; self.dropout=dropout; self.slope=slope
        self.lin=nn.Linear(in_dim,heads*out_per_head,bias=False); self.att=nn.Parameter(torch.empty(heads,2*out_per_head)); self.reset_parameters()
    def reset_parameters(self): nn.init.xavier_uniform_(self.lin.weight); nn.init.xavier_uniform_(self.att)
    def forward_query(self,target_x,source_x,weights):
        """Aggregate a single unseen target from frozen source representations plus a self-loop."""
        zt=self.lin(target_x.reshape(1,-1)).view(1,self.h,self.d)[0]
        zs=self.lin(source_x).view(source_x.size(0),self.h,self.d)
        zall=torch.cat([zs,zt.unsqueeze(0)],0); ztarget=zt.expand(zall.size(0),-1,-1)
        pair=torch.cat([ztarget,zall],dim=-1); u=F.leaky_relu((pair*self.att).sum(-1),negative_slope=self.slope)
        ww=torch.cat([weights,torch.ones(1,device=weights.device)]); logits=u+torch.log(ww.clamp_min(1e-12)).unsqueeze(-1)
        alpha=torch.softmax(logits,dim=0); return (zall*alpha.unsqueeze(-1)).sum(0).reshape(-1),alpha

    def forward(self,x,edge_index,edge_weight,return_attention=False):
        n=x.size(0); device=x.device
        # add self-loops with Aii=1
        loops=torch.arange(n,device=device); lei=torch.stack([loops,loops]); edge_index=torch.cat([edge_index,lei],1); edge_weight=torch.cat([edge_weight,torch.ones(n,device=device)])
        z=self.lin(x).view(n,self.h,self.d); src,dst=edge_index
        pair=torch.cat([z[dst],z[src]],dim=-1); u=F.leaky_relu((pair*self.att).sum(-1),negative_slope=self.slope)
        logits=u+torch.log(edge_weight.clamp_min(1e-12)).unsqueeze(-1)
        alpha=softmax(logits,dst,num_nodes=n); alpha=F.dropout(alpha,p=self.dropout,training=self.training)
        msg=z[src]*alpha.unsqueeze(-1); out=torch.zeros((n,self.h,self.d),device=x.device,dtype=msg.dtype); out.index_add_(0,dst,msg); out=out.reshape(n,self.h*self.d)
        return (out,(edge_index,alpha)) if return_attention else out

class GNNDHRS(nn.Module):
    def __init__(self,input_dim,hidden=128,heads=8,dropout=.3,slope=.2):
        super().__init__(); assert hidden%heads==0; d=hidden//heads; self.drop=dropout
        self.g1=SimilarityGATLayer(input_dim,d,heads,dropout,slope); self.b1=nn.BatchNorm1d(hidden)
        self.g2=SimilarityGATLayer(hidden,d,heads,dropout,slope); self.b2=nn.BatchNorm1d(hidden); self.out=nn.Linear(hidden,1); self.reset_output()
    def reset_output(self): nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)
    def encode(self,x,ei,ew,return_attention=False):
        h=self.g1(x,ei,ew); h=F.dropout(F.relu(self.b1(h)),self.drop,self.training)
        if return_attention:
            h,(e,a)=self.g2(h,ei,ew,True); h=F.dropout(F.relu(self.b2(h)),self.drop,self.training); return h,(e,a)
        h=self.g2(h,ei,ew); return F.dropout(F.relu(self.b2(h)),self.drop,self.training)
    def inductive_probability(self,q0,neighbor_x0,neighbor_h1,weights):
        """Strict two-layer inference for one query from frozen original-training references."""
        h1,_=self.g1.forward_query(q0,neighbor_x0,weights); h1=F.relu(self.b1(h1.reshape(1,-1))).reshape(-1)
        h2,alpha=self.g2.forward_query(h1,neighbor_h1,weights); h2=F.relu(self.b2(h2.reshape(1,-1))).reshape(-1)
        return torch.sigmoid(self.out(h2)).squeeze(),alpha

    def forward(self,x,ei,ew,return_attention=False):
        if return_attention:
            h,att=self.encode(x,ei,ew,True); return self.out(h).squeeze(-1),att
        return self.out(self.encode(x,ei,ew)).squeeze(-1)
