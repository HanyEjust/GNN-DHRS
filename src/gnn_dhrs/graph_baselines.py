import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv
class GraphBaseline(nn.Module):
 def __init__(self,d,kind='gcn',h=128,heads=8,drop=.3):
  super().__init__();self.kind=kind;self.drop=drop
  if kind=='gcn':self.a=GCNConv(d,h);self.b=GCNConv(h,h)
  elif kind=='sage':self.a=SAGEConv(d,h);self.b=SAGEConv(h,h)
  else:self.a=GATv2Conv(d,h//heads,heads=heads,dropout=drop);self.b=GATv2Conv(h,h//heads,heads=heads,dropout=drop)
  self.out=nn.Linear(h,1)
 def forward(self,x,ei,ew=None):
  if self.kind=='gcn':h=self.a(x,ei,ew);h=F.relu(h);h=F.dropout(h,self.drop,self.training);h=self.b(h,ei,ew)
  else:h=self.a(x,ei);h=F.relu(h);h=F.dropout(h,self.drop,self.training);h=self.b(h,ei)
  return self.out(F.relu(h)).squeeze(-1)
def focal_loss(logits,y,alpha=None,gamma=2.0):
 p=torch.sigmoid(logits);pt=torch.where(y>0,p,1-p);a=1.0 if alpha is None else torch.where(y>0,torch.as_tensor(alpha[1],device=y.device),torch.as_tensor(alpha[0],device=y.device));return (-a*(1-pt).pow(gamma)*torch.log(pt.clamp_min(1e-8))).mean()
