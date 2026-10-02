import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from .model import GNNDHRS

class MLP(nn.Module):
    def __init__(self,d,h=128,drop=.3): super().__init__(); self.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(drop),nn.Linear(h,h),nn.ReLU(),nn.Dropout(drop),nn.Linear(h,1))
    def forward(self,x,*args): return self.net(x).squeeze(-1)

class UniformGraph(nn.Module):
    def __init__(self,d,h=128,drop=.3): super().__init__();self.l1=nn.Linear(d,h);self.l2=nn.Linear(h,h);self.out=nn.Linear(h,1);self.drop=drop
    def agg(self,x,ei):
        s,d=ei; n=len(x); out=torch.zeros_like(x);out.index_add_(0,d,x[s]);deg=torch.zeros(n,device=x.device);deg.index_add_(0,d,torch.ones(len(d),device=x.device));return (out+x)/(deg[:,None]+1)
    def forward(self,x,ei,ew=None):
        h=F.relu(self.l1(self.agg(x,ei)));h=F.dropout(h,self.drop,self.training);h=F.relu(self.l2(self.agg(h,ei)));return self.out(h).squeeze(-1)

def make_variant(name,input_dim,cfg):
    if name=='no_graph_mlp': return MLP(input_dim,cfg['model']['hidden_dim'],cfg['model']['dropout'])
    if name=='uniform_aggregation': return UniformGraph(input_dim,cfg['model']['hidden_dim'],cfg['model']['dropout'])
    m=GNNDHRS(input_dim,cfg['model']['hidden_dim'],cfg['model']['heads'],cfg['model']['dropout'],cfg['model']['leaky_relu_slope'])
    return m
