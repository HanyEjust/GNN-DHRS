import copy,numpy as np,torch
from torch import nn
from sklearn.metrics import f1_score
from .repro import set_seed
class FTTransformer(nn.Module):
    def __init__(self,d_in,d_token=64,n_heads=8,n_layers=3,dropout=.2):
        super().__init__();self.tokens=nn.Parameter(torch.empty(d_in,d_token));self.bias=nn.Parameter(torch.zeros(d_in,d_token));nn.init.xavier_uniform_(self.tokens)
        enc=nn.TransformerEncoderLayer(d_token,n_heads,dim_feedforward=4*d_token,dropout=dropout,batch_first=True,norm_first=True,activation="gelu");self.enc=nn.TransformerEncoder(enc,n_layers);self.cls=nn.Parameter(torch.zeros(1,1,d_token));self.head=nn.Sequential(nn.LayerNorm(d_token),nn.ReLU(),nn.Linear(d_token,1))
    def forward(self,x):
        t=x.unsqueeze(-1)*self.tokens.unsqueeze(0)+self.bias.unsqueeze(0); c=self.cls.expand(x.size(0),-1,-1);h=self.enc(torch.cat([c,t],1));return self.head(h[:,0]).squeeze(-1)
def train_ft(X,y,Xv,yv,seed=11,class_weight=True,epochs=200,patience=20):
    set_seed(seed);dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");X=torch.tensor(X,dtype=torch.float32,device=dev);y=torch.tensor(y,dtype=torch.float32,device=dev);Xv=torch.tensor(Xv,dtype=torch.float32,device=dev);yv=np.asarray(yv)
    m=FTTransformer(X.shape[1]).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=5e-4);pos=(len(y)/max(1,float(y.sum()))-1) if class_weight else 1.;crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device=dev));best=-1;state=None;bad=0
    for _ in range(epochs):
        m.train();opt.zero_grad();loss=crit(m(X),y);loss.backward();opt.step();m.eval()
        with torch.no_grad():p=torch.sigmoid(m(Xv)).cpu().numpy();f=f1_score(yv,p>=.5,zero_division=0)
        if f>best:best=f;state=copy.deepcopy(m.state_dict());bad=0
        else:bad+=1
        if bad>=patience:break
    m.load_state_dict(state);return m
@torch.no_grad()
def ft_predict(m,X):
    dev=next(m.parameters()).device;m.eval();return torch.sigmoid(m(torch.tensor(X,dtype=torch.float32,device=dev))).cpu().numpy()
