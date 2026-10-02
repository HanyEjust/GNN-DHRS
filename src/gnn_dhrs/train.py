import copy, numpy as np, torch
from sklearn.metrics import f1_score
from .model import GNNDHRS
from .repro import set_seed

def class_weights(y):
    n=len(y); n0=max(1,np.sum(y==0)); n1=max(1,np.sum(y==1)); return torch.tensor([n/(2*n0),n/(2*n1)],dtype=torch.float32)
def weighted_bce(logits,y,w):
    per=torch.nn.functional.binary_cross_entropy_with_logits(logits,y.float(),reduction="none"); ww=torch.where(y>0,w[1],w[0]); return (per*ww).mean()
def train_gnn(X,y,ei,ew,val_mask,cfg,seed=11,loss_name="wbce"):
    set_seed(seed); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); X=torch.as_tensor(X,dtype=torch.float32,device=dev); y=torch.as_tensor(y,dtype=torch.long,device=dev); ei=ei.to(dev); ew=ew.to(dev); val_mask=torch.as_tensor(val_mask,dtype=torch.bool,device=dev); tr=~val_mask
    m=GNNDHRS(X.shape[1],cfg["model"]["hidden_dim"],cfg["model"]["heads"],cfg["model"]["dropout"],cfg["model"]["leaky_relu_slope"]).to(dev)
    tc=cfg["training"]; opt=torch.optim.AdamW(m.parameters(),lr=tc["lr"],weight_decay=tc["weight_decay"],betas=tuple(tc["betas"]),eps=tc["eps"]); sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=tc["scheduler_factor"],patience=tc["scheduler_patience"],min_lr=tc["min_lr"])
    w=class_weights(y[tr].cpu().numpy()).to(dev); best=-1; state=None; bad=0
    for epoch in range(tc["max_epochs"]):
        m.train(); opt.zero_grad(); z=m(X,ei,ew)
        loss=weighted_bce(z[tr],y[tr],w) if loss_name=="wbce" else torch.nn.functional.binary_cross_entropy_with_logits(z[tr],y[tr].float())
        loss.backward(); opt.step(); m.eval()
        with torch.no_grad(): pv=torch.sigmoid(m(X,ei,ew)[val_mask]).cpu().numpy(); yy=y[val_mask].cpu().numpy(); score=f1_score(yy,pv>=.5,zero_division=0)
        sch.step(score)
        if score>best+1e-8: best=score; state=copy.deepcopy(m.state_dict()); bad=0
        else: bad+=1
        if bad>=tc["early_stopping_patience"]: break
    if state is not None:m.load_state_dict(state)
    return m
@torch.no_grad()
def predict(m,X,ei,ew,attention=False):
    dev=next(m.parameters()).device; X=torch.as_tensor(X,dtype=torch.float32,device=dev); ei=ei.to(dev); ew=ew.to(dev); m.eval()
    if attention:
        z,att=m(X,ei,ew,True); return torch.sigmoid(z).cpu().numpy(),(att[0].cpu(),att[1].cpu())
    return torch.sigmoid(m(X,ei,ew)).cpu().numpy()
def train_fixed_epochs(X,y,ei,ew,cfg,seed,epochs,loss_name="wbce"):
    set_seed(seed); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); X=torch.as_tensor(X,dtype=torch.float32,device=dev); y=torch.as_tensor(y,dtype=torch.long,device=dev); ei=ei.to(dev); ew=ew.to(dev)
    m=GNNDHRS(X.shape[1],cfg["model"]["hidden_dim"],cfg["model"]["heads"],cfg["model"]["dropout"],cfg["model"]["leaky_relu_slope"]).to(dev); tc=cfg["training"]
    opt=torch.optim.AdamW(m.parameters(),lr=tc["lr"],weight_decay=tc["weight_decay"],betas=tuple(tc["betas"]),eps=tc["eps"]); w=class_weights(y.cpu().numpy()).to(dev)
    for _ in range(max(1,int(epochs))):
        m.train(); opt.zero_grad(); z=m(X,ei,ew); loss=weighted_bce(z,y,w) if loss_name=="wbce" else torch.nn.functional.binary_cross_entropy_with_logits(z,y.float()); loss.backward(); opt.step()
    return m
