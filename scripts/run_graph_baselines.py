import sys,numpy as np,pandas as pd,torch
sys.path.insert(0,'src')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from gnn_dhrs.config import load_config
from gnn_dhrs.data import load_dataset,FoldPreprocessor
from gnn_dhrs.dhrs import dhrs
from gnn_dhrs.graph import directed_cosine_knn
from gnn_dhrs.graph_baselines import GraphBaseline,focal_loss
from gnn_dhrs.train import class_weights,weighted_bce
from gnn_dhrs.metrics import evaluate
cfg=load_config();rows=[];variants=[('GCN-W','gcn','wbce'),('SAGE-W','sage','wbce'),('GAT-BCE','gat','bce'),('GAT-W','gat','wbce'),('GAT-FL','gat','focal'),('GAT-CB','gat','cb'),('GAT-CBF','gat','cbf')]
def fit(kind,loss,X,y,ei,ew):
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=GraphBaseline(X.shape[1],kind).to(dev);xx=torch.tensor(X,dtype=torch.float32,device=dev);yy=torch.tensor(y,dtype=torch.long,device=dev);ei=ei.to(dev);ew=ew.to(dev);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0005);w=class_weights(y).to(dev)
 for _ in range(120):
  m.train();o.zero_grad();z=m(xx,ei,ew)
  if loss=='wbce':L=weighted_bce(z,yy,w)
  elif loss=='bce':L=torch.nn.functional.binary_cross_entropy_with_logits(z,yy.float())
  elif loss=='focal':L=focal_loss(z,yy)
  elif loss=='cb':L=weighted_bce(z,yy,w)
  else:L=focal_loss(z,yy,w)
  L.backward();o.step()
 return m
def pred(m,X,ei,ew):
 dev=next(m.parameters()).device;m.eval();
 with torch.no_grad():return torch.sigmoid(m(torch.tensor(X,dtype=torch.float32,device=dev),ei.to(dev),ew.to(dev))).cpu().numpy()
for seed in cfg['cv']['seeds']:
    for dn,spec in cfg['datasets'].items():
        ds=load_dataset(dn,spec); cv=StratifiedKFold(10,shuffle=True,random_state=seed)
        for fold,(tr,te) in enumerate(cv.split(ds.X,ds.y),1):
            pp=FoldPreprocessor(.9,10).fit(ds.X.iloc[tr],ds.y[tr]); A,T=pp.transform(ds.X.iloc[tr]),pp.transform(ds.X.iloc[te]); D,y,_,_=dhrs(A,ds.y[tr],seed,1.0,pp.onehot_groups); ei,ew=directed_cosine_knn(D,10)
            for name,kind,loss in variants:
                m=fit(kind,loss,D,y,ei,ew); ps=[]
                for q in T:
                    s=cosine_similarity(q.reshape(1,-1),D)[0]; nbr=np.where(s>0)[0]; nbr=nbr[np.argsort(-s[nbr])[:10]]; qe=torch.tensor([nbr,[len(D)]*len(nbr)],dtype=torch.long); qw=torch.tensor(s[nbr],dtype=torch.float32); ps.append(pred(m,np.vstack([D,q]),torch.cat([ei,qe],1),torch.cat([ew,qw]))[-1])
                rows.append({'dataset':dn,'seed':seed,'fold':fold,'model':name,**evaluate(ds.y[te],np.array(ps),.5)})
pd.DataFrame(rows).to_csv('outputs/tables/graph_baselines.csv',index=False)
