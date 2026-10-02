import numpy as np

def reliability_bins(y,p,bins=10):
    y=np.asarray(y);p=np.asarray(p); edges=np.linspace(0,1,bins+1); rows=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        m=(p>=lo)&(p<(hi if hi<1 else hi+1e-12))
        if m.any(): rows.append({'bin':i,'lower':lo,'upper':hi,'n':int(m.sum()),'mean_probability':float(p[m].mean()),'observed_rate':float(y[m].mean())})
    return rows
