from dataclasses import dataclass
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import f_classif

@dataclass
class Dataset:
    name: str; X: pd.DataFrame; y: np.ndarray; ids: pd.Series|None

def load_dataset(name, spec):
    df=pd.read_csv(spec["path"])
    target=spec.get("target","stroke")
    ids=None
    for c in spec.get("id_candidates",[]):
        if c in df.columns: ids=df[c].copy(); df=df.drop(columns=[c]); break
    y=df.pop(target).astype(int).to_numpy()
    return Dataset(name,df,y,ids)

def integrity_audit(ds):
    z=ds.X.copy(); z["__y__"]=ds.y
    exact=z.duplicated(keep=False); redundant=z.duplicated(keep="first")
    pred_dup=ds.X.duplicated(keep=False)
    conflict_groups=0
    if pred_dup.any():
        tmp=ds.X.copy(); tmp["__y__"]=ds.y
        conflict_groups=int((tmp.groupby(list(ds.X.columns),dropna=False)["__y__"].nunique()>1).sum())
    return {"dataset":ds.name,"n":len(ds.y),"stroke":int(ds.y.sum()),"nonstroke":int((1-ds.y).sum()),
            "exact_duplicate_rows":int(exact.sum()),"redundant_rows":int(redundant.sum()),"conflicting_groups":conflict_groups}

class FoldPreprocessor:
    def __init__(self,corr_threshold=.90,feature_count=10): self.corr_threshold=corr_threshold; self.feature_count=feature_count
    def fit(self,X,y):
        self.num=[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cat=[c for c in X.columns if c not in self.num]
        num=Pipeline([("imp",SimpleImputer(strategy="mean")),("scale",StandardScaler())])
        cat=Pipeline([("imp",SimpleImputer(strategy="constant",fill_value="Unknown")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
        self.ct=ColumnTransformer([("num",num,self.num),("cat",cat,self.cat)],sparse_threshold=0).fit(X)
        Z=np.asarray(self.ct.transform(X),dtype=float); names=np.asarray(self.ct.get_feature_names_out())
        # remove later member of highly correlated pairs using training data only
        corr=np.corrcoef(Z,rowvar=False); keep=np.ones(Z.shape[1],dtype=bool)
        if Z.shape[1]>1:
            for j in range(Z.shape[1]):
                if not keep[j]: continue
                bad=np.where(np.abs(corr[j,j+1:])>self.corr_threshold)[0]+j+1
                keep[bad]=False
        self.corr_keep=np.where(keep)[0]; Z=Z[:,self.corr_keep]; names=names[self.corr_keep]
        f,_=f_classif(Z,y); f=np.nan_to_num(f,nan=-np.inf,posinf=np.finfo(float).max)
        order=np.argsort(-f)
        n=len(order) if self.feature_count=="all" else min(int(self.feature_count),len(order))
        self.sel=order[:n]; self.feature_names_=names[self.sel]
        # one-hot groups after selection, for categorical projection after SMOTE
        self.onehot_groups=[]
        for raw in self.cat:
            idx=[i for i,nm in enumerate(self.feature_names_) if nm.startswith(f"cat__{raw}_")]
            if len(idx)>1:self.onehot_groups.append(idx)
        return self
    def transform(self,X): return np.asarray(self.ct.transform(X),dtype=float)[:,self.corr_keep][:,self.sel]
    def fit_transform(self,X,y): return self.fit(X,y).transform(X)
