import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def make_tabular(name,seed=11):
    models={
      "LR":LogisticRegression(max_iter=3000,class_weight="balanced",random_state=seed),
      "DT":DecisionTreeClassifier(class_weight="balanced",random_state=seed),
      "RF":RandomForestClassifier(n_estimators=500,class_weight="balanced",n_jobs=-1,random_state=seed),
      "ExtraTrees":ExtraTreesClassifier(n_estimators=500,class_weight="balanced",n_jobs=-1,random_state=seed),
      "GB":GradientBoostingClassifier(random_state=seed),
      "MLP":MLPClassifier(hidden_layer_sizes=(128,64),early_stopping=True,max_iter=500,random_state=seed),
      "SVM":SVC(C=1,kernel="rbf",class_weight="balanced",probability=True,random_state=seed),
      "XGBoost-CW":XGBClassifier(n_estimators=500,max_depth=4,learning_rate=.03,subsample=.8,colsample_bytree=.8,eval_metric="logloss",random_state=seed,n_jobs=-1),
      "LightGBM-CW":LGBMClassifier(n_estimators=500,num_leaves=31,learning_rate=.03,class_weight="balanced",random_state=seed,verbosity=-1),
      "CatBoost-CW":CatBoostClassifier(iterations=500,depth=6,learning_rate=.03,auto_class_weights="Balanced",random_seed=seed,verbose=False)
    }
    return models[name]

def fit_with_imbalance(model,X,y):
    if model.__class__.__name__=="XGBClassifier":
        pos=max(1,np.sum(y==1)); neg=max(1,np.sum(y==0)); model.set_params(scale_pos_weight=neg/pos)
    model.fit(X,y); return model

def proba(model,X):
    if hasattr(model,"predict_proba"): return model.predict_proba(X)[:,1]
    z=model.decision_function(X); return 1/(1+np.exp(-z))
