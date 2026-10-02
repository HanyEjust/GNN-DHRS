from pathlib import Path
import joblib, torch

def save_bundle(path,model,preprocessor,X_original,X_graph,y_graph,H1,threshold,metadata=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),path.with_suffix('.pt'))
    joblib.dump({'preprocessor':preprocessor,'X_original':X_original,'X_graph':X_graph,'y_graph':y_graph,'H1':H1,'threshold':threshold,'metadata':metadata or {}},path.with_suffix('.joblib'))
