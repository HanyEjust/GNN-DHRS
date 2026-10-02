from pathlib import Path
import yaml

def load_config(path="configs/paper.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(Path(path).resolve().parent.parent)
    return cfg
