import trimesh
import torch
from pathlib import Path

def load_linemod_models(models_dir, device="cpu"):
    models = {}
    for ply in Path(models_dir).glob("obj_*.ply"):
        obj_id = int(ply.stem.split("_")[1])
        mesh = trimesh.load(ply, process=False)
        pts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        models[obj_id] = pts
    return models