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

def add_metric(
    model_points,   # (N,3)
    R_pred, t_pred, # (3,3), (3,)
    R_gt,   t_gt,   # (3,3), (3,)
):
    # Transform points
    pts_pred = (R_pred @ model_points.T).T + t_pred
    pts_gt   = (R_gt   @ model_points.T).T + t_gt

    # Pairwise distances (N,N)
    dists = torch.cdist(pts_pred, pts_gt, p=2)

    return dists.mean()


