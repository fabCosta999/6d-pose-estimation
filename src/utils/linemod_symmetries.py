from enum import Enum
import torch

class SymmetryType(Enum):
    NONE = 0
    DISCRETE = 1


LINEMOD_SYMMETRIES = {
    7: SymmetryType.DISCRETE,  # eggbox
    8: SymmetryType.DISCRETE,  # glue
}

SYMMETRIC_QUATS = {
    7: torch.tensor([
        [1., 0., 0., 0.],
        [0., 0., 0., 1.],   # 180° z
    ]),
    8: torch.tensor([
        [1., 0., 0., 0.],
        [0., 0., 0., 1.],
    ]),
}