from enum import Enum
import torch

class SymmetryType(Enum):
    NONE = 0
    DISCRETE = 1
    AXIAL = 2


LINEMOD_SYMMETRIES = {
    3:  SymmetryType.AXIAL,     # can
    7: SymmetryType.DISCRETE,  # eggbox
    8: SymmetryType.AXIAL,     # glue
    9: SymmetryType.DISCRETE,  # holepuncher (approx)
}

SYMMETRIC_QUATS = {
    7: torch.tensor([
        [1., 0., 0., 0.],
        [0., 0., 0., 1.],   # 180° z
    ]),
    9: torch.tensor([
        [1., 0., 0., 0.],
        [0., 0., 0., 1.],
    ]),
}