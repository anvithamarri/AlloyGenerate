import numpy as np
from pymatgen.core import Structure


class CIFScorer:
    """
    Abstract base class for CIF scorers.
    A scorer provides a heuristic score for a completed CIF string.
    Higher scores are better.
    """
    def score(self, cif: str) -> float:
        raise NotImplementedError


class HeuristicPhysicalScorer(CIFScorer):
    """
    Heuristic scorer that rewards realistic density, cubic-like cells,
    appropriate atomic spacing, and high-symmetry space groups.

    Used as the MCTS eval_function in app.py to steer generation toward
    physically reasonable intermetallic candidates during rollout.

    Scoring terms:
      - Density error vs. target (default 2.16 g/cm³)
      - Cubic shape regularity (penalizes a≠b≠c)
      - Lattice angle regularity (penalizes deviation from 90°)
      - Maximum axis penalty (penalizes any axis > 7.0 Å)
      - Minimum interatomic distance penalty (penalizes < 2.0 Å)
      - Symmetry bonus for Fm-3m / No. 225 (rock-salt / FCC prototype)
    """

    def __init__(self, target_density: float = 2.16):
        self.target_rho = target_density

    def score(self, cif: str) -> float:
        try:
            struct = Structure.from_str(cif, fmt="cif")

            # 1. Density error
            rho_error = abs(struct.density - self.target_rho)

            # 2. Cubic shape (a=b=c)
            a, b, c = struct.lattice.abc
            shape_error = (abs(a - b) + abs(b - c) + abs(a - c)) * 15.0

            # 3. Angle penalty (all angles should be 90°)
            angles = struct.lattice.angles
            angle_error = sum([abs(ang - 90.0) for ang in angles]) * 2.0

            # 4. Axis length penalty
            axis_penalty = 20.0 if max(a, b, c) > 7.0 else 0.0

            # 5. Minimum interatomic distance
            min_dist = (
                struct.get_all_neighbors(1.0)[0][0].nn_distance
                if len(struct) > 1 else 0
            )
            dist_penalty = 10.0 if min_dist < 2.0 else 0.0

            # 6. Symmetry bonus (Fm-3m, No. 225)
            sym_bonus = 10.0 if struct.get_space_group_info()[1] == 225 else 0.0

            return -(rho_error + shape_error + angle_error + axis_penalty + dist_penalty) + sym_bonus

        except Exception:
            return -100.0