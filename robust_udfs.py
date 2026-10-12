import math
import bisect
from typing import Optional

RESIDUAL_GRID_MIN = [
    -20.0, -15.0, -10.0, -8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 45.0, 60.0,
]

def nearest_residual_threshold_py(x: Optional[float]) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if x <= RESIDUAL_GRID_MIN[0]:
        return float(RESIDUAL_GRID_MIN[0])
    if x >= RESIDUAL_GRID_MIN[-1]:
        return float(RESIDUAL_GRID_MIN[-1])
    i = bisect.bisect_right(RESIDUAL_GRID_MIN, float(x)) - 1
    return float(RESIDUAL_GRID_MIN[max(0, i)])
