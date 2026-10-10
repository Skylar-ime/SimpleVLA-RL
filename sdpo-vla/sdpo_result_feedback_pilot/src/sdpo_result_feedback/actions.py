from __future__ import annotations

import numpy as np

ACTION_LABELS = ("A", "B", "C", "D", "E")
ACTION_VECTORS = {
    "A": np.array([0.4, 0.0, 0.0], dtype=np.float32),
    "B": np.array([-0.4, 0.0, 0.0], dtype=np.float32),
    "C": np.array([0.0, 0.4, 0.0], dtype=np.float32),
    "D": np.array([0.0, -0.4, 0.0], dtype=np.float32),
    "E": np.array([0.0, 0.0, 0.0], dtype=np.float32),
}

ACTION_SCALE_M = 0.05
DECISION_SIM_STEPS = 4
RESULT_DEADBAND_M = 0.002


def action_vector(label: str) -> np.ndarray:
    if label not in ACTION_VECTORS:
        raise ValueError(f"unknown action label: {label}")
    return ACTION_VECTORS[label].copy()
