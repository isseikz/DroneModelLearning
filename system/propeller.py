"""Model of the propellers."""

import numpy as np


class Propeller(object):
    """Class for Propeller."""

    def __init__(self, position, direction, T_min, T_max):
        """Initialization."""
        if len(position) != 3:
            raise ValueError("position vector must be 3 dimensional.")
        if len(direction) != 3:
            raise ValueError("thrust direction must be 3 dimensional.")
        if len(T_min) != 1:
            raise ValueError("T_min must be a scaler")
        if len(T_max) != 1:
            raise ValueError("T_max must be a scaler.")

        super(Propeller, self).__init__()
        self.position = position    # Alignment, body frame [m]
        self.direction = direction  # Thrust direction, body frame [m]
        self.T_min = T_min          # Minimal thrust size [N]
        self.T_max = T_max          # Maximum thrust size [N]

    def linear_thrust(self, input, k):
        """Thrust model (linear)."""
        return k * input
