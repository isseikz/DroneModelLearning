"""Model of the propellers."""

import numpy as np


class Propeller(object):
    """Class for Propeller."""

    def __init__(self, position, direction, k_F=1.0, k_M=1.0):
        """Initialization."""
        if len(position) != 3:
            raise ValueError("position vector must be 3 dimensional.")
        if len(direction) != 3:
            raise ValueError("thrust direction must be 3 dimensional.")

        super(Propeller, self).__init__()
        self.position = position           # Alignment, body frame [m]
        self.direction = direction         # Thrust direction, body frame [m]
        self.F_func = LinearFunction(k_F)  # Thrust response to the input
        self.M_func = LinearFunction(k_M)  # Torque response to the input

    def linear_func(self, input, k):
        """Thrust model (linear)."""
        return k * input

    def get_force_torque(self, input):
        """Calculate the response of the prop from the input."""
        F = np.dot(self.F_func.get_value(input), self.direction)
        M = np.dot(self.M_func.get_value(input), self.direction) + np.cross(self.position.T, F.T).T
        return np.vstack((F, M))

    def set_thrust_response_function(self, function, *args):
        self.F_func = function(*args)

    def set_torque_response_function(self, function, *args):
        self.M_func = function(*args)


class LinearFunction(object):
    """docstring for LinearFunction."""
    def __init__(self, coefficient):
        super(LinearFunction, self).__init__()
        self.k = coefficient

    def get_value(self, input):
        return self.k * input


if __name__ == '__main__':
    p = Propeller(np.array([0,0,0]), np.array([0,1,0]))
    print(p.get_force_torque(3.0))
