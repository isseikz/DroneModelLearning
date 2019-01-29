"""6-Dimensional Dynamics model."""

import numpy as np
from scipy.integrate import ode


class dynamics(object):
    """Class for dynamics."""

    """
        x[0:3]    : position, inertial frame (m)
        x[3:6]    : velocity, inertial frame (m/s)
        x[6:10]   : quartanion[q0, q1, q2, q3] (non-dim)
        x[10:13]  : angular velocity, body frame (rad/s)
    """

    def __init__(self, x0, t0, dvdt, dwdt, dt):
        """Initialization."""
        if x0.shape[0] != 13 | x0.shape[1] != 1:
            raise ValueError(
                "The dimension of the initial state value must be 13."
            )
        if dvdt.shape[0] != 3 | dvdt.shape[1] != 1:
            raise ValueError(
                "Derivative of the velocity must be a 3 * 1 vector."
            )
        if dwdt.shape[0] != 3 | dwdt.shape[1] != 1:
            raise ValueError(
                "Derivative of the rotation must be a 3 * 1 vector."
            )

        super(dynamics, self).__init__()
        self.solver = ode(self.system).set_integrator('vode', method='bdf')
        self.solver.set_initial_value(x0, t0)
        self.solver.set_f_params((dvdt, dwdt))
        self.dt = dt

    def get_X(self):
        """Obtain the current state variable."""
        return self.solver.y

    def get_X_vel_rot(self):
        """Obtain the current velocity and rotation."""
        x_star = np.zeros((6, 1))
        x_star[0:3, 0] = self.solver.y[3:6, 0]
        x_star[3:6, 0] = self.solver.y[10:13, 0]
        return x_star

    def system(self, dvdt, dwdt):
        """Calculate dX/dt."""
        if dvdt.shape[0] != 3 | dvdt.shape[1] != 1:
            raise ValueError(
                "Derivative of the velocity should be a 3 * 1 vector."
            )
        if dwdt.shape[0] != 3 | dwdt.shape[1] != 1:
            raise ValueError(
                "Derivative of the rotation should be a 3 * 1 vector."
            )

        dxdt = np.zeros((13, 1))
        dxdt[0:3, 0] = self.velocity
        dxdt[3:6, 0] = dvdt
        dxdt[10:13, 0] = dwdt

        w = self.solver.y[10:13, 0].T
        dxdt[6:10, 0] = np.dot(
            np.array([
                [0, -w[0], -w[1], -w[2]],
                [w[0], 0, -w[1], -w[2]],
                [w[1], w[2], 0, -w[0]],
                [w[2], -w[1], w[0], 0]
            ]),
            self.attitude
        )
        return dxdt

    def simulate_onestep(self, dvdt, dwdt):
        """Proceed one step with params."""
        self.set_f_params(dvdt, dwdt)
        self.solver.integrate(self.solver.t + self.dt)
        return self.solver.y, self.solver.t

    def DCM(self):
        """Obtain direction cosine matrix A which rotates position vector."""
        q0 = self.solver.y[6, 0]
        q1 = self.solver.y[7, 0]
        q2 = self.solver.y[8, 0]
        q3 = self.solver.y[9, 0]
        q02 = q0 ** 2
        q12 = q1 ** 2
        q22 = q2 ** 2
        q32 = q3 ** 2
        A = np.array([
            [q02 + q12 - q22 - q32, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), q02 - q12 + q22 - q32, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q02 - q12 - q22 + q32]
        ])
        return A
