"""Model of the quadrotor."""

import sys
sys.path.append('../utils')

from six_dimension_dynamics import dynamics
import propeller
import numpy as np
from math_function import euler2Quartanion

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Quadrotor(object):
    """docstring for Quadrotor."""
    def __init__(self, m, I, arr_propellers):
        """Initialization."""
        super(Quadrotor, self).__init__()
        self.weight = m             # Weight of the machine [kg]
        self.moment_of_inertia = I  # Moment of inertia, 3 * 3 matrix [kg m^2]
        self.Inverse_of_I = np.linalg.inv(I)
        self.propellers = arr_propellers  # instances of propellers

        x0 = np.zeros((13, 1), dtype=float)
        x0[6:10, 0] = euler2Quartanion(0.0, 0.0, 0.0)
        # x0[8, 0] = 1.0
        t0 = 0.0
        dvdt = np.zeros((3, 1))
        dwdt = np.zeros((3, 1))
        dt = 0.001
        self.dynamics = dynamics(x0, t0, dvdt, dwdt, dt)

    def calc_Force_Torque(self, inputs):
        """Calculate force and torque from the inputs."""
        if len(self.propellers) != len(inputs):
            raise ValueError("the number of propellers is not same as inputs.")

        sum_FM = np.zeros((6, 1))
        for id, propeller in enumerate(self.propellers):
            sum_FM += propeller.get_force_torque(inputs[id])
        return sum_FM

    def calc_dvdt(self, force_body):
        """Calculate dvdt, inertial frame."""
        print(self.dynamics.DCM())
        f_inertial = np.dot(self.dynamics.DCM(), force_body)
        return f_inertial / self.weight

    def calc_dwdt(self, moment_body):
        """Calculate dwdt, body frame."""
        return np.dot(self.Inverse_of_I, moment_body)

    def calc_one_step(self, inputs):
        FM = self.calc_Force_Torque(inputs)
        x, t = self.dynamics.simulate_onestep(
            dvdt=self.calc_dvdt(FM[0:3, 0]),
            dwdt=self.calc_dwdt(FM[3:6, 0])
        )

if __name__ == '__main__':
    array_propellers = []

    array_propellers.append(
        propeller.Propeller(
            position=np.array([[1.0, 1.0, 0.0]]).T,
            direction=np.array([[0.0, 0.0, -1.0]]).T
        )
    )

    array_propellers.append(
        propeller.Propeller(
            position=np.array([[1.0, -1.0, 0.0]]).T,
            direction=np.array([[0.0, 0.0, -1.0]]).T,
            k_F=1.0,
            k_M=-1.0
        )
    )

    array_propellers.append(
        propeller.Propeller(
            position=np.array([[-1.0, -1.0, 0.0]]).T,
            direction=np.array([[0.0, 0.0, -1.0]]).T
        )
    )

    array_propellers.append(
        propeller.Propeller(
            position=np.array([[-1.0, 1.0, 0.0]]).T,
            direction=np.array([[0.0, 0.0, -1.0]]).T,
            k_F=1.0,
            k_M=-1.0
        )
    )

    model = Quadrotor(
        m=1.0,
        I=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]),
        arr_propellers=array_propellers
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(2000):
        model.calc_one_step([5.0, 5.0, 0.0, 0.0])
        x = model.dynamics.get_X()
        ax.scatter3D(x[0, 0], x[1, 0], x[2, 0])
    print(model.dynamics.solver.y)
    plt.show()
