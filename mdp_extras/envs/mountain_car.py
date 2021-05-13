import numpy as np
import itertools as it

from scipy.stats import multivariate_normal

from mdp_extras import FeatureFunction


# Range of the mountain car state space
MC_POSITION_RANGE = (-1.2, 0.6)
MC_VELOCITY_RANGE = (-0.07, 0.07)


class GaussianBasis(FeatureFunction):
    """Gaussian bases feature function for MountainCar

    A set of Gaussian functions spanning the state space
    """

    def __init__(self, num=5, pos_range=(-1.2, 0.6), vel_range=(-0.07, 0.07)):
        """C-tor"""
        super().__init__(self.Type.OBSERVATION)

        self.dim = num ** 2
        pos_delta = pos_range[1] - pos_range[0]
        vel_delta = vel_range[1] - vel_range[0]

        pos_mean_diff = pos_delta / (num + 1)
        pos_basis_means = (
            np.linspace(pos_mean_diff * 0.5, pos_delta - pos_mean_diff * 0.5, num)
            + pos_range[0]
        )
        pos_basis_std = pos_mean_diff ** 2 / 10

        vel_mean_diff = vel_delta / (num + 1)
        vel_basis_means = (
            np.linspace(vel_mean_diff * 0.5, vel_delta - vel_mean_diff * 0.5, num)
            + vel_range[0]
        )
        vel_basis_std = vel_mean_diff ** 2 / 10

        covariance = np.diag([pos_basis_std, vel_basis_std])
        means = np.array(list(it.product(pos_basis_means, vel_basis_means)))

        self.rvs = [multivariate_normal(m, covariance) for m in means]

    def __len__(self):
        """Get the length of the feature vector"""
        return self.dim

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action"""
        return np.array([rv.pdf(o1) for rv in self.rvs])
