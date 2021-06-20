import numpy as np

from scipy.stats import norm, vonmises

from mdp_extras import FeatureFunction


class VonMisesNormalBasis(FeatureFunction):
    """VonMises-Normal basis feature function for Pendulum-v0

    A set of wrapped 'gaussian' functions spanning position space appended to a set of
    gaussian functions spanning the velocity space
    """

    def __init__(self, num=4, pos_range=(-np.pi, np.pi), vel_range=(-8.0, 8.0)):
        """C-tor"""
        super().__init__(self.Type.OBSERVATION)

        self.dim = num ** 2
        pos_delta = pos_range[1] - pos_range[0]
        vel_delta = vel_range[1] - vel_range[0]

        pos_offset = pos_delta / (num)
        pos_basis_means = np.linspace(0, pos_delta - pos_offset, num)
        pos_basis_kappa = pos_offset ** 5 / 10
        self.pos_rvs = [vonmises(kappa=pos_basis_kappa, loc=m) for m in pos_basis_means]

        vel_mean_diff = vel_delta / (num + 1)
        vel_basis_means = (
            np.linspace(vel_mean_diff * 0.5, vel_delta - vel_mean_diff * 0.5, num)
            + vel_range[0]
        )
        vel_basis_std = vel_mean_diff ** 2 / 10
        self.vel_rvs = [norm(loc=m, scale=vel_basis_std) for m in vel_basis_means]

    def __len__(self):
        """Get the length of the feature vector"""
        return self.dim

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action"""
        pos, vel = o1
        return np.array(
            [rv.pdf(pos) for rv in self.pos_rvs] + [rv.pdf(vel) for rv in self.vel_rvs]
        )
