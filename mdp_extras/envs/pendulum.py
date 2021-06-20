import numpy as np
import itertools as it

from scipy.stats import norm, vonmises

from mdp_extras import FeatureFunction


def pendulum_obs_to_state(obs):
    costheta, sintheta, thetadot = obs
    theta = np.arctan2(sintheta, costheta)
    return np.array([theta, thetadot])


class VonMisesNormalBasis(FeatureFunction):
    """VonMises-Normal basis feature function for Pendulum-v0

    A set of wrapped 'gaussian' functions spanning position space
                    appended or multiplied with
    a set of gaussian functions spanning the velocity space
    """

    def __init__(
        self,
        num=4,
        pos_range=(-np.pi, np.pi),
        vel_range=(-8.0, 8.0),
        kappa=10,
        std=1.0,
    ):
        """C-tor"""
        super().__init__(self.Type.OBSERVATION)

        self.dim = num ** 2
        pos_delta = pos_range[1] - pos_range[0]
        vel_delta = vel_range[1] - vel_range[0]

        pos_offset = pos_delta / (num)
        pos_basis_means = np.linspace(0, pos_delta - pos_offset, num)
        self.pos_rvs = [vonmises(kappa=kappa, loc=m) for m in pos_basis_means]

        vel_mean_diff = vel_delta / (num + 1)
        vel_basis_means = (
            np.linspace(vel_mean_diff * 0.5, vel_delta - vel_mean_diff * 0.5, num)
            + vel_range[0]
        )
        self.vel_rvs = [norm(loc=m, scale=std) for m in vel_basis_means]

    def __len__(self):
        """Get the length of the feature vector"""
        return self.dim

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action"""
        pos, vel = o1

        # For cartesian product of feature vectors
        return np.array(
            list(
                np.product(pd)
                for pd in it.product(
                    [rv.pdf(pos) for rv in self.pos_rvs],
                    [rv.pdf(vel) for rv in self.vel_rvs],
                )
            )
        )

        # For appended feature vectors
        # return np.array(
        #     [rv.pdf(pos) for rv in self.pos_rvs] + [rv.pdf(vel) for rv in self.vel_rvs]
        # )


if __name__ == "__main__":

    # A brief demo
    basis_dim = 4
    phi = VonMisesNormalBasis(num=basis_dim)

    num = 20
    t = np.linspace(-np.pi, np.pi, num)
    td = np.linspace(-8.0, 8.0, num)

    X, Y = np.meshgrid(t, td)
    Z = np.zeros([X.shape[0], X.shape[1], basis_dim ** 2])
    for id0 in range(X.shape[0]):
        for id1 in range(X.shape[1]):
            _x = X[id0, id1]
            _y = Y[id0, id1]
            Z[id0, id1] = phi([_x, _y])

    import matplotlib.pyplot as plt

    for _idx in range(basis_dim ** 2):
        plt.figure()
        _Z = Z[:, :, _idx]
        plt.contour(X, Y, _Z)
        plt.show()
        plt.close()
