import abc
import warnings
from enum import Enum

import numpy as np


class FeatureFunction(abc.ABC):
    """Abstract feature function base class"""

    class Type(Enum):
        """Simple enum for input type to a feature/reward function"""

        OBSERVATION = "observation"
        OBSERVATION_ACTION = "observation_action"
        OBSERVATION_ACTION_OBSERVATION = "observation_action_observation"

        def check_args(self, o1, a, o2):
            """Checks that the given set of args are sufficient for this input type"""
            if self.has_o1:
                assert o1 is not None

            if self.has_a:
                assert a is not None

            if self.has_o2:
                assert o2 is not None

        @property
        def has_o1(self):
            return (
                (self == FeatureFunction.Type.OBSERVATION)
                or (self == FeatureFunction.Type.OBSERVATION_ACTION)
                or (self == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION)
            )

        @property
        def has_a(self):
            return (self == FeatureFunction.Type.OBSERVATION_ACTION) or (
                self == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION
            )

        @property
        def has_o2(self):
            return self == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION

    def __init__(self, type):
        """C-tor
        
        Args:
            type (InputType): Type of indicator features to construct
        """
        self.type = type

    def __len__(self):
        """Get length of the feature vector
        
        Returns:
            (int): Length of this feature vector
        """
        raise NotImplementedError

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action
        
        Args:
            o1 (any): Current observation
            a (any): Current action
            o2 (any): Next observation
        
        Returns:
            (numpy array): Feature vector for the current state(s) and/or action
        """
        raise NotImplementedError

    def expectation(self, rollouts, gamma=1.0, weights=None):
        """Get empirical discounted feature expectation for a collection of rollouts
        
        Args:
        
        Returns:
        
        """

        # Catch case when only a single rollout is passed
        if isinstance(rollouts[0], tuple):
            rollouts = [rollouts]

        if weights is None:
            # Default to uniform path weighting
            weights = np.ones(len(rollouts)) / len(rollouts)
        else:
            assert len(weights) == len(
                rollouts
            ), f"Path weights are not correct size, should be {len(rollouts)}, are {len(weights)}"

        phi_bar = np.zeros(len(self))
        for rollout, weight in zip(rollouts, weights):

            if self.type == self.Type.OBSERVATION:
                for t, (o1, _) in enumerate(rollout):
                    phi_bar += weight * self(o1) * (gamma ** t)
            elif self.type == self.Type.OBSERVATION_ACTION:
                for t, (o1, a) in enumerate(rollout[:-1]):
                    phi_bar += weight * self(o1, a) * (gamma ** t)
            elif self.type == self.Type.OBSERVATION_ACTION_OBSERVATION:
                for t, (o1, a) in enumerate(rollout[:-1]):
                    o2 = rollout[t + 1][0]
                    phi_bar += weight * self(o1, a, o2) * (gamma ** t)
            else:
                raise ValueError

        # Apply normalization
        phi_bar /= np.sum(weights)

        return phi_bar


class Indicator(FeatureFunction):
    """Indicator feature function"""

    def __init__(self, type, xtr):
        """C-tor
        
        Args:
            type (InputType): Type of indicator features to construct
            xtr (DiscreteExplicitExtras): MDP definition
        """
        super().__init__(type)

        if self.type == FeatureFunction.Type.OBSERVATION:
            self._vec = np.zeros(len(xtr.states))
        elif self.type == FeatureFunction.Type.OBSERVATION_ACTION:
            self._vec = np.zeros((len(xtr.states), len(xtr.actions)))
        elif self.type == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION:
            self._vec = np.zeros((len(xtr.states), len(xtr.actions), len(xtr.sates)))
        else:
            raise ValueError

    def __len__(self):
        """Get length of the feature vector
        
        Returns:
            (int): Length of this feature vector
        """
        return len(self._vec.flatten())

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action
        
        Args:
            o1 (any): Current observation
            a (any): Current action
            o2 (any): Next observation
        
        Returns:
            (numpy array): Feature vector for the current state(s) and/or action
        """
        self.type.check_args(o1, a, o2)

        self._vec.flat[:] = 0
        if self.type == FeatureFunction.Type.OBSERVATION:
            self._vec[o1] = 1.0
            return self._vec.flatten().copy()
        elif self.type == FeatureFunction.Type.OBSERVATION_ACTION:
            self._vec[o1, a] = 1.0
            return self._vec.flatten().copy()
        elif self.type == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION:
            self._vec[o1, a, o2] = 1.0
            return self._vec.flatten().copy()
        else:
            raise ValueError


class Disjoint(FeatureFunction):
    """A feature function where each input has one of a set of disjoint values"""

    def __init__(self, type, xtr, values):
        """C-tor
        
        Args:
            type (InputType): Type of indicator features to construct
            xtr (DiscreteExplicitExtras): MDP definition
            values (numpy array): Vector of integer feature values, one for each
                state/state-action/state-action-state tuple, depending on type.
        """
        super().__init__(type)

        # How many feature values do we have?
        self._len = np.max(values) + 1

        self._vec = np.zeros(self._len)

        if self.type == FeatureFunction.Type.OBSERVATION:
            self._values = values.reshape((len(xtr.states)))
        elif self.type == FeatureFunction.Type.OBSERVATION_ACTION:
            self._values = values.reshape((len(xtr.states), len(xtr.actions)))
        elif self.type == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION:
            self._values = values.reshape(
                (len(xtr.states), len(xtr.actions), len(xtr.states))
            )
        else:
            raise ValueError

    def __len__(self):
        return self._len

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action
        
        Args:
            o1 (any): Current observation
            a (any): Current action
            o2 (any): Next observation
        
        Returns:
            (numpy array): Feature vector for the current state(s) and/or action
        """
        self.type.check_args(o1, a, o2)

        self._vec[:] = 0
        try:
            if self.type == FeatureFunction.Type.OBSERVATION:
                self._vec[self._values[o1]] = 1.0
            elif self.type == FeatureFunction.Type.OBSERVATION_ACTION:
                self._vec[self._values[o1, a]] = 1.0
            elif self.type == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION:
                self._vec[self._values[o1, a, o2]] = 1.0
            else:
                raise ValueError
        except IndexError:
            pass
            # warnings.warn(
            #     f"Requested φ({o1}, {a}, {o2}), however slice is out-of-bounds. This could be due to using padded rollouts, in which case you can safely ignore this warning."
            # )
        return self._vec.copy()
