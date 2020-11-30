"""Various utilites for generating 'extra' objects"""

import copy

import numpy as np
from gym.spaces import Discrete


def compute_parents_children(t_mat, terminal_state_mask):
    """Compute parent and child dictionaries
    
    Args:
        t_mat (numpy array): |S|x|A|x|S| array of transition dynamics
        terminal_state_mask (numpy array): |S| vector indicating terminal states
    
    Returns:
        (dict): Dictionary mapping states to (s, a) parent tuples
        (dict): Dictionary mapping states to (a, s') child tuples
    """
    parents = {s: [] for s in range(t_mat.shape[0])}
    children = copy.deepcopy(parents)
    for s2 in range(t_mat.shape[0]):
        for s1, a in np.argwhere(t_mat[:, :, s2] > 0):
            if not terminal_state_mask[s1]:
                parents[s2].append(tuple((s1, a)))
                children[s1].append(tuple((a, s2)))

    return parents, children


def trajectory_reward(xtr, phi, reward, rollout):
    """Get discounted reward of a trajectory
    
    Args:
        xtr (mdp_extras.DiscreteExplicitExtras): MDP extras
        phi (mdp_extras.FeatureFunction): Feature function to use with linear reward
            parameters.
        reward (mdp_extras.RewardFunction): Reward function
        rollout (list): Trajectory as a list of (s, a) tuples
    """
    return reward(phi.expectation(rollout, gamma=xtr.gamma))


class DiscreteExplicitLinearEnv:
    """Get a gym.Env definition from extras, features and reward function"""

    def __init__(self, xtr, phi, reward):
        """C-tor
        
        Args:
            xtr (DiscreteExplicitExtras): Extras object for this MDP
            phi (FeatureFunction): Feature function for this MDP
            reward (Linear): Linear reward function
        """
        self._xtr = xtr
        self._phi = phi
        self._reward = reward

        self.reward_range = self._reward.range
        self.observation_space = Discrete(len(self._xtr.states))
        self.action_space = Discrete(len(self._xtr.actions))

        self.state = self.reset()

    def reset(self):
        """Reset the MDP, getting a new state
        
        Returns:
            (int): New state sampled from starting state distribution
        """
        self.state = self._xtr.states[
            np.random.choice(range(len(self._xtr.states)), p=self._xtr.p0s)
        ]
        return self.state

    def step(self, a):
        """Take an action in the MDP
        
        Args:
            a (int): Action to take
        
        Returns:
            (int): New state
            (float): Reward for taking action a from previous state
            (bool): Has the MPD terminated?
            (dict): Optional extra information
        """
        s2_probs = self._xtr.t_mat[self.state, a, :]
        new_state = self._xtr.states[np.random.choice(range(len(s2_probs)), p=s2_probs)]
        feature_vec = self._phi(self.state, a, new_state)
        reward = self._reward(feature_vec)
        done = self._xtr.terminal_state_mask[new_state]
        info = {}

        self.state = new_state

        return self.state, reward, done, info
