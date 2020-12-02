"""Various utilities for generating 'extra' objects"""

import copy
import warnings

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


def padding_trick(xtr, phi, reward, rollouts=None, max_length=None):
    """Apply padding trick, adding an auxiliary state and action to an MDP
    
    We gain a O(|S|) space and time efficiency improvement with our MaxEnt IRL algorithm
    for MDPs with terminal states by transforming them to have no terminal states. This
    is done by adding a dummy state and action that pad any trajectories out to a fixed
    upper length.
    
    Args:
        xtr (DiscreteExplicitExtras): Extras object
        phi (Indicator): Indicator/Disjoint feature function
        reward (Linear): Linear reward function
        
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] rollouts to pad
        max_length (int): Optional maximum length to pad to, otherwise paths are padded
            to match the length of the longest path
        
    Returns:
        (DiscreteExplicitExtras): Extras object, padded with auxiliary state and action
        (Indicator): Indicator feature function, padded with auxiliary state and action
        (Linear): Linear reward function, padded with auxiliary state and action
        
        (list): List of rollouts, padded to max_length. Only returned if rollouts is not
            None
    """

    from mdp_extras import DiscreteExplicitExtras, Indicator, Disjoint, Linear

    t_mat = np.pad(xtr.t_mat, (0, 1), mode="constant")
    s_aux = t_mat.shape[0] - 1
    a_aux = t_mat.shape[1] - 1

    p0s = np.pad(xtr.p0s, (0, 1), mode="constant")
    states = np.arange(t_mat.shape[0])
    actions = np.arange(t_mat.shape[1])

    # Auxiliary state is absorbing
    t_mat[-1, -1, -1] = 1

    # Terminal states are no longer absorbing
    for terminal_state in np.argwhere(xtr.terminal_state_mask):
        t_mat[terminal_state, :, terminal_state] = 0
    terminal_state_mask = np.zeros(t_mat.shape[0])

    # Auxiliary state reachable anywhere if auxiliary action is taken
    t_mat[:, -1, -1] = 1

    xtr2 = DiscreteExplicitExtras(
        states, actions, p0s, t_mat, terminal_state_mask, xtr.gamma, True
    )

    # Auxiliary state, action don't modify rewards
    if isinstance(phi, Indicator):
        # Pad an indicator feature function and linear reward function
        rs, rsa, rsas = reward.structured(xtr, phi)
        rs = np.pad(rs, (0, 1), mode="constant")
        rs[-1] = 0

        rsa = np.pad(rsa, (0, 1), mode="constant")
        rsa[:, -1] = 0

        rsas = np.pad(rsas, (0, 1), mode="constant")
        rsas[:, 0:-1, -1] = -np.inf  # Illegal transition
        rsas[:, -1, -1] = 0

        if phi.type == Indicator.Type.OBSERVATION:
            r2 = Linear(rs.flatten())
        elif phi.type == Indicator.Type.OBSERVATION_ACTION:
            r2 = Linear(rsa.flatten())
        elif phi.type == Indicator.Type.OBSERVATION_ACTION_OBSERVATION:
            r2 = Linear(rsas.flatten())
        else:
            raise ValueError

        phi2 = Indicator(phi.type, xtr2)

    elif isinstance(phi, Disjoint):
        # Pad a disjoint feature function and linear reward function

        phi2 = phi
        r2 = reward
    else:
        raise ValueError

    if rollouts is None:
        return xtr2, phi2, r2
    else:
        # Measure the length of the rollouts
        r_len = [len(r) for r in rollouts]
        if max_length is None:
            max_length = max(r_len)
        elif max_length < max(r_len):
            warnings.warn(
                f"Provided max length ({max_length}) is < maximum path length ({max(r_len)}), using maximum path length instead"
            )
            max_length = max(r_len)

        # Finally, pad the trajectories out to the maximum length
        rollouts2 = []
        for rollout in rollouts:
            rollout2 = rollout.copy()
            if len(rollout2) < max_length:
                rollout2[-1] = (rollout2[-1][0], a_aux)
                while len(rollout2) != max_length - 1:
                    rollout2.append((s_aux, a_aux))
                rollout2.append((s_aux, None))
            rollouts2.append(rollout2)

        return xtr2, phi2, r2, rollouts2


def padding_trick_mm(xtr, phi, rewards, rollouts, max_length=None):
    """Apply padding trick to a Multi Modal MDP
    
    Args:
        xtr (DiscreteExplicitExtras): Extras object
        phi (Indicator): Indicator/Disjoint feature function
        rewards (list/dict): List or dict of Linear reward functions
        
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] rollouts to pad
        max_length (int): Optional maximum length to pad to, otherwise paths are padded
            to match the length of the longest path
        
    Returns:
        (DiscreteExplicitExtras): Extras object, padded with auxiliary state and action
        (Indicator): Indicator feature function, padded with auxiliary state and action
        (list/dict): List or dict of Linear reward functions, padded with auxiliary
            state and action
        
        (list): List of rollouts, padded to max_length. Only returned if rollouts is not
            None
    
    """
    if isinstance(rewards, list):
        rewards_padded = []
        for reward in rewards:
            xtr_padded, phi_padded, reward_padded, rollouts_padded = padding_trick(
                xtr, phi, reward, rollouts, max_length
            )
            rewards_padded.append(reward)
        return xtr_padded, phi_padded, rewards_padded, rollouts_padded
    elif isinstance(rewards, dict):
        rewards_padded = {}
        for reward_name, reward in rewards.items():
            xtr_padded, phi_padded, reward_padded, rollouts_padded = padding_trick(
                xtr, phi, reward, rollouts, max_length
            )
            rewards_padded[reward_name] = reward_padded
        return xtr_padded, phi_padded, rewards_padded, rollouts_padded
    else:
        raise ValueError
