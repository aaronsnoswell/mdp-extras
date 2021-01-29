"""Various utilities for generating 'extra' objects"""

import copy
import types
import difflib
import warnings

import numpy as np
from gym.spaces import Discrete


class PaddedMDPWarning(UserWarning):
    pass


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


def padding_trick(xtr, rollouts=None, max_length=None):
    """Apply padding trick, adding an auxiliary state and action to an MDP
    
    We gain a O(|S|) space and time efficiency improvement with our MaxEnt IRL algorithm
    for MDPs with terminal states by transforming them to have no terminal states. This
    is done by adding a dummy state and action that pad any trajectories out to a fixed
    upper length.
    
    Args:
        xtr (DiscreteExplicitExtras): Extras object
        
        rollouts (list): List of [(s, a), (s, a), ..., (s, None)] rollouts to pad
        max_length (int): Optional maximum length to pad to, otherwise paths are padded
            to match the length of the longest path
        
    Returns:
        (DiscreteExplicitExtras): Extras object, padded with auxiliary state and action
        (list): List of rollouts, padded to max_length, or None if rollouts was not
            passed.
    """

    from mdp_extras import DiscreteImplicitExtras, DiscreteExplicitExtras

    # Create the auxiliary state
    s_aux = len(xtr.states)
    states = np.pad(xtr.states, (0, 1), mode="constant", constant_values=s_aux)

    # Create the auxiliary state
    a_aux = len(xtr.actions)
    actions = np.pad(xtr.actions, (0, 1), mode="constant", constant_values=a_aux)

    # Add extra state and action to p0s - auxiliary state has 0 starting probability
    p0s = np.pad(xtr.p0s, (0, 1), mode="constant", constant_values=0.0)

    if isinstance(xtr, DiscreteExplicitExtras):
        # Handle tabular MDP

        t_mat = np.pad(xtr.t_mat, (0, 1), mode="constant")

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

    elif isinstance(xtr, DiscreteImplicitExtras):
        # Handle implicit dynamics MDP

        # Duplicate the extras as a starting point
        xtr2 = copy.copy(xtr)

        def t_prob_padded(self, s1, a, s2):
            """Transition probability function for padded DiscreteImplicitExtras object"""
            s_aux = self.states[-1]
            a_aux = self.actions[-1]

            if s1 == s_aux:
                # Starting at auxiliary state you can only go to the auxiliary state
                if s2 == s_aux:
                    return 1.0
                else:
                    return 0.0
            else:
                # Starting at any other state
                if a == a_aux:
                    # Taking auxiliary action leads you to the auxiliary state only
                    if s2 == s_aux:
                        return 1.0
                    else:
                        return 0.0
                else:
                    # Taking any other action
                    if s2 == s_aux:
                        # You cannot reach the auxiliary state in any other way
                        return 0.0
                    else:
                        if s1 == s2 and self._terminal_state_mask_old[s1]:
                            # Terminal states are no longer absorbing
                            return 0.0
                        else:
                            # Refer to existing dynamics for this transition
                            return self.t_prob_old(s1, a, s2)

        # Store copies of old objects
        xtr2.t_prob_old = xtr2.t_prob
        xtr2._terminal_state_mask_old = xtr2._terminal_state_mask

        # Overwrite properties
        xtr2._states = states
        xtr2._actions = actions
        xtr2._p0s = p0s

        # Update transition dynamics function
        xtr2.t_prob = types.MethodType(t_prob_padded, xtr2)

        # Clear terminal state mask
        xtr2._terminal_state_mask = np.zeros_like(xtr2._states)

        # Update children dict
        xtr2._children[s_aux] = []
        max_children = 0
        for s in xtr2.states:
            xtr2._children[s].append(s_aux)
            max_children = max(max_children, len(xtr2._children[s]))

        # Update parents
        xtr2._parents[s_aux] = xtr2.states.copy()

        # Update children_fixedsize array
        xtr2._children_fixedsize = (
            np.zeros((len(xtr2._states), max_children), dtype=int) - 1
        )
        xtr2._children_fixedsize[
            :-1, : xtr.children_fixedsize.shape[1]
        ] = xtr.children_fixedsize.copy()
        xtr2._children_fixedsize[s_aux, : len(xtr2._children[s_aux])] = xtr2._children[
            s_aux
        ]

        warnings.warn(
            "Padding a DiscreteImplicitExtras object will *not* update the parents_fixedsize member as this leads to a MemoryError. Please handle this property with caution."
        )
        # # Update parents_fixedsize array
        # xtr2._parents_fixedsize = csr_matrix()
        # # XXX ajs MemoryError here
        # xtr2._parents_fixedsize = (
        #     np.zeros((len(xtr2._states), len(xtr2._states)), dtype=int) - 1
        # )
        # xtr2._parents_fixedsize[
        #     :-1, : xtr.parents_fixedsize.shape[1]
        # ] = xtr.parents_fixedsize.copy()
        # xtr2._parents_fixedsize[s_aux:] = xtr2.states.copy()

        # Flag as padded
        xtr2._is_padded = True

    else:

        raise ValueError(f"Unknown MDP class {xtr}")

    # Auxiliary state, action don't modify rewards
    # To achive this we simply leave the feature function alone - if a feature of
    # an auxiliary state/action is requested, we will throw a warning (which can be
    # suppressed)

    rollouts2 = None

    if rollouts is not None:
        # Pad rollouts as well

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

    return xtr2, rollouts2


def nonoverlapping_shared_subsequences(list1, list2):
    """Find all non-overlapping shared subsequences between two sequences
    
    Algorithm is from https://stackoverflow.com/a/32318377/885287
    NB: This method does not guarantee the sub-sequences will be in any order.
    
    This is used in the %-distance-missed metric
    
    Args:
        list1 (sequence): First sequence
        list2 (sequence): Second sequence
        
    Yields:
        (list): The next sub-sequence that is shared by the list
    """

    list1 = copy.copy(list1)
    list2 = copy.copy(list2)

    while True:
        mbs = difflib.SequenceMatcher(None, list1, list2).get_matching_blocks()
        if len(mbs) == 1:
            break
        for i, j, n in mbs[::-1]:
            if n > 0:
                yield list1[i : i + n]
            del list1[i : i + n]
            del list2[j : j + n]


def log_sum_exp(x_vals):
    """Compute ln(\sum_i(\exp(x_i))) Without numerical overflow

    This operation comes up surprisingly often in machine learning!

    Args:
        x_vals (numpy array): Vector of values that need to be exponentiated, summed, and logged
            (in that order)

    Returns:
        (float): The log-sum-exp of x_vals
    """
    max_val = np.max(x_vals)

    # Apply log-sum-exp trick
    return max_val + np.log(np.sum(np.exp(x_vals - max_val)))
