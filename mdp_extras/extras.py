"""Extra object definitions
"""

import warnings

import numpy as np

from mdp_extras.utils import compute_parents_children


class DiscreteExplicitExtras:
    """Extras for a discrete state, discrete action, explicit dynamics MDP"""

    def __init__(
        self, states, actions, p0s, t_mat, terminal_state_mask, gamma=1.0, padded=False
    ):
        """C-tor
        
        Args:
            states (iterable): Iterable containing MDP states
            actions (iterable): Iterable containing MDP actions
            p0s (numpy array): |S| vector of starting state probabilities
            t_mat (numpy array): |S|x|A|x|S| numpy array of transition probabilities
            terminal_state_mask (numpy array): |S| boolean vector indicating if states
                are terminal or not
                
            gamma (float): Discount factor
            padded (bool): True if this MDP has been padded to include an extra state
                and action.
        """

        self._states = states
        self._actions = actions
        self._p0s = p0s
        self._t_mat = t_mat
        self._terminal_state_mask = terminal_state_mask
        self._gamma = gamma

        self._parents, self._children = compute_parents_children(
            self._t_mat, self._terminal_state_mask
        )

        # Has this MDP been padded with an extra state and action?
        # See utils.padding_trick()
        self._is_padded = padded

    @staticmethod
    def fromdiscrete(env, gamma=1.0):
        """Builds DiscreteExplicitExtras from DiscreteEnv
        
        Args:
            env (gym.envs.toy_text.discrete.DiscreteEnv): Environment to build Extras
                from
            gamma (float): Discount factor
            
        Returns:
            (DiscreteExplicitExtras): Extras object
        """

        states = np.arange(env.nS)
        actions = np.arange(env.nA)
        p0s = np.array(env.isd)

        # Build transition dynamics
        t_mat = np.zeros((env.nS, env.nA, env.nS))
        terminal_state_mask = np.zeros_like(states)
        for s1 in states:
            for a in actions:
                for prob, s2, r, done in env.P[s1][a]:
                    t_mat[s1, a, s2] += prob
                    if done:
                        terminal_state_mask[s2] = 1.0

        # Sanity check - is the transition matrix valid
        for s1 in states:
            for a in actions:
                transition_prob = np.sum(t_mat[s1, a, :])
                if transition_prob < 1.0:
                    warnings.warn(
                        "This environment has inconsistent dynamics - normalizing state-action {}-{}!".format(
                            s1, a
                        )
                    )
                t_mat[s1, a, :] /= transition_prob

        return DiscreteExplicitExtras(
            states, actions, p0s, t_mat, terminal_state_mask, gamma
        )

    @property
    def states(self):
        """Iterable over MDP states"""
        return self._states

    @property
    def actions(self):
        """Iterable over MDP actions"""
        return self._actions

    @property
    def p0s(self):
        """|S| vector of starting state probabilities"""
        return self._p0s

    @property
    def t_mat(self):
        """|S|x|A|x|S| array of transition probabilities"""
        return self._t_mat

    @property
    def terminal_state_mask(self):
        """|S| vector indicating terminal states"""
        return self._terminal_state_mask

    @property
    def parents(self):
        """Dict mapping a state to it's (s, a) parents"""
        return self._parents

    @property
    def children(self):
        """Dict mapping a state to it's (a, s') children"""
        return self._children

    @property
    def gamma(self):
        """Discount factor"""
        return self._gamma

    @property
    def is_padded(self):
        return self._is_padded

    @property
    def as_unpadded(self):
        if not self.is_padded:
            return self
        else:
            return DiscreteExplicitExtras(
                self.states[:-1],
                self.actions[:-1],
                self.p0s[:-1],
                self.t_mat[:-1, :-1, :-1],
                self.terminal_state_mask[:-1],
                self.gamma,
                False,
            )

    def path_log_probability(self, p):
        """Get log probability of a state-action path under MDP dynamics
        
        Args:
            p (list): (s, a) path, stored as a list
        
        Returns:
            (float): Log probability of path under dynamics
        """
        path_log_prob = np.log(self.p0s[p[0][0]])
        for (s1, a), (s2, _) in zip(p[:-1], p[1:]):
            path_log_prob += np.log(self.t_mat[s1][a][s2])
        return path_log_prob
