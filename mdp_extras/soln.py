"""Find solutions to MDPs via value iteration methods"""

import abc
import copy
import pickle
import warnings

import numpy as np
import itertools as it

from numba import jit

from mdp_extras.features import FeatureFunction
from mdp_extras.rewards import Linear
from mdp_extras.utils import DiscreteExplicitLinearEnv, PaddedMDPWarning


def q_vi(xtr, phi, reward, eps=1e-6, verbose=False, max_iter=None):
    """Value iteration to find the optimal state-action value function
    
    Args:
        xtr (DiscreteExplicitExtras):
        phi (FeatureFunction): Feature function
        reward (RewardFunction): Reward function
        
        eps (float): Value convergence tolerance
        verbose (bool): Extra logging
        max_iter (int): If provided, iteration will terminate regardless of convergence
            after this many iterations.
    
    Returns:
        (numpy array): |S|x|A| matrix of state-action values
    """

    @jit(nopython=True)
    def _nb_q_value_iteration(
        t_mat, gamma, rs, rsa, rsas, eps=1e-6, verbose=False, max_iter=None
    ):
        """Value iteration to find the optimal state-action value function
        
        Args:
            t_mat (numpy array): |S|x|A|x|S| transition matrix
            gamma (float): Discount factor
            rs (numpy array): |S| State reward vector
            rsa (numpy array): |S|x|A| State-action reward vector
            rsas (numpy array): |S|x|A|x|S| State-action-state reward vector
            
            eps (float): Value convergence tolerance
            verbose (bool): Extra logging
            max_iter (int): If provided, iteration will terminate regardless of convergence
                after this many iterations.
        
        Returns:
            (numpy array): |S|x|A| matrix of state-action values
        """

        q_value_fn = np.zeros((t_mat.shape[0], t_mat.shape[1]))

        _iter = 0
        while True:
            delta = 0
            for s1 in range(t_mat.shape[0]):
                for a in range(t_mat.shape[1]):
                    q = q_value_fn[s1, a]
                    state_values = np.zeros(t_mat.shape[0])
                    for s2 in range(t_mat.shape[2]):
                        state_values[s2] += t_mat[s1, a, s2] * (
                            rs[s1]
                            + rsa[s1, a]
                            + rsas[s1, a, s2]
                            + gamma * np.max(q_value_fn[s2, :])
                        )
                    q_value_fn[s1, a] = np.sum(state_values)
                    delta = max(delta, np.abs(q - q_value_fn[s1, a]))

            if max_iter is not None and _iter >= max_iter:
                if verbose:
                    print("Terminating before convergence, # iterations = ", _iter)
                    break

            # Check value function convergence
            if delta < eps:
                break
            else:
                if verbose:
                    print("Value Iteration #", _iter, " delta=", delta)

            _iter += 1

        return q_value_fn

    xtr = xtr.as_unpadded
    return _nb_q_value_iteration(
        xtr.t_mat, xtr.gamma, *reward.structured(xtr, phi), eps, verbose, max_iter
    )


def q2v(q_star):
    """Convert optimal state-action value function to optimal state value function
    
    Args:
        q_star (numpy array): |S|x|A| Optimal state-action value function array
    
    Returns:
        (numpy array): |S| Optimal state value function vector
    """
    return np.max(q_star, axis=1)


def v_vi(xtr, phi, reward, eps=1e-6, verbose=False, max_iter=None):
    """Value iteration to find the optimal state value function
    
    Args:
        xtr (DiscreteExplicitExtras): Extras object
        phi (FeatureFunction): Feature function
        reward (RewardFunction): Reward function
        
        eps (float): Value convergence tolerance
        verbose (bool): Extra logging
        max_iter (int): If provided, iteration will terminate regardless of convergence
            after this many iterations.
    
    Returns:
        (numpy array): |S|x|A| matrix of state-action values
    """

    @jit(nopython=True)
    def _nb_value_iteration(
        t_mat, gamma, rs, rsa, rsas, eps=1e-6, verbose=False, max_iter=None
    ):
        """Value iteration to find the optimal value function
        
        Args:
            t_mat (numpy array): |S|x|A|x|S| transition matrix
            gamma (float): Discount factor
            rs (numpy array): |S| State reward vector
            rsa (numpy array): |S|x|A| State-action reward vector
            rsas (numpy array): |S|x|A|x|S| State-action-state reward vector
            
            eps (float): Value convergence tolerance
            verbose (bool): Extra logging
            max_iter (int): If provided, iteration will terminate regardless of convergence
                after this many iterations.
        
        Returns:
            (numpy array): |S| vector of state values
        """

        value_fn = np.zeros(t_mat.shape[0])

        _iter = 0
        while True:
            delta = 0
            for s1 in range(t_mat.shape[0]):
                v = value_fn[s1]
                action_values = np.zeros(t_mat.shape[1])
                for a in range(t_mat.shape[1]):
                    for s2 in range(t_mat.shape[2]):
                        action_values[a] += t_mat[s1, a, s2] * (
                            rsa[s1, a] + rsas[s1, a, s2] + rs[s2] + gamma * value_fn[s2]
                        )
                value_fn[s1] = np.max(action_values)
                delta = max(delta, np.abs(v - value_fn[s1]))

            if max_iter is not None and _iter >= max_iter:
                if verbose:
                    print("Terminating before convergence, # iterations = ", _iter)
                    break

            # Check value function convergence
            if delta < eps:
                break
            else:
                if verbose:
                    print("Value Iteration #", _iter, " delta=", delta)

            _iter += 1

        return value_fn

    xtr = xtr.as_unpadded
    return _nb_value_iteration(
        xtr.t_mat, xtr.gamma, *reward.structured(xtr, phi), eps, verbose, max_iter
    )


def v2q(v_star, xtr, phi, reward):
    """Convert optimal state value function to optimal state-action value function
    
    Args:
        v_star (numpy array): |S| Optimal state value function vector
        xtr (DiscreteExplicitExtras):
        phi (FeatureFunction): Feature function
        reward (RewardFunction): Reward function
    
    Returns:
        (numpy array): |S|x|A| Optimal state-action value function array
    """

    @jit(nopython=True)
    def _nb_q_from_v(
        v_star,
        t_mat,
        gamma,
        state_rewards,
        state_action_rewards,
        state_action_state_rewards,
    ):
        """Find Q* given V* (numba optimized version)
        
        Args:
            v_star (numpy array): |S| vector of optimal state values
            t_mat (numpy array): |S|x|A|x|S| transition matrix
            gamma (float): Discount factor
            state_rewards (numpy array): |S| array of state rewards
            state_action_rewards (numpy array): |S|x|A| array of state-action rewards
            state_action_state_rewards (numpy array): |S|x|A|x|S| array of state-action-state rewards
        
        Returns:
            (numpy array): |S|x|A| array of optimal state-action values
        """

        q_star = np.zeros(t_mat.shape[0 : 1 + 1])

        for s1 in range(t_mat.shape[0]):
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    q_star[s1, a] += t_mat[s1, a, s2] * (
                        state_action_rewards[s1, a]
                        + state_action_state_rewards[s1, a, s2]
                        + state_rewards[s2]
                        + gamma * v_star[s2]
                    )

        return q_star

    xtr = xtr.as_unpadded
    return _nb_q_from_v(v_star, xtr.t_mat, xtr.gamma, *reward.structured(xtr, phi))


def pi_eval(xtr, phi, reward, policy, eps=1e-6, num_runs=1):
    """Determine the value function of a given policy
    
    Args:
        xtr (DiscreteExplicitExtras): Extras object
        phi (FeatureFunction): Feature function
        reward (RewardFunction): Reward function
        policy (object): Policy object providing a .predict(s) method to match the
            stable-baselines policy API
        
        eps (float): State value convergence threshold
        num_runs (int): Number of policy evaluations to average over - for deterministic
            policies, leave this as 1, but for stochastic policies, set to a large
            number (the function will then sample actions stochastically from the
            policy).
    
    Returns:
        (numpy array): |S| state value vector
    """

    @jit(nopython=True)
    def _nb_policy_evaluation(
        t_mat, gamma, rs, rsa, rsas, policy_vector, eps=1e-6,
    ):
        """Determine the value function of a given deterministic policy
        
        Args:
            t_mat (numpy array): |S|x|A|x|S| transition matrix
            gamma (float): Discount factor
            rs (numpy array): |S| State reward vector
            rsa (numpy array): |S|x|A| State-action reward vector
            rsas (numpy array): |S|x|A|x|S| State-action-state reward vector
            policy_vector (numpy array): |S| vector indicating action to take from each
                state
            
            eps (float): State value convergence threshold
        
        Returns:
            (numpy array): |S| state value vector
        """

        v_pi = np.zeros(t_mat.shape[0])

        _iteration = 0
        while True:
            delta = 0
            for s1 in range(t_mat.shape[0]):
                v = v_pi[s1]
                _tmp = 0
                for a in range(t_mat.shape[1]):
                    if policy_vector[s1] != a:
                        continue
                    for s2 in range(t_mat.shape[2]):
                        _tmp += t_mat[s1, a, s2] * (
                            rsa[s1, a] + rsas[s1, a, s2] + rs[s2] + gamma * v_pi[s2]
                        )
                v_pi[s1] = _tmp
                delta = max(delta, np.abs(v - v_pi[s1]))

            if delta < eps:
                break
            _iteration += 1

        return v_pi

    xtr = xtr.as_unpadded
    policy_state_values = []
    for _ in range(num_runs):
        action_vector = np.array([policy.predict(s)[0] for s in xtr.states])
        policy_state_values.append(
            _nb_policy_evaluation(
                xtr.t_mat,
                xtr.gamma,
                *reward.structured(xtr, phi),
                action_vector,
                eps=eps,
            )
        )

    # Average over runs
    return np.mean(policy_state_values, axis=0)


def q_grad_fpi(theta, xtr, phi, tol=1e-3):
    """Estimate the Q-gradient with a Fixed Point Iteration
        
    TODO ajs 07/dec/2020 Handle state-action-state feature functions?
    
    This method uses a Fixed-Point estimate by Neu and Szepesvari 2007, and is
    considered by me to be the 'gold standard' for Q-gradient estimation.
    
    See "Apprenticeship learning using inverse reinforcement learning and gradient
    methods." by Neu and Szepesvari in UAI, 2007.
    
    This method requires |S|x|S|x|A|x|A| updates per iteration, and empirically appears
    to have exponential convergence in the number of iterations. That is,
    δ α O(exp(-1.0 x iteration)).
    
    Args:
        theta (numpy array): Current reward parameters
        xtr (mdp_extras.DiscreteExplicitExtras): MDP definition
        phi (mdp_extras.FeatureFunction): A state-action feature function
    
    Returns:
        (numpy array): |S|x|A|x|φ| Array of partial derivatives δQ(s, a)/dθ
    """

    assert (
        phi.type == phi.Type.OBSERVATION or phi.type == phi.Type.OBSERVATION_ACTION
    ), "Currently, state-action-state features are not supported with this method"

    xtr = xtr.as_unpadded

    # Get optimal *DETERMINISTIC* policy
    # (the fixed point iteration is only valid for deterministic policies)
    reward = Linear(theta)
    q_star = q_vi(xtr, phi, reward)
    pi_star = OptimalPolicy(q_star, stochastic=False)

    @jit(nopython=True)
    def _nb_fpi(states, actions, t_mat, gamma, phi, pi_star, tol):
        """Plain-object core loop for numba optimization
        
        TODO ajs 07/dec/2020 Handle state-action-state feature functions?
        
        Args:
            states (list): States
            actions (list): Actions
            t_mat (numpy array): |S|x|A|x|S| transition matrix
            gamma (float): Discount factor
            phi (numpy array): |S|x|A|x|φ| state-action feature matrix
            pi_star (numpy array): |S|x|A| policy matrix
            tol (float): Convergence threshold
        
        Returns:
            |S|x|A|x|φ| Estimate of gradient of Q function
        """
        # Initialize
        dq_dtheta = phi.copy()

        # Apply fixed point iteration
        it = 0
        while True:
            # Use full-width backups
            dq_dtheta_old = dq_dtheta.copy()
            dq_dtheta[:, :, :] = 0.0
            for s1 in states:
                for a1 in actions:
                    dq_dtheta[s1, a1, :] = phi[s1, a1, :]
                    for s2 in states:
                        for a2 in actions:
                            dq_dtheta[s1, a1, :] += (
                                gamma
                                * t_mat[s1, a1, s2]
                                * pi_star[s2, a2]
                                * dq_dtheta_old[s2, a2, :]
                            )

            delta = np.max(np.abs(dq_dtheta_old.flatten() - dq_dtheta.flatten()))
            it += 1

            if delta <= tol:
                break

        return dq_dtheta

    # Build plain object arrays
    _pi_star = np.zeros((len(xtr.states), len(xtr.actions)))
    _phi = np.zeros((len(xtr.states), len(xtr.actions), len(phi)))
    for s in xtr.states:
        for a in xtr.actions:
            _pi_star[s, a] = pi_star.prob_for_state_action(s, a)
            _phi[s, a, :] = phi(s, a)

    dq_dtheta = _nb_fpi(
        xtr.states, xtr.actions, xtr.t_mat, xtr.gamma, _phi, _pi_star, tol
    )

    return dq_dtheta


def q_grad_sim(
    theta, xtr, phi, max_rollout_length, rollouts_per_sa=100,
):
    """Estimate the Q-gradient with simulation
    
    This method samples many rollouts from the optimal stationary stochastic policy for
    every possible (s, a) pair. This can give arbitrarily bad gradient estimates
    when used with non-episodic MDPs due to the early truncation of rollouts.
    This method also gives arbitrarily bad gradient estimates for terminal states in
    episodic MDPs, unless the max rollout length is set sufficiently high.
    
    This method requires sampling |S|x|A|x(rollouts_per_sa) rollouts from the MDP,
    and is by far the slowest of the Q-gradient estimators.
    
    Args:
        theta (numpy array): Current reward parameters
        xtr (mdp_extras.DiscreteExplicitExtras): MDP definition
        phi (mdp_extras.FeatureFunction): A state-action feature function
        max_rollout_length (int): Maximum rollout length - this value is rather
            arbitrary, but must be set to a large value to give accurate estimates.
        
        rollouts_per_sa (int): Number of rollouts to sample for each (s, a) pair. If
            the environment has deterministic dynamics, it's OK to set this to a small
            number (i.e. 1).
    
    Returns:
        (numpy array): |S|x|A|x|φ| Array of partial derivatives δQ(s, a)/dθ
    """

    xtr = xtr.as_unpadded

    # Get optimal policy
    reward = Linear(theta)
    q_star = q_vi(xtr, phi, reward)
    pi_star = OptimalPolicy(q_star, stochastic=True)

    # Duplicate the MDP, but clear all terminal states
    xtr = copy.deepcopy(xtr)
    xtr._terminal_state_mask[:] = False
    env = DiscreteExplicitLinearEnv(xtr, phi, reward)

    # Calculate expected feature vector under pi for all starting state-action pairs
    dq_dtheta = np.zeros((len(xtr.states), len(xtr.actions), len(phi)))
    for s in xtr.states:
        for a in xtr.actions:
            # Start with desired state, action
            rollouts = pi_star.get_rollouts(
                env,
                rollouts_per_sa,
                max_path_length=max_rollout_length,
                start_state=s,
                start_action=a,
            )
            phi_bar = phi.expectation(rollouts, gamma=xtr.gamma)
            dq_dtheta[s, a, :] = phi_bar

    return dq_dtheta


def q_grad_nd(theta, xtr, phi, dtheta=0.01):
    """Estimate the Q-gradient with 2-point numerical differencing
    
    This method requires solving for 2x|φ| Q* functions
    
    Args:
        theta (numpy array): Current reward parameters
        xtr (mdp_extras.DiscreteExplicitExtras): MDP definition
        phi (mdp_extras.FeatureFunction): A state-action feature function
        
        dtheta (float): Amount to increment reward parameters by
    
    Returns:
        (numpy array): |S|x|A|x|φ| Array of partial derivatives δQ(s, a)/dθ
    """

    xtr = xtr.as_unpadded

    # Calculate expected feature vector under pi for all starting state-action pairs
    dq_dtheta = np.zeros((len(xtr.states), len(xtr.actions), len(phi)))

    # Sweep linear reward parameters
    for theta_i in range(len(theta)):
        # Solve for Q-function with upper and lower reward parameter increments
        theta_lower = theta.copy()
        theta_lower[theta_i] -= dtheta
        q_star_lower = q_vi(xtr, phi, Linear(theta_lower))

        theta_upper = theta.copy()
        theta_upper[theta_i] += dtheta
        q_star_upper = q_vi(xtr, phi, Linear(theta_upper))

        # Take numerical difference to estimate gradient
        dq_dtheta[:, :, theta_i] = (q_star_upper - q_star_lower) / (2.0 * dtheta)

    return dq_dtheta


class Policy(abc.ABC):
    """A simple Policy base class
    
    Provides a .predict(s) method to match the stable-baselines policy API
    """

    def __init__(self):
        """C-tor"""
        raise NotImplementedError

    def save(self, path):
        """Save policy to file"""
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        """Load policy from file"""
        with open(path, "rb") as file:
            _self = pickle.load(file)
            return _self

    def predict(self, s):
        """Predict next action and distribution over states
        
        N.b. This function matches the API of the stabe-baselines policies.
        
        Args:
            s (int): Current state
        
        Returns:
            (int): Sampled action
            (None): Placeholder to ensure this function matches the stable-baselines
                policy interface. Some policies use this argument to return a prediction
                over future state distributions - we do not.
        """
        action = np.random.choice(np.arange(self.q.shape[1]), p=self.prob_for_state(s))
        return action, None

    def path_log_likelihood(self, p):
        """Compute log-likelihood of [(s, a), ..., (s, None)] path under this policy
        
        N.B. - this does NOT account for the likelihood of starting at state s1 under
            the MDP dynamics, or the MDP dynamics themselves
        
        Args:
            p (list): List of state-action tuples
        
        Returns:
            (float): Absolute log-likelihood of the path under this policy
        """

        # We start with probability 1.0
        ll = np.log(1.0)

        # N.b. - final tuple is (s, None), which we skip
        for s, a in p[:-1]:
            log_action_prob = self.log_prob_for_state_action(s, a)
            if np.isneginf(log_action_prob):
                return -np.inf

            ll += log_action_prob

        return ll

    def log_prob_for_state(self, s):
        """Get the action log probability vector for the given state
        
        Args:
            s (int): Current state
        
        Returns:
            (numpy array): Log probability distribution over actions
        """
        raise NotImplementedError

    def prob_for_state(self, s):
        """Get the action probability vector for the given state
        
        Args:
            s (int): Current state
        
        Returns:
            (numpy array): Probability distribution over actions
        """
        return np.exp(self.log_prob_for_state(s))

    def prob_for_state_action(self, s, a):
        """Get the probability for the given state, action
        
        Args:
            s (int): Current state
            a (int): Chosen action
        
        Returns:
            (float): Probability of choosing a from s
        """
        action_probs = self.prob_for_state(s)
        if a > len(action_probs) - 1:
            warnings.warn(
                f"Requested π({a}|{s}), but |A| = {len(action_probs)} - returning 1.0. If {a} is a dummy action you can safely ignore this warning.",
                PaddedMDPWarning,
            )
            return 1.0
        else:
            return action_probs[a]

    def log_prob_for_state_action(self, s, a):
        """Get the log probability for the given state, action
        
        Args:
            s (int): Current state
            a (int): Chosen action
        
        Returns:
            (float): Log probability of choosing a from s
        """
        log_action_probs = self.log_prob_for_state(s)
        if a > len(log_action_probs) - 1:
            warnings.warn(
                f"Requested log π({a}|{s}), but |A| = {len(log_action_probs)} - returning 0.0. If {a} is a dummy action you can safely ignore this warning.",
                PaddedMDPWarning,
            )
            return 0.0
        else:
            return log_action_probs[a]

    def get_rollouts(
        self, env, num, max_path_length=None, start_state=None, start_action=None
    ):
        """Sample state-action rollouts from this policy in the provided environment
        
        Args:
            env (gym.Env): Environment
            num (int): Number of rollouts to sample
            
            max_path_length (int): Optional maximum path length - episodes will be
                prematurely terminated after this many time steps
            start_state (any): Optional starting state for the policy - Warning: this
                functionality isn't actually supported by the OpenAI Gym class, we just
                hope that the 'env' definition has a writeable '.state' parameter
            start_action (any): Optional starting action for the policy
        
        Returns:
            (list): List of state-action rollouts
        """
        rollouts = []
        for _ in range(num):
            rollout = []
            s = env.reset()
            if start_state is not None:
                """XXX ajs 28/Oct/2020 The OpenAI Gym interface doesn't actually expose
                a `state` parameter, making it impossible to force a certain starting
                state reliably. Someone needs to publish a standardized MDP interface
                that isn't a steaming pile of b*******.
                """
                assert (
                    "state" in env.__dir__()
                ), "XXX This environment doesn't have a 'state' property - I'm unable to force it into a desired starting state!"
                s = env.state = start_state

            for t in it.count():
                if t == 0 and start_action is not None:
                    a = start_action
                else:
                    a, _ = self.predict(s)
                s2, r, done, info = env.step(a)
                rollout.append((s, a))
                s = s2
                if done or (max_path_length is not None and t == max_path_length - 2):
                    break
            rollout.append((s, None))
            rollouts.append(rollout)
        return rollouts


class EpsilonGreedyPolicy(Policy):
    """An Epsilon Greedy Policy wrt. a provided Q function
    """

    def __init__(self, q, epsilon=0.1):
        """C-tor
        
        Args:
            q (numpy array): |S|x|A| Q-matrix
            epsilon (float): Probability of taking a random action. Set to 0 to create
                an optimal stochsatic policy. Specifically,
                    Epsilon == 0.0 will make the policy sample between equally good
                        (== Q value) actions. If a single action has the highest Q
                        value, that action will always be chosen
                    Epsilon > 0.0 will make the policy act in an epsilon greedy
                        fashion - i.e. a random action is chosen with probability
                        epsilon, and an optimal action is chosen with probability
                        (1 - epsilon) + (epsilon / |A|).
        """
        self.q = q
        self.epsilon = epsilon

    def prob_for_state(self, s):
        """Get the action probability vector for the given state
        
        Args:
            s (int): Current state
        
        Returns:
            (numpy array): Probability distribution over actions, respecting the
                self.stochastic and self.epsilon parameters
        """

        num_states, num_actions = self.q.shape
        if s > num_states - 1:
            warnings.warn(
                f"Requested π(*|{s}), but |S| = {num_states}, returning [1.0, ...]. If {s} is a dummy state you can safely ignore this warning.",
                PaddedMDPWarning,
            )
            return np.ones(num_actions)

        # Get a list of the optimal actions
        action_values = self.q[s, :]
        best_action_value = np.max(action_values)
        best_action_mask = action_values == best_action_value
        best_actions = np.where(best_action_mask)[0]

        # Prepare action probability vector
        p = np.zeros(self.q.shape[1])

        # All actions share probability epsilon
        p[:] += self.epsilon / self.q.shape[1]

        # Optimal actions share additional probability (1 - epsilon)
        p[best_actions] += (1 - self.epsilon) / len(best_actions)

        return p

    def log_prob_for_state(self, s):
        """Get the action log probability vector for the given state
        
        Args:
            s (int): Current state
        
        Returns:
            (numpy array): Log probability distribution over actions
        """
        return np.log(self.prob_for_state(s))


class OptimalPolicy(EpsilonGreedyPolicy):
    """An optimal policy - can be deterministic or stochastic"""

    def __init__(self, q, stochastic=True, q_precision=None):
        """C-tor
        
        Args:
            q (numpy array): |S|x|A| Q-matrix
            
            stochastic (bool): If true, this policy will sample amongst optimal actions.
                Otherwise, the first optimal action will always be chosen.
            q_precision (int): Precision level in digits of the q-function. If a
                stochastic optimal policy is requested, Q-values will be rounded to
                this many digits before equality checks. Set to None to disable.
        """
        super().__init__(q, epsilon=0.0)

        self.stochastic = stochastic
        self.q_precision = q_precision

    def prob_for_state(self, s):

        num_states, num_actions = self.q.shape
        if s > num_states - 1:
            warnings.warn(
                f"Requested π(*|{s}), but |S| = {num_states}, returning [1.0, ...]. If {s} is a dummy state you can safely ignore this warning.",
                PaddedMDPWarning,
            )
            return np.ones(num_actions)

        if self.stochastic:

            if self.q_precision is None:
                p = super().prob_for_state(s)
            else:
                # Apply q_precision rounding to the q-function
                action_values = np.array(
                    [round(v, self.q_precision) for v in self.q[s]]
                )
                best_action_value = np.max(action_values)
                best_action_mask = action_values == best_action_value
                best_actions = np.where(best_action_mask)[0]

                p = np.zeros(self.q.shape[1])
                p[best_actions] += 1.0 / len(best_actions)

        if not self.stochastic:
            # Always select the first optimal action
            p = super().prob_for_state(s)
            a_star = np.where(p != 0)[0][0]
            p *= 0
            p[a_star] = 1.0
        return p

    def log_prob_for_state(self, s):
        """Get the action log probability vector for the given state
        
        Args:
            s (int): Current state
        
        Returns:
            (numpy array): Log probability distribution over actions
        """
        return np.log(self.prob_for_state())


class BoltzmannExplorationPolicy(Policy):
    """A Boltzmann exploration policy wrt. a provided Q function
    """

    def __init__(self, q, scale=1.0):
        """C-tor
        
        Args:
            q (numpy array): |S|x|A| Q-matrix
            scale (float): Temperature scaling factor on the range [0, inf).
                Actions are chosen proportional to exp(scale * Q(s, a)), so...
                 * Scale > 1.0 will exploit optimal actions more often
                 * Scale == 1.0 samples actions proportional to the exponent of their
                    value
                 * Scale < 1.0 will explore sub-optimal actions more often
                 * Scale == 0.0 will uniformly sample actions
                 * Scale < 0.0 will prefer non-optimal actions
        """
        self.q = q
        self.scale = scale

    def prob_for_state(self, s):
        """Get the action probability vector for the given state
        
        Args:
            s (int): Current state
        
        Returns:
            (numpy array): Probability distribution over actions, respecting the
                self.stochastic and self.epsilon parameters
        """
        return np.exp(self.log_prob_for_state(s))

    def log_prob_for_state(self, s):

        num_states, num_actions = self.q.shape
        if s > num_states - 1:
            warnings.warn(
                f"Requested π(*|{s}), but |S| = {num_states}, returning [1.0, ...]. If {s} is a dummy state you can safely ignore this warning.",
                PaddedMDPWarning,
            )
            return np.ones(num_actions)

        log_prob = self.scale * self.q[s]
        total_log_prob = np.log(np.sum(np.exp(log_prob)))
        log_prob -= total_log_prob
        return log_prob
