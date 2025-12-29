# Utility functions for regime mapping and HMM parameter initialization.
import numpy as np
import torch
from typing import Dict, Tuple, Optional


class RegimeMapper:
    @staticmethod
    def map_by_correlation(states: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
        unique_states = np.unique(states)
        state_correlations = {}
        
        for state in unique_states:
            # Create binary array for this state
            state_presence = (states == state).astype(int)
            # Calculate correlation with actual labels
            if np.std(state_presence) > 0 and np.std(labels) > 0:
                corr = np.corrcoef(state_presence, labels)[0, 1]
                corr = 0.0 if np.isnan(corr) else corr
            else:
                corr = 0.0
            state_correlations[state] = corr
        
        # Assign states to bear/bull based on correlation
        state_to_label = {}
        for state in unique_states:
            # Positive correlation with bull market -> predict bull
            state_to_label[state] = 1 if state_correlations[state] >= 0 else 0
        
        return state_to_label
    
    @staticmethod
    def map_by_majority(states: np.ndarray, labels: np.ndarray, 
                       threshold: float = 0.5) -> Dict[int, int]:
       
        unique_states = np.unique(states)
        state_to_label = {}
        
        for state in unique_states:
            mask = (states == state)
            avg_label = np.mean(labels[mask])
            state_to_label[state] = 1 if avg_label >= threshold else 0
        
        return state_to_label
    
    @staticmethod
    def map_by_returns(states: np.ndarray, returns: np.ndarray,
                      threshold: float = 0.0) -> Dict[int, int]:
        unique_states = np.unique(states)
        state_to_label = {}
        
        for state in unique_states:
            mask = (states == state)
            mean_return = np.mean(returns[mask])
            state_to_label[state] = 1 if mean_return >= threshold else 0
        
        return state_to_label

def initialize_random_params(num_states: int, num_observations: int,
                            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize HMM parameters randomly with Dirichlet priors.
    
    Parameters:
    -----------
    num_states : int
        Number of hidden states
    num_observations : int
        Number of observations
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    T : np.ndarray
        Transition matrix (num_states x num_states)
    E : np.ndarray
        Emission matrix (num_states x num_observations)
    T0 : np.ndarray
        Initial state distribution (num_states,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Transition matrix with slight bias toward staying in same state
    T = np.random.uniform(0, 1, (num_states, num_states))
    # Add self-transition bias
    for i in range(num_states):
        T[i, i] += 1.0
    T = T / T.sum(axis=1, keepdims=True)
    
    # Emission matrix
    E = np.random.uniform(0, 1, (num_states, num_observations))
    E = E / E.sum(axis=1, keepdims=True)
    
    # Initial state distribution
    T0 = np.random.uniform(0, 1, num_states)
    T0 = T0 / T0.sum()
    
    return T, E, T0


def initialize_structured_params(num_states: int, num_observations: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Structured emission matrix for market regimes
    T = np.ones((num_states, num_states)) / num_states
    for i in range(num_states):
        T[i, i] = 0.4  # Increased self-transition probability
    T = T / T.sum(axis=1, keepdims=True)
    
    # Structured emission matrix
    E = np.ones((num_states, num_observations)) / num_observations
    
    # Structure emissions for 5 states (if applicable)
    if num_states >= 5 and num_observations >= 10:
        # Strong bull market - higher probability for low volatility
        E[0, :num_observations//5] = 0.7 / (num_observations//5)
        E[0, num_observations//5:] = 0.3 / (num_observations - num_observations//5)
        
        # Moderate bull market - higher probability for low-medium volatility
        mid_low = num_observations//5
        mid_high = 2*num_observations//5
        E[1, mid_low:mid_high] = 0.6 / (mid_high - mid_low)
        E[1, :mid_low] = 0.2 / mid_low
        E[1, mid_high:] = 0.2 / (num_observations - mid_high)
        
        # Sideways market - higher probability for medium volatility
        mid_low = 2*num_observations//5
        mid_high = 3*num_observations//5
        E[2, mid_low:mid_high] = 0.6 / (mid_high - mid_low)
        E[2, :mid_low] = 0.2 / mid_low
        E[2, mid_high:] = 0.2 / (num_observations - mid_high)
        
        # Moderate bear market - higher probability for medium-high volatility
        mid_low = 3*num_observations//5
        mid_high = 4*num_observations//5
        E[3, mid_low:mid_high] = 0.6 / (mid_high - mid_low)
        E[3, :mid_low] = 0.2 / mid_low
        E[3, mid_high:] = 0.2 / (num_observations - mid_high)
        
        # Strong bear market - higher probability for high volatility
        E[4, 4*num_observations//5:] = 0.7 / (num_observations - 4*num_observations//5)
        E[4, :4*num_observations//5] = 0.3 / (4*num_observations//5)
    
    # Normalize
    E = E / E.sum(axis=1, keepdims=True)
    
    # Initial state distribution - balanced
    T0 = np.ones(num_states) / num_states
    if num_states >= 5:
        # Slight bias toward first 3 states (bull market)
        T0[:3] = 0.6 / 3
        T0[3:] = 0.4 / (num_states - 3)
    
    return T, E, T0

def initialize_kmeans_params(observations: np.ndarray, num_states: int,
                            num_observations: int,
                            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Use K-means to get initial state assignments
    kmeans = KMeans(n_clusters=num_states, random_state=random_state)
    initial_states = kmeans.fit_predict(observations.reshape(-1, 1))
    
    # Initialize transition matrix based on state sequence
    T = np.zeros((num_states, num_states))
    for t in range(len(initial_states) - 1):
        T[initial_states[t], initial_states[t+1]] += 1
    # Normalize with Laplace smoothing
    T = T + 0.1
    T = T / T.sum(axis=1, keepdims=True)
    
    # Initialize emission matrix based on state-observation co-occurrences
    E = np.zeros((num_states, num_observations))
    for t in range(len(observations)):
        obs_idx = int(observations[t])
        if obs_idx >= num_observations:
            obs_idx = num_observations - 1
        E[initial_states[t], obs_idx] += 1
    # Normalize with Laplace smoothing
    E = E + 0.1
    E = E / E.sum(axis=1, keepdims=True)
    
    # Initial state distribution
    T0 = np.bincount(initial_states, minlength=num_states).astype(float)
    T0 = T0 + 0.1
    T0 = T0 / T0.sum()
    
    return T, E, T0
