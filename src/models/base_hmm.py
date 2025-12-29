# This file contains the abstract base class for HMM implementations with common methods
import torch
import numpy as np
from abc import ABC, abstractmethod

class BaseHMM(ABC):
    def __init__(self, num_states: int, T=None, T0=None, device='cpu', epsilon=1e-10):
        self.num_states = num_states
        self.device = device
        self.epsilon = epsilon
        
        if T is not None:
            self.T = self._normalize_tensor(T, is_matrix=True)
        else:
            self.T = torch.ones((num_states, num_states), dtype=torch.float64) / num_states
            
        if T0 is not None:
            self.T0 = self._normalize_tensor(T0, is_matrix=False)
        else:
            self.T0 = torch.ones(num_states, dtype=torch.float64) / num_states
    
    def _normalize_tensor(self, tensor, is_matrix=True):
        if isinstance(tensor, np.ndarray):
            tensor = torch.tensor(tensor, dtype=torch.float64)
        elif isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().clone().double()
        else:
            tensor = torch.tensor(tensor, dtype=torch.float64)
        
        tensor = torch.clamp(tensor, min=self.epsilon)
        if is_matrix:
            tensor = tensor / tensor.sum(dim=1, keepdim=True)
        else:
            tensor = tensor / tensor.sum()
        return tensor
    
    @abstractmethod
    def emission_prob(self, observation, state):
        pass
    
    def viterbi_inference(self, observations):
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.int64)
        
        N = len(observations)
        S = self.num_states
        
        path_states = torch.zeros((N, S), dtype=torch.int64)
        path_scores = torch.zeros((N, S), dtype=torch.float64)
        
        # Get emission probabilities
        log_emissions = []
        for t in range(N):
            obs_probs = torch.zeros(S, dtype=torch.float64)
            for s in range(S):
                obs_probs[s] = self.emission_prob(observations[t], s)
            log_emissions.append(torch.log(torch.clamp(obs_probs, min=self.epsilon)))
        log_emissions = torch.stack(log_emissions)
        
        # Initialize
        log_T0 = torch.log(torch.clamp(self.T0, min=self.epsilon))
        path_scores[0] = log_T0 + log_emissions[0]
        
        # Forward pass
        log_T = torch.log(torch.clamp(self.T, min=self.epsilon))
        for t in range(1, N):
            for s in range(S):
                scores = path_scores[t-1] + log_T[:, s]
                best_prev = torch.argmax(scores)
                path_states[t, s] = best_prev
                path_scores[t, s] = scores[best_prev] + log_emissions[t, s]
        
        # Backward pass
        states_seq = torch.zeros(N, dtype=torch.int64)
        states_seq[-1] = torch.argmax(path_scores[-1])
        for t in range(N-2, -1, -1):
            states_seq[t] = path_states[t+1, states_seq[t+1]]
        
        return states_seq, torch.exp(path_scores)
    
    def forward_backward(self, observations):
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.int64)
        
        N = len(observations)
        S = self.num_states
        
        # Get emission probabilities in log-space
        log_obs = []
        for t in range(N):
            obs_probs = torch.zeros(S, dtype=torch.float64)
            for s in range(S):
                obs_probs[s] = self.emission_prob(observations[t], s)
            log_obs.append(torch.log(torch.clamp(obs_probs, min=self.epsilon)))
        log_obs = torch.stack(log_obs)
        
        log_T = torch.log(torch.clamp(self.T, min=self.epsilon))
        log_T0 = torch.log(torch.clamp(self.T0, min=self.epsilon))
        
        # Forward pass
        log_forward = torch.zeros((N, S), dtype=torch.float64)
        log_forward[0] = log_T0 + log_obs[0]
        log_forward[0] = log_forward[0] - torch.logsumexp(log_forward[0], dim=0)
        
        for t in range(1, N):
            log_transitions = log_forward[t-1].unsqueeze(1) + log_T
            log_forward[t] = torch.logsumexp(log_transitions, dim=0) + log_obs[t]
            log_forward[t] = log_forward[t] - torch.logsumexp(log_forward[t], dim=0)
        
        # Backward pass
        log_backward = torch.zeros((N, S), dtype=torch.float64)
        for t in range(N-2, -1, -1):
            log_transitions = log_T + (log_obs[t+1] + log_backward[t+1]).unsqueeze(0)
            log_backward[t] = torch.logsumexp(log_transitions, dim=1)
            log_backward[t] = log_backward[t] - torch.logsumexp(log_backward[t], dim=0)
        
        forward = torch.exp(log_forward)
        backward = torch.exp(log_backward)
        forward = forward / (forward.sum(dim=1, keepdim=True) + self.epsilon)
        backward = backward / (backward.sum(dim=1, keepdim=True) + self.epsilon)
        
        return forward, backward
    
    def validate(self, raise_on_error=True):
        errors = []
        tol = 1e-6
        
        if self.T.shape[0] != self.num_states or self.T.shape[1] != self.num_states:
            errors.append(f"Transition matrix T has incorrect shape: {self.T.shape}")
        if self.T0.shape[0] != self.num_states:
            errors.append(f"Initial state distribution T0 has incorrect shape: {self.T0.shape}")
        if torch.isnan(self.T).any() or torch.isinf(self.T).any():
            errors.append("Transition matrix T contains NaN or Inf values")
        if torch.isnan(self.T0).any() or torch.isinf(self.T0).any():
            errors.append("Initial state distribution T0 contains NaN or Inf values")
        if (self.T < 0).any():
            errors.append("Transition matrix T contains negative values")
        if (self.T0 < 0).any():
            errors.append("Initial state distribution T0 contains negative values")
        
        T_row_sums = self.T.sum(dim=1)
        if not torch.allclose(T_row_sums, torch.ones(self.num_states, dtype=torch.float64), atol=tol):
            errors.append(f"Transition matrix T rows do not sum to 1")
        T0_sum = self.T0.sum()
        if not torch.allclose(T0_sum, torch.tensor(1.0, dtype=torch.float64), atol=tol):
            errors.append(f"Initial state distribution T0 does not sum to 1")
        
        if errors:
            error_msg = "HMM validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            if raise_on_error:
                raise ValueError(error_msg)
            else:
                print(f"WARNING: {error_msg}")
                return False
        return True
    
    def get_params(self):
        return {
            'T': self.T.clone(),
            'T0': self.T0.clone(),
            'num_states': self.num_states
        }
    
    def set_params(self, params):
        if 'T' in params:
            self.T = self._normalize_tensor(params['T'], is_matrix=True)
        if 'T0' in params:
            self.T0 = self._normalize_tensor(params['T0'], is_matrix=False)
        if 'num_states' in params:
            self.num_states = params['num_states']
    
    def save(self, filepath):
        torch.save({
            'T': self.T,
            'T0': self.T0,
            'num_states': self.num_states,
            'model_class': self.__class__.__name__
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        raise NotImplementedError("load() must be implemented by subclass")
