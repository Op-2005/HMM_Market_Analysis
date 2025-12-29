# This file contains the discrete HMM implementation with categorical emissions 

import torch
import numpy as np
from .base_hmm import BaseHMM

class DiscreteHMM(BaseHMM):
    def __init__(self, num_states: int, num_observations: int, 
                 T=None, E=None, T0=None, device='cpu', epsilon=1e-10):
        super().__init__(num_states, T=T, T0=T0, device=device, epsilon=epsilon)
        self.num_observations = num_observations
        
        if E is not None:
            self.E = self._normalize_tensor(E, is_matrix=True)
        else:
            self.E = torch.ones((num_states, num_observations), dtype=torch.float64) / num_observations
    
    def emission_prob(self, observation, state):
        obs_idx = int(observation)
        if obs_idx < 0:
            obs_idx = 0
        if obs_idx >= self.num_observations:
            obs_idx = self.num_observations - 1
        return float(self.E[state, obs_idx])
    
    def validate(self, raise_on_error=True):
        base_valid = super().validate(raise_on_error=False)
        errors = []
        tol = 1e-6
        
        if self.E.shape[0] != self.num_states:
            errors.append(f"Emission matrix E has incorrect first dimension: {self.E.shape[0]}")
        if self.E.shape[1] != self.num_observations:
            errors.append(f"Emission matrix E has incorrect second dimension: {self.E.shape[1]}")
        if torch.isnan(self.E).any() or torch.isinf(self.E).any():
            errors.append("Emission matrix E contains NaN or Inf values")
        if (self.E < 0).any():
            errors.append("Emission matrix E contains negative values")
        
        E_row_sums = self.E.sum(dim=1)
        if not torch.allclose(E_row_sums, torch.ones(self.num_states, dtype=torch.float64), atol=tol):
            errors.append(f"Emission matrix E rows do not sum to 1")
        
        if errors:
            error_msg = "DiscreteHMM validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            if raise_on_error:
                raise ValueError(error_msg)
            else:
                print(f"WARNING: {error_msg}")
                return False
        return base_valid
    
    def get_params(self):
        params = super().get_params()
        params['E'] = self.E.clone()
        params['num_observations'] = self.num_observations
        return params
    
    def set_params(self, params):
        super().set_params(params)
        if 'E' in params:
            self.E = self._normalize_tensor(params['E'], is_matrix=True)
        if 'num_observations' in params:
            self.num_observations = params['num_observations']
    
    def save(self, filepath):
        torch.save({
            'T': self.T,
            'E': self.E,
            'T0': self.T0,
            'num_states': self.num_states,
            'num_observations': self.num_observations,
            'model_class': 'DiscreteHMM'
        }, filepath)
        print(f"DiscreteHMM saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        model = cls(
            num_states=checkpoint['num_states'],
            num_observations=checkpoint['num_observations'],
            T=checkpoint['T'],
            E=checkpoint['E'],
            T0=checkpoint['T0']
        )
        print(f"DiscreteHMM loaded from {filepath}")
        return model
