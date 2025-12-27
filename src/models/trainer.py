"""
HMM Trainer for Baum-Welch EM algorithm.
"""
import torch
import numpy as np
import time
from .discrete_hmm import DiscreteHMM


class HMMTrainer:
    """
    Handles Baum-Welch training, convergence checking, and early stopping.
    """
    
    def __init__(self, model: DiscreteHMM, epsilon=1e-3, max_steps=100, verbose=True):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        model : DiscreteHMM
            HMM model to train
        epsilon : float
            Convergence threshold
        max_steps : int
            Maximum number of EM iterations
        verbose : bool
            Whether to print progress
        """
        self.model = model
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.verbose = verbose
        self.history = {
            'log_likelihood': [],
            'step_times': [],
            'converged': False
        }
    
    def fit(self, observations, val_observations=None, patience=5, min_delta=1e-6):
        """
        Train model using Baum-Welch EM algorithm.
        
        Parameters:
        -----------
        observations : torch.Tensor or array-like
            Training observations
        val_observations : torch.Tensor or array-like, optional
            Validation observations for early stopping
        patience : int
            Number of iterations without improvement before early stopping
        min_delta : float
            Minimum change to qualify as improvement
        
        Returns:
        --------
        converged : bool
            Whether training converged
        """
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.int64)
        
        N = len(observations)
        best_log_likelihood = float('-inf')
        patience_counter = 0
        best_params = None
        
        if self.verbose:
            print(f"Starting Baum-Welch EM with {self.max_steps} max steps")
        
        start_time = time.time()
        
        for step in range(self.max_steps):
            step_start = time.time()
            
            try:
                # E-step and M-step
                converged = self._expectation_maximization_step(observations)
                
                step_time = time.time() - step_start
                self.history['step_times'].append(step_time)
                
                # Calculate log-likelihood on validation set if provided
                if val_observations is not None:
                    log_likelihood = self._calculate_log_likelihood(val_observations)
                    self.history['log_likelihood'].append(log_likelihood)
                    
                    # Check for improvement
                    if log_likelihood > best_log_likelihood + min_delta:
                        best_log_likelihood = log_likelihood
                        patience_counter = 0
                        # Save best parameters
                        best_params = {
                            'T': self.model.T.clone(),
                            'E': self.model.E.clone(),
                            'T0': self.model.T0.clone()
                        }
                    else:
                        patience_counter += 1
                    
                    if self.verbose:
                        print(f"  Step {step+1}/{self.max_steps} completed in {step_time:.2f}s, "
                              f"val_log_likelihood={log_likelihood:.4f}, patience={patience_counter}/{patience}")
                    
                    # Early stopping
                    if patience_counter >= patience:
                        if self.verbose:
                            print(f"Early stopping at step {step+1}")
                        if best_params is not None:
                            self.model.T = best_params['T']
                            self.model.E = best_params['E']
                            self.model.T0 = best_params['T0']
                        self.history['converged'] = False
                        break
                else:
                    if self.verbose:
                        print(f"  Step {step+1}/{self.max_steps} completed in {step_time:.2f}s")
                
                if converged:
                    if self.verbose:
                        print(f'Converged at step {step+1}')
                    self.history['converged'] = True
                    break
                    
            except Exception as step_error:
                if self.verbose:
                    print(f"Error in EM step {step+1}: {str(step_error)}")
                continue
        
        total_time = time.time() - start_time
        if self.verbose:
            print(f"Total training time: {total_time:.2f} seconds")
        
        return self.history['converged']
    
    def _expectation_maximization_step(self, observations):
        """
        Single EM step: expectation (forward-backward) and maximization (parameter update).
        """
        N = len(observations)
        
        # E-step: Forward-backward algorithm
        forward, backward = self.model.forward_backward(observations)
        
        # M-step: Re-estimate parameters
        new_T0, new_T = self._re_estimate_transition(observations, forward, backward)
        new_E = self._re_estimate_emission(observations, forward, backward)
        
        # Check convergence
        converged = self._check_convergence(new_T0, new_T, new_E)
        
        # Update parameters
        self.model.T0 = new_T0
        self.model.T = new_T
        self.model.E = new_E
        
        return converged
    
    def _re_estimate_transition(self, observations, forward, backward):
        """Re-estimate transition matrix and initial state distribution."""
        N = len(observations)
        S = self.model.num_states
        
        # Calculate gamma (state occupancy probabilities)
        gamma = forward * backward
        gamma = gamma / (gamma.sum(dim=1, keepdim=True) + self.model.epsilon)
        
        # Re-estimate initial state distribution
        T0_new = gamma[0].clone()
        T0_new = torch.clamp(T0_new, min=self.model.epsilon)
        T0_new = T0_new / T0_new.sum()
        
        # Re-estimate transition matrix
        # Calculate xi (state transition probabilities)
        log_T = torch.log(torch.clamp(self.model.T, min=self.model.epsilon))
        T_new = torch.zeros_like(self.model.T)
        
        for t in range(N - 1):
            obs_idx = int(observations[t + 1])
            if obs_idx >= self.model.num_observations:
                obs_idx = self.model.num_observations - 1
            
            emission_prob = self.model.E[:, obs_idx]
            log_emission = torch.log(torch.clamp(emission_prob, min=self.model.epsilon))
            
            # Calculate xi[t, i, j] = P(z_t=i, z_{t+1}=j | x)
            log_xi = (forward[t].unsqueeze(1) + log_T + 
                     (log_emission + backward[t+1]).unsqueeze(0))
            log_xi = log_xi - torch.logsumexp(log_xi.view(-1), dim=0)
            xi = torch.exp(log_xi)
            
            T_new += xi
        
        # Normalize
        T_new = torch.clamp(T_new, min=self.model.epsilon)
        T_new = T_new / T_new.sum(dim=1, keepdim=True)
        
        return T0_new, T_new
    
    def _re_estimate_emission(self, observations, forward, backward):
        """Re-estimate emission matrix."""
        N = len(observations)
        S = self.model.num_states
        O = self.model.num_observations
        
        # Calculate gamma (state occupancy probabilities)
        gamma = forward * backward
        gamma = gamma / (gamma.sum(dim=1, keepdim=True) + self.model.epsilon)
        
        # Re-estimate emission matrix
        E_new = torch.zeros_like(self.model.E)
        
        for t in range(N):
            obs_idx = int(observations[t])
            if obs_idx >= O:
                obs_idx = O - 1
            
            # Add state probabilities to corresponding emission entry
            E_new[:, obs_idx] += gamma[t]
        
        # Normalize
        E_new = torch.clamp(E_new, min=self.model.epsilon)
        E_new = E_new / E_new.sum(dim=1, keepdim=True)
        
        return E_new
    
    def _check_convergence(self, new_T0, new_T, new_E):
        """Check if parameters have converged."""
        with torch.no_grad():
            delta_T0 = torch.max(torch.abs(self.model.T0 - new_T0)).item() < self.epsilon
            delta_T = torch.max(torch.abs(self.model.T - new_T)).item() < self.epsilon
            delta_E = torch.max(torch.abs(self.model.E - new_E)).item() < self.epsilon
        
        return delta_T0 and delta_T and delta_E
    
    def _calculate_log_likelihood(self, observations):
        """Calculate log-likelihood of observations."""
        forward, _ = self.model.forward_backward(observations)
        # Log-likelihood is log(sum of forward probabilities at final time step)
        log_likelihood = torch.logsumexp(torch.log(torch.clamp(forward[-1], min=self.model.epsilon)), dim=0).item()
        return log_likelihood

