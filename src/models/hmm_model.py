#This file contains the main HMM class with Viterbi inference and Baum-Welch training. Backward-compatible API.
import torch
import numpy as np
import time

class HiddenMarkovModel(object):
    def __init__(self, T, E, T0, device='cpu', epsilon=0.001, maxStep=10):
        self.device = 'cpu'
        self.maxStep = maxStep
        self.epsilon = epsilon
        self.S = T.shape[0]  # Number of states
        self.O = E.shape[0]  # Number of observations
        self.prob_state_1 = []
        
        # Convert to numpy then tensor for consistency
        if isinstance(T, torch.Tensor):
            T = T.detach().cpu().numpy()
        if isinstance(E, torch.Tensor):
            E = E.detach().cpu().numpy()
        if isinstance(T0, torch.Tensor):
            T0 = T0.detach().cpu().numpy()
            
        self.E = torch.tensor(E, dtype=torch.float64)
        self.T = torch.tensor(T, dtype=torch.float64)
        self.T0 = torch.tensor(T0, dtype=torch.float64)
        
        # Normalize to valid probability distributions (ensure rows sum to 1 and no zeros)
        epsilon = 1e-10
        self.T = torch.clamp(self.T, min=epsilon)
        self.E = torch.clamp(self.E, min=epsilon)
        self.T0 = torch.clamp(self.T0, min=epsilon)
        self.T = self.T / self.T.sum(dim=1, keepdim=True)  # Normalize transition matrix rows
        self.E = self.E / self.E.sum(dim=1, keepdim=True)  # Normalize emission matrix rows
        self.T0 = self.T0 / self.T0.sum()  # Normalize initial distribution

    def viterbi_inference(self, x):
        # Viterbi algorithm finds the most likely sequence of hidden states
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.int64)
        
        self.N = len(x)
        pathStates = torch.zeros([self.N, self.S], dtype=torch.float64)  # Stores best previous state
        pathScores = torch.zeros([self.N, self.S], dtype=torch.float64)  # Stores best path score
        states_seq = torch.zeros([self.N], dtype=torch.int64)
        
        # Get emission probabilities for each observation (probability of seeing obs given each state)
        obs_prob_full = []
        for t in range(self.N):
            obs_idx = x[t]
            if obs_idx >= self.E.shape[1]:  # Clip out-of-bounds observations
                obs_idx = self.E.shape[1] - 1
            obs_prob = self.E[:, obs_idx]
            obs_prob_full.append(obs_prob)
        
        obs_prob_full = torch.stack(obs_prob_full)
        obs_prob_full = torch.log(obs_prob_full)  # Convert to log space for numerical stability
        
        # Initialize first step with initial state distribution
        pathScores[0] = torch.log(self.T0) + obs_prob_full[0]
        
        # Forward pass: find best path score at each step
        for step, obs_prob in enumerate(obs_prob_full[1:]):
            # For each state, find which previous state gives best score
            belief = pathScores[step, :].view(-1, 1) + torch.log(self.T)
            pathStates[step + 1] = torch.argmax(belief, 0)  # Store best previous state
            pathScores[step + 1] = torch.max(belief, 0)[0] + obs_prob  # Store best score
        
        # Backward pass: trace back through pathStates to get final sequence
        states_seq[self.N - 1] = torch.argmax(pathScores[self.N-1, :], 0)
        for step in range(self.N - 1, 0, -1):
            state = states_seq[step]
            states_seq[step - 1] = pathStates[step][state]  # Look up best previous state
        
        return states_seq, torch.exp(pathScores)

    def forward_backward(self, obs_prob_seq):
        # Forward-backward algorithm in log space to prevent numerical underflow
        if len(obs_prob_seq.shape) != 2 or obs_prob_seq.shape[1] != self.S:
            raise ValueError(f"Expected shape (seq_len, {self.S}), got {obs_prob_seq.shape}")
        
        self.N = len(obs_prob_seq)
        epsilon = 1e-10
        
        # Clamp probabilities to avoid log(0)
        obs_prob_seq = torch.clamp(obs_prob_seq, min=epsilon, max=1.0)
        log_obs = torch.log(obs_prob_seq)
        log_T = torch.log(torch.clamp(self.T, min=epsilon))
        log_T0 = torch.log(torch.clamp(self.T0, min=epsilon))
        
        log_forward = torch.zeros([self.N, self.S], dtype=torch.float64)
        log_backward = torch.zeros([self.N, self.S], dtype=torch.float64)
        
        # Forward pass: compute probability of being in each state given observations up to time t
        log_forward[0, :] = log_T0 + log_obs[0, :]
        log_forward[0, :] = log_forward[0, :] - torch.logsumexp(log_forward[0, :], dim=0)  # Normalize
        
        for t in range(1, self.N):
            # Sum over all possible previous states using logsumexp
            log_transitions = log_forward[t-1, :].unsqueeze(1) + log_T
            log_forward[t, :] = torch.logsumexp(log_transitions, dim=0) + log_obs[t, :]
            log_forward[t, :] = log_forward[t, :] - torch.logsumexp(log_forward[t, :], dim=0)  # Normalize
        
        # Backward pass: compute probability of future observations given current state
        for t in range(self.N-2, -1, -1):
            log_transitions = log_T + (log_obs[t+1, :] + log_backward[t+1, :]).unsqueeze(0)
            log_backward[t, :] = torch.logsumexp(log_transitions, dim=1)
            log_backward[t, :] = log_backward[t, :] - torch.logsumexp(log_backward[t, :], dim=0)  # Normalize
        
        # Convert back from log space to probabilities
        self.forward = torch.exp(log_forward)
        self.backward = torch.exp(log_backward)
        self.forward = self.forward / (self.forward.sum(dim=1, keepdim=True) + epsilon)
        self.backward = self.backward / (self.backward.sum(dim=1, keepdim=True) + epsilon)

    def re_estimate_transition(self, x):
        # Re-estimate transition matrix using expected counts from forward-backward
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.int64)
        
        self.M = torch.zeros([self.N - 1, self.S, self.S], dtype=torch.float64)  # Expected transition counts
        
        for t in range(self.N - 1):
            obs_idx_t1 = x[t + 1]
            if obs_idx_t1 >= self.E.shape[1]:
                obs_idx_t1 = self.E.shape[1] - 1
            emission_probs_t1 = self.E[:, obs_idx_t1]  # Emission probs for observation at t+1
            
            # Compute denominator (probability of observation sequence)
            tmp_0 = torch.matmul(self.forward[t].unsqueeze(0), self.T)
            tmp_1 = tmp_0 * emission_probs_t1.unsqueeze(0)
            denom = torch.matmul(tmp_1, self.backward[t + 1].unsqueeze(1)).squeeze()
            
            # Compute expected transition counts for each state pair
            trans_re_estimate = torch.zeros([self.S, self.S], dtype=torch.float64)
            for i in range(self.S):
                numer = self.forward[t, i] * self.T[i, :] * emission_probs_t1 * self.backward[t+1]
                trans_re_estimate[i] = numer / denom
            self.M[t] = trans_re_estimate
        
        # Sum expected counts over time and normalize
        self.gamma = self.M.sum(2).squeeze()  # State occupancy probabilities
        T_new = self.M.sum(0) / self.gamma.sum(0).unsqueeze(1)  # Normalize transition counts
        T0_new = self.gamma[0, :]  # Initial state distribution from first time step
        
        # Add final time step to gamma
        prod = (self.forward[self.N-1] * self.backward[self.N-1]).unsqueeze(0)
        s = prod / prod.sum()
        self.gamma = torch.cat([self.gamma, s], 0)
        self.prob_state_1.append(self.gamma[:, 0].detach().numpy())
        
        # Add epsilon and renormalize to prevent zeros
        epsilon = 1e-10
        T_new = T_new + epsilon
        T0_new = T0_new + epsilon
        T_new = T_new / T_new.sum(dim=1, keepdim=True)
        T0_new = T0_new / T0_new.sum()
        
        return T0_new, T_new

    def re_estimate_emission(self, x):
        # Re-estimate emission matrix using expected state occupancies
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.int64)
        
        E_new = torch.zeros_like(self.E)
        
        # Accumulate expected counts: for each observation, add probability of being in each state
        for t in range(self.N):
            obs_idx = x[t]
            if obs_idx >= self.E.shape[1]:
                obs_idx = self.E.shape[1] - 1
            
            # Get state occupancy probabilities (gamma)
            if t < len(self.gamma):
                state_probs = self.gamma[t]
            else:
                # Fallback: compute from forward and backward if gamma not available
                state_probs = self.forward[t] * self.backward[t]
                state_probs = state_probs / state_probs.sum()
            
            # Add state probabilities to emission counts for this observation
            for i in range(self.S):
                E_new[i, obs_idx] += state_probs[i]
        
        # Normalize rows so each state's emission probabilities sum to 1
        epsilon = 1e-10
        E_new = E_new + epsilon
        row_sums = E_new.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        E_new = E_new / row_sums
        
        return E_new

    def expectation_maximization_step(self, obs_seq):
        # Single EM step: E-step (forward-backward) then M-step (parameter re-estimation)
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.tensor(obs_seq, dtype=torch.int64)
        
        # Build emission probability sequence for forward-backward algorithm
        obs_prob_seq = []
        for t in range(len(obs_seq)):
            obs_idx = obs_seq[t]
            if obs_idx >= self.E.shape[1]:
                obs_idx = self.E.shape[1] - 1
            emission_probs = self.E[:, obs_idx]  # Probability of this observation given each state
            obs_prob_seq.append(emission_probs)
        
        obs_prob_seq = torch.stack(obs_prob_seq)
        self.forward_backward(obs_prob_seq)  # E-step: compute state probabilities
        
        # M-step: re-estimate parameters using expected counts
        new_T0, new_transition = self.re_estimate_transition(obs_seq)
        new_emission = self.re_estimate_emission(obs_seq)
        
        converged = self.check_convergence(new_T0, new_transition, new_emission)
        self.T0 = new_T0
        self.E = new_emission
        self.T = new_transition
        
        return converged

    def check_convergence(self, new_T0, new_transition, new_emission):
        with torch.no_grad():
            delta_T0 = torch.max(torch.abs(self.T0 - new_T0)).item() < self.epsilon
            delta_T = torch.max(torch.abs(self.T - new_transition)).item() < self.epsilon
            delta_E = torch.max(torch.abs(self.E - new_emission)).item() < self.epsilon
        return delta_T0 and delta_T and delta_E

    def Baum_Welch_EM(self, obs_seq):
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.tensor(obs_seq, dtype=torch.int64)
        
        self.N = len(obs_seq)
        self.forward = torch.zeros([self.N, self.S], dtype=torch.float64)
        self.backward = torch.zeros([self.N, self.S], dtype=torch.float64)
        self.posterior = torch.zeros([self.N, self.S], dtype=torch.float64)
        converged = False
        
        print(f"Starting Baum-Welch EM with {self.maxStep} max steps")
        start_time = time.time()
        
        for i in range(self.maxStep):
            iter_start = time.time()
            try:
                converged = self.expectation_maximization_step(obs_seq)
                iter_time = time.time() - iter_start
                print(f"  Step {i+1}/{self.maxStep} completed in {iter_time:.2f}s")
                
                if converged:
                    print(f'Converged at step {i+1}')
                    break
            except Exception as step_error:
                print(f"Error in EM step {i+1}: {str(step_error)}")
                continue
        
        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")
        
        return self.T0, self.T, self.E, converged

    def save_model(self, filepath):
        torch.save({
            'T': self.T,
            'E': self.E,
            'T0': self.T0,
            'S': self.S,
            'O': self.O,
            'prob_state_1': self.prob_state_1
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, device=None):
        checkpoint = torch.load(filepath, map_location='cpu')
        T = checkpoint['T'].numpy()
        E = checkpoint['E'].numpy()
        T0 = checkpoint['T0'].numpy()
        model = cls(T, E, T0, device='cpu')
        model.prob_state_1 = checkpoint.get('prob_state_1', [])
        print(f"Model loaded from {filepath}")
        return model

    def interpret_states(self, states_seq, observations, actual_labels=None):
        unique_states = torch.unique(states_seq).numpy()
        state_interpretations = {}
        
        for state in unique_states:
            state_mask = (states_seq.numpy() == state)
            state_obs = observations[state_mask]
            mean_value = np.mean(state_obs)
            std_value = np.std(state_obs)
            
            if actual_labels is not None:
                state_labels = actual_labels[state_mask]
                bull_ratio = np.mean(state_labels)
                
                if bull_ratio > 0.7:
                    state_type = "Bull Market"
                elif bull_ratio < 0.3:
                    state_type = "Bear Market"
                else:
                    state_type = "Sideways/Mixed Market"
                
                state_interpretations[state] = {
                    'type': state_type,
                    'bull_ratio': bull_ratio,
                    'mean': mean_value,
                    'std': std_value
                }
            else:
                state_type = "Likely Bull Market" if mean_value > 0 else ("Likely Bear Market" if mean_value < 0 else "Likely Sideways Market")
                state_interpretations[state] = {
                    'type': state_type,
                    'mean': mean_value,
                    'std': std_value
                }
        
        return state_interpretations

    def evaluate(self, observations, mode='classification', actual_values=None, actual_labels=None, 
                 observation_map=None, class_threshold=0.5, direct_states=False):
        # Evaluate model performance on given observations
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.int64)
        
        states_seq, state_probs = self.viterbi_inference(observations)  # Get most likely state sequence
        states_seq_np = states_seq.numpy()
        metrics = {'states_seq': states_seq_np}
        
        if mode == 'classification':
            if actual_labels is None:
                raise ValueError("actual_labels must be provided for classification mode")
            
            if direct_states:
                # Map states to labels using correlation: positive correlation = bull, negative = bear
                unique_states = np.unique(states_seq_np)
                state_correlations = {}
                for state in unique_states:
                    state_presence = (states_seq_np == state).astype(int)  # Binary indicator for this state
                    corr = np.corrcoef(state_presence, actual_labels)[0, 1]
                    state_correlations[state] = corr
                
                state_to_label = {state: 1 if state_correlations[state] >= 0 else 0 
                                 for state in unique_states}
                pred_labels = np.array([state_to_label[state] for state in states_seq_np])
                
                print("\nState to Market Regime Correlations:")
                for state, corr in state_correlations.items():
                    regime = "Bull" if state_to_label[state] == 1 else "Bear"
                    print(f"  State {state}: {corr:.4f} correlation, classified as {regime} Market")
            else:
                # Majority voting: assign label based on most common label within each state
                unique_states = np.unique(states_seq_np)
                state_to_label = {}
                for state in unique_states:
                    mask = (states_seq_np == state)
                    avg_label = np.mean(actual_labels[mask])  # Average label (proportion of bull) for this state
                    state_to_label[state] = 1 if avg_label >= class_threshold else 0
                pred_labels = np.array([state_to_label[state] for state in states_seq_np])
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            
            accuracy = accuracy_score(actual_labels, pred_labels)
            precision = precision_score(actual_labels, pred_labels, zero_division=0)
            recall = recall_score(actual_labels, pred_labels, zero_division=0)
            f1 = f1_score(actual_labels, pred_labels, zero_division=0)
            conf_matrix = confusion_matrix(actual_labels, pred_labels)
            
            metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix,
                'predicted_labels': pred_labels,
            })
            
            metrics['state_interpretations'] = self.interpret_states(
                states_seq, actual_values, actual_labels)
        
        elif mode == 'forecasting':
            if observation_map is None:
                raise ValueError("observation_map must be provided for forecasting mode")
            
            forecasts = []
            for t in range(len(observations) - 1):
                current_probs = torch.zeros(self.S, dtype=torch.float64)
                current_probs[states_seq[t]] = 1.0
                next_obs_probs = torch.matmul(
                    torch.matmul(current_probs.unsqueeze(0), self.T).squeeze(0).unsqueeze(0),
                    self.E.T).squeeze(0)
                prediction = np.sum(next_obs_probs.numpy() * observation_map)
                forecasts.append(prediction)
            
            forecasts = np.array(forecasts)
            if actual_values is not None:
                actual_next_values = actual_values[1:]
                mse = np.mean((forecasts - actual_next_values) ** 2)
                mae = np.mean(np.abs(forecasts - actual_next_values))
                correlation = np.corrcoef(forecasts, actual_next_values)[0, 1]
                metrics.update({
                    'mse': mse,
                    'mae': mae,
                    'correlation': correlation,
                    'forecasts': forecasts,
                    'actual_next_values': actual_next_values
                })
            else:
                metrics['forecasts'] = forecasts
        
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")
        
        return metrics
