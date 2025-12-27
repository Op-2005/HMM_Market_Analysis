"""
Backward-compatible HiddenMarkovModel class.

This class maintains the original API for backward compatibility.
For new code, consider using DiscreteHMM and HMMTrainer from the refactored API
which provide better separation of concerns and cleaner architecture.
"""
import torch
import numpy as np
import time


class HiddenMarkovModel(object):
    def __init__(self, T, E, T0, device='cpu', epsilon=0.001, maxStep=10):
        self.device = 'cpu'
        self.maxStep = maxStep
        self.epsilon = epsilon
        self.S = T.shape[0]
        self.O = E.shape[0]
        self.prob_state_1 = []
        
        # Convert to NumPy arrays first if they're not already
        if isinstance(T, torch.Tensor):
            T = T.detach().cpu().numpy()
        if isinstance(E, torch.Tensor):
            E = E.detach().cpu().numpy()
        if isinstance(T0, torch.Tensor):
            T0 = T0.detach().cpu().numpy()
            
        # Create tensors with better handling
        try:
            self.E = torch.tensor(E, dtype=torch.float64)
            self.T = torch.tensor(T, dtype=torch.float64)
            self.T0 = torch.tensor(T0, dtype=torch.float64)
            
            # Ensure parameters are properly normalized (probability distributions)
            epsilon = 1e-10
            self.T = torch.clamp(self.T, min=epsilon)
            self.E = torch.clamp(self.E, min=epsilon)
            self.T0 = torch.clamp(self.T0, min=epsilon)
            
            # Normalize to ensure rows sum to 1
            self.T = self.T / self.T.sum(dim=1, keepdim=True)
            self.E = self.E / self.E.sum(dim=1, keepdim=True)
            self.T0 = self.T0 / self.T0.sum()
        except Exception as e:
            print(f"Error initializing HMM tensors: {str(e)}")
            print(f"T shape: {T.shape}, E shape: {E.shape}, T0 shape: {T0.shape}")
            raise

    def initialize_viterbi_variables(self, shape):
        try:
            pathStates = torch.zeros(shape, dtype=torch.float64)
            pathScores = torch.zeros_like(pathStates)
            states_seq = torch.zeros([shape[0]], dtype=torch.int64)
            return pathStates, pathScores, states_seq
        except Exception as e:
            print(f"Error initializing Viterbi variables: {str(e)}")
            print(f"Shape: {shape}")
            raise

    def belief_propagation(self, scores):
        try:
            return scores.view(-1, 1) + torch.log(self.T)
        except Exception as e:
            print(f"Error in belief propagation: {str(e)}")
            print(f"Scores shape: {scores.shape}, T shape: {self.T.shape}")
            raise

    def viterbi_inference(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.int64)

        self.N = len(x)
        shape = [self.N, self.S]
        
        try:
            pathStates, pathScores, states_seq = self.initialize_viterbi_variables(shape)
            
            # KEY FIX: Access emission prob correctly - E has shape (num_states, num_observations)
            # So we need to get probability for each state given observation x[t]
            obs_prob_full = []
            for t in range(self.N):
                # For each observation x[t], get probabilities for all states
                # This handles the emission matrix E with shape (num_states, num_observations)
                obs_idx = x[t]
                if obs_idx >= self.E.shape[1]:
                    print(f"Warning: Observation index {obs_idx} out of bounds for emission matrix with {self.E.shape[1]} observations")
                    # Clip to valid range
                    obs_idx = self.E.shape[1] - 1
                obs_prob = self.E[:, obs_idx]  # Get probability for all states given observation x[t]
                obs_prob_full.append(obs_prob)
            
            obs_prob_full = torch.stack(obs_prob_full)
            obs_prob_full = torch.log(obs_prob_full)
            
            # First step initialization with observation probability
            pathScores[0] = torch.log(self.T0) + obs_prob_full[0]

            for step, obs_prob in enumerate(obs_prob_full[1:]):
                belief = self.belief_propagation(pathScores[step, :])
                pathStates[step + 1] = torch.argmax(belief, 0)
                pathScores[step + 1] = torch.max(belief, 0)[0] + obs_prob

            states_seq[self.N - 1] = torch.argmax(pathScores[self.N-1, :], 0)

            for step in range(self.N - 1, 0, -1):
                state = states_seq[step]
                state_prob = pathStates[step][state]
                states_seq[step - 1] = state_prob

            return states_seq, torch.exp(pathScores)
        except Exception as e:
            print(f"Error in Viterbi inference: {str(e)}")
            print(f"x shape: {x.shape}, x unique values: {torch.unique(x)}")
            print(f"x min: {x.min()}, x max: {x.max()}, E shape: {self.E.shape}")
            raise

    def initialize_forw_back_variables(self, shape):
        try:
            self.forward = torch.zeros(shape, dtype=torch.float64)
            self.backward = torch.zeros_like(self.forward)
            self.posterior = torch.zeros_like(self.forward)
        except Exception as e:
            print(f"Error initializing forward-backward variables: {str(e)}")
            print(f"Shape: {shape}")
            raise

    def _forward(self, obs_prob_seq):
        try:
            self.scale = torch.zeros([self.N], dtype=torch.float64)
            init_prob = self.T0 * obs_prob_seq[0]
            
            # Handle numerical issues
            sum_init = init_prob.sum()
            if sum_init > 0:
                self.scale[0] = 1.0 / sum_init
            else:
                print("Warning: Zero probability in forward algorithm initialization")
                self.scale[0] = 1.0
                
            self.forward[0] = self.scale[0] * init_prob

            for step, obs_prob in enumerate(obs_prob_seq[1:]):
                prev_prob = self.forward[step].unsqueeze(0)
                prior_prob = torch.matmul(prev_prob, self.T)
                forward_score = prior_prob * obs_prob
                forward_prob = torch.squeeze(forward_score)
                
                # Handle numerical issues
                sum_forward = forward_prob.sum()
                if sum_forward > 0:
                    self.scale[step + 1] = 1 / sum_forward
                else:
                    print(f"Warning: Zero probability in forward algorithm at step {step+1}")
                    self.scale[step + 1] = 1.0
                    
                self.forward[step + 1] = self.scale[step + 1] * forward_prob
        except Exception as e:
            print(f"Error in forward algorithm: {str(e)}")
            print(f"obs_prob_seq shape: {obs_prob_seq.shape}")
            raise

    def _backward(self, obs_prob_seq_rev):
        try:
            self.backward[0] = self.scale[self.N - 1] * \
                torch.ones([self.S], dtype=torch.float64)

            for step, obs_prob in enumerate(obs_prob_seq_rev[:-1]):
                next_prob = self.backward[step, :].unsqueeze(1)
                obs_prob_d = torch.diag(obs_prob)
                prior_prob = torch.matmul(self.T, obs_prob_d)
                backward_prob = torch.matmul(prior_prob, next_prob).squeeze()
                self.backward[step + 1] = self.scale[self.N -
                                                    2 - step] * backward_prob

            self.backward = torch.flip(self.backward, [0, 1])
        except Exception as e:
            print(f"Error in backward algorithm: {str(e)}")
            print(f"obs_prob_seq_rev shape: {obs_prob_seq_rev.shape}")
            raise

    def forward_backward(self, obs_prob_seq):
        """
        Forward-backward algorithm with improved numerical stability using log-space.
        
        This implementation uses log-space computations to prevent underflow on long sequences,
        while maintaining efficiency and accuracy.
        """
        try:
            # Check input shape - obs_prob_seq should be (seq_len, num_states)
            if len(obs_prob_seq.shape) != 2 or obs_prob_seq.shape[1] != self.S:
                raise ValueError(
                    f"Expected obs_prob_seq shape (seq_len, {self.S}), got {obs_prob_seq.shape}")
            
            self.N = len(obs_prob_seq)
            epsilon = 1e-10  # Small epsilon to prevent log(0)
            
            # Clamp probabilities to avoid zeros
            obs_prob_seq = torch.clamp(obs_prob_seq, min=epsilon, max=1.0)
            
            # Convert to log-space
            log_obs = torch.log(obs_prob_seq)
            log_T = torch.log(torch.clamp(self.T, min=epsilon))
            log_T0 = torch.log(torch.clamp(self.T0, min=epsilon))
            
            # Initialize forward and backward in log-space
            log_forward = torch.zeros([self.N, self.S], dtype=torch.float64)
            log_backward = torch.zeros([self.N, self.S], dtype=torch.float64)
            
            # Forward pass initialization
            log_forward[0, :] = log_T0 + log_obs[0, :]
            # Normalize using logsumexp to prevent underflow
            log_forward[0, :] = log_forward[0, :] - torch.logsumexp(log_forward[0, :], dim=0)
            
            # Forward pass iteration using log-space
            for t in range(1, self.N):
                # log P(x_t | z_t) + logsumexp_z_{t-1} [log P(z_t | z_{t-1}) + log alpha_{t-1}(z_{t-1})]
                # We compute: log(forward[t-1]) + log(T) for all transitions, then logsumexp
                log_transitions = log_forward[t-1, :].unsqueeze(1) + log_T  # (S, S)
                log_forward[t, :] = torch.logsumexp(log_transitions, dim=0) + log_obs[t, :]
                # Normalize
                log_forward[t, :] = log_forward[t, :] - torch.logsumexp(log_forward[t, :], dim=0)
            
            # Backward pass initialization
            log_backward[self.N-1, :] = torch.zeros(self.S, dtype=torch.float64)
            
            # Backward pass iteration using log-space
            for t in range(self.N-2, -1, -1):
                # log beta_t(z_t) = logsumexp_z_{t+1} [log P(z_{t+1} | z_t) + log P(x_{t+1} | z_{t+1}) + log beta_{t+1}(z_{t+1})]
                log_transitions = log_T + (log_obs[t+1, :] + log_backward[t+1, :]).unsqueeze(0)  # (S, S)
                log_backward[t, :] = torch.logsumexp(log_transitions, dim=1)
                # Normalize for numerical stability
                log_backward[t, :] = log_backward[t, :] - torch.logsumexp(log_backward[t, :], dim=0)
            
            # Convert back to probability space (these are properly normalized)
            self.forward = torch.exp(log_forward)
            self.backward = torch.exp(log_backward)
            
            # Ensure proper normalization (double-check)
            self.forward = self.forward / (self.forward.sum(dim=1, keepdim=True) + epsilon)
            self.backward = self.backward / (self.backward.sum(dim=1, keepdim=True) + epsilon)
            
        except Exception as e:
            print(f"Error in forward-backward algorithm: {str(e)}")
            raise

    def re_estimate_transition(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.int64)

        self.M = torch.zeros([self.N - 1, self.S, self.S], dtype=torch.float64)

        for t in range(self.N - 1):
            # Get observation probabilities for each state
            obs_idx_t1 = x[t + 1]
            # Clip index if out of bounds
            if obs_idx_t1 >= self.E.shape[1]:
                obs_idx_t1 = self.E.shape[1] - 1
                print(f"Warning: Observation index {x[t+1]} clipped to {obs_idx_t1}")
            
            # Get emission probabilities for all states given observation x[t+1]
            emission_probs_t1 = self.E[:, obs_idx_t1]  # Shape: (num_states,)
            
            tmp_0 = torch.matmul(self.forward[t].unsqueeze(0), self.T)
            tmp_1 = tmp_0 * emission_probs_t1.unsqueeze(0)
            denom = torch.matmul(
                tmp_1, self.backward[t + 1].unsqueeze(1)).squeeze()

            trans_re_estimate = torch.zeros(
                [self.S, self.S], dtype=torch.float64)

            for i in range(self.S):
                numer = self.forward[t, i] * self.T[i, :] * emission_probs_t1 * self.backward[t+1]
                trans_re_estimate[i] = numer / denom

            self.M[t] = trans_re_estimate

        self.gamma = self.M.sum(2).squeeze()
        T_new = self.M.sum(0) / self.gamma.sum(0).unsqueeze(1)
        T0_new = self.gamma[0, :]
        prod = (self.forward[self.N-1] * self.backward[self.N-1]).unsqueeze(0)
        s = prod / prod.sum()
        # Restore these lines to maintain compatibility with existing code
        self.gamma = torch.cat([self.gamma, s], 0)
        self.prob_state_1.append(self.gamma[:, 0].detach().numpy())

        # Add epsilon flooring to prevent zero probabilities (improves numerical stability)
        epsilon = 1e-10
        T_new = T_new + epsilon
        T0_new = T0_new + epsilon
        
        # Renormalize to ensure proper probability distributions
        T_new = T_new / T_new.sum(dim=1, keepdim=True)
        T0_new = T0_new / T0_new.sum()

        return T0_new, T_new

    def re_estimate_emission(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.int64)

        # Create a new emission matrix with shape (num_states, num_observations)
        E_new = torch.zeros_like(self.E)
        
        # Count occurrences for each observation
        for t in range(self.N):
            obs_idx = x[t]
            if obs_idx >= self.E.shape[1]:
                print(f"Warning: Observation {obs_idx} out of bounds, clipping to {self.E.shape[1]-1}")
                obs_idx = self.E.shape[1] - 1
            
            # Add probability of being in each state at time t to the corresponding emission entry
            if t < len(self.gamma):
                # For t < N-1, use gamma from the re_estimate_transition function
                state_probs = self.gamma[t]
            else:
                # For t = N-1, calculate state probabilities
                state_probs = self.forward[t] * self.backward[t]
                state_probs = state_probs / state_probs.sum()
            
            # Update emission probability for each state i and observation obs_idx
            for i in range(self.S):
                E_new[i, obs_idx] += state_probs[i]
        
        # Add epsilon flooring to prevent zero probabilities (improves numerical stability)
        epsilon = 1e-10
        E_new = E_new + epsilon
        
        # Normalize each row (each state) to sum to 1
        row_sums = E_new.sum(dim=1, keepdim=True)
        # Avoid division by zero
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        E_new = E_new / row_sums
        
        return E_new

    def validate(self, raise_on_error=True):
        """
        Validate HMM parameters to ensure they satisfy constraints.
        
        Checks:
        - Matrix shapes are correct
        - Transition matrix rows sum to 1
        - Emission matrix rows sum to 1
        - Initial state distribution sums to 1
        - All probabilities are non-negative
        - No NaN or Inf values
        
        Parameters:
        -----------
        raise_on_error : bool
            If True, raise ValueError on validation failure. If False, return False.
        
        Returns:
        --------
        bool
            True if validation passes, False otherwise
        """
        errors = []
        epsilon = 1e-6  # Tolerance for floating point comparison
        
        # Check shapes
        if self.T.shape[0] != self.S or self.T.shape[1] != self.S:
            errors.append(f"Transition matrix T has incorrect shape: {self.T.shape}, expected ({self.S}, {self.S})")
        
        if self.E.shape[0] != self.S:
            errors.append(f"Emission matrix E has incorrect first dimension: {self.E.shape[0]}, expected {self.S}")
        
        if self.T0.shape[0] != self.S:
            errors.append(f"Initial state distribution T0 has incorrect shape: {self.T0.shape}, expected ({self.S},)")
        
        # Check for NaN or Inf
        if torch.isnan(self.T).any():
            errors.append("Transition matrix T contains NaN values")
        if torch.isinf(self.T).any():
            errors.append("Transition matrix T contains Inf values")
        
        if torch.isnan(self.E).any():
            errors.append("Emission matrix E contains NaN values")
        if torch.isinf(self.E).any():
            errors.append("Emission matrix E contains Inf values")
        
        if torch.isnan(self.T0).any():
            errors.append("Initial state distribution T0 contains NaN values")
        if torch.isinf(self.T0).any():
            errors.append("Initial state distribution T0 contains Inf values")
        
        # Check non-negativity
        if (self.T < 0).any():
            errors.append("Transition matrix T contains negative values")
        if (self.E < 0).any():
            errors.append("Emission matrix E contains negative values")
        if (self.T0 < 0).any():
            errors.append("Initial state distribution T0 contains negative values")
        
        # Check row sums for transition matrix
        T_row_sums = self.T.sum(dim=1)
        if not torch.allclose(T_row_sums, torch.ones(self.S, dtype=torch.float64), atol=epsilon):
            errors.append(f"Transition matrix T rows do not sum to 1. Sums: {T_row_sums}")
        
        # Check row sums for emission matrix
        E_row_sums = self.E.sum(dim=1)
        if not torch.allclose(E_row_sums, torch.ones(self.S, dtype=torch.float64), atol=epsilon):
            errors.append(f"Emission matrix E rows do not sum to 1. Sums: {E_row_sums}")
        
        # Check initial state distribution sum
        T0_sum = self.T0.sum()
        if not torch.allclose(T0_sum, torch.tensor(1.0, dtype=torch.float64), atol=epsilon):
            errors.append(f"Initial state distribution T0 does not sum to 1. Sum: {T0_sum}")
        
        if errors:
            error_msg = "HMM validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            if raise_on_error:
                raise ValueError(error_msg)
            else:
                print(f"WARNING: {error_msg}")
                return False
        
        return True
    
    def check_convergence(self, new_T0, new_transition, new_emission):
        with torch.no_grad():
            delta_T0 = torch.max(torch.abs(self.T0 - new_T0)
                                 ).item() < self.epsilon
            delta_T = torch.max(
                torch.abs(self.T - new_transition)).item() < self.epsilon
            delta_E = torch.max(
                torch.abs(self.E - new_emission)).item() < self.epsilon

        return delta_T0 and delta_T and delta_E

    def expectation_maximization_step(self, obs_seq):
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.tensor(obs_seq, dtype=torch.int64)

        # Get emission probabilities for all states and all observations in sequence
        # This builds a sequence of emission probabilities with shape (seq_len, num_states)
        obs_prob_seq = []
        for t in range(len(obs_seq)):
            obs_idx = obs_seq[t]
            if obs_idx >= self.E.shape[1]:
                print(f"Warning: Observation {obs_idx} out of bounds, clipping to {self.E.shape[1]-1}")
                obs_idx = self.E.shape[1] - 1
            
            # Get emission probabilities for all states given observation obs_idx
            emission_probs = self.E[:, obs_idx]  # Shape: (num_states,)
            obs_prob_seq.append(emission_probs)
        
        # Stack into tensor of shape (seq_len, num_states)
        obs_prob_seq = torch.stack(obs_prob_seq)
        
        # Perform forward-backward algorithm
        self.forward_backward(obs_prob_seq)
        
        # Re-estimate parameters
        new_T0, new_transition = self.re_estimate_transition(obs_seq)
        new_emission = self.re_estimate_emission(obs_seq)
        
        # Check convergence
        converged = self.check_convergence(
            new_T0, new_transition, new_emission)

        self.T0 = new_T0
        self.E = new_emission
        self.T = new_transition

        return converged

    def Baum_Welch_EM(self, obs_seq):
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.tensor(obs_seq, dtype=torch.int64)

        try:
            self.N = len(obs_seq)
            shape = [self.N, self.S]
            self.initialize_forw_back_variables(shape)
            converged = False

            start_time = time.time()
            print(f"Starting Baum-Welch EM with {self.maxStep} max steps")

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
                    # Continue with next iteration instead of failing completely
                    continue

            total_time = time.time() - start_time
            print(f"Total training time: {total_time:.2f} seconds")

            return self.T0, self.T, self.E, converged
        except Exception as e:
            print(f"Error in Baum-Welch algorithm: {str(e)}")
            print(f"obs_seq shape: {obs_seq.shape}, unique values: {torch.unique(obs_seq)}")
            # Return current parameters even if we failed
            return self.T0, self.T, self.E, False

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
                if mean_value > 0:
                    state_type = "Likely Bull Market"
                elif mean_value < 0:
                    state_type = "Likely Bear Market"
                else:
                    state_type = "Likely Sideways Market"

                state_interpretations[state] = {
                    'type': state_type,
                    'mean': mean_value,
                    'std': std_value
                }

        return state_interpretations

    def predict_one_step_ahead(self, current_state_probs, observation_map=None):
        next_state_probs = torch.matmul(
            current_state_probs.unsqueeze(0), self.T).squeeze(0)
        next_obs_probs = torch.matmul(
            next_state_probs.unsqueeze(0), self.E.T).squeeze(0)
        next_obs_probs_np = next_obs_probs.numpy()

        if observation_map is not None:
            prediction = np.sum(next_obs_probs_np * observation_map)
            return next_obs_probs_np, prediction
        else:
            return next_obs_probs_np, None

    def evaluate(self, observations, mode='classification', actual_values=None, actual_labels=None, observation_map=None, class_threshold=0.5, direct_states=False):
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.int64)

        states_seq, state_probs = self.viterbi_inference(observations)
        states_seq_np = states_seq.numpy()
        metrics = {'states_seq': states_seq_np}

        if mode == 'classification':
            if actual_labels is None:
                raise ValueError(
                    "actual_labels must be provided for classification mode")

            if direct_states:
                # Use states directly to predict labels
                # Calculate correlation between each state and bull/bear markets
                unique_states = np.unique(states_seq_np)
                state_correlations = {}

                for state in unique_states:
                    # Create a binary array for this state (1 where this state is active)
                    state_presence = (states_seq_np == state).astype(int)
                    # Calculate correlation with actual labels
                    corr = np.corrcoef(state_presence, actual_labels)[0, 1]
                    state_correlations[state] = corr

                # Assign states to bear/bull based on correlation
                state_to_label = {}
                for state in unique_states:
                    # If positive correlation with bull market (actual_labels==1), then predict bull
                    # If negative correlation, then predict bear
                    state_to_label[state] = 1 if state_correlations[state] >= 0 else 0

                pred_labels = np.array([state_to_label[state]
                                       for state in states_seq_np])

                # Print correlation information
                print("\nState to Market Regime Correlations:")
                for state, corr in state_correlations.items():
                    regime = "Bull" if state_to_label[state] == 1 else "Bear"
                    print(
                        f"  State {state}: {corr:.4f} correlation, classified as {regime} Market")
            else:
                # Original method - use majority voting within each state
                unique_states = np.unique(states_seq_np)
                state_to_label = {}

                for state in unique_states:
                    mask = (states_seq_np == state)
                    avg_label = np.mean(actual_labels[mask])
                    state_to_label[state] = 1 if avg_label >= class_threshold else 0

                pred_labels = np.array([state_to_label[state]
                                       for state in states_seq_np])

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            accuracy = accuracy_score(actual_labels, pred_labels)
            precision = precision_score(
                actual_labels, pred_labels, zero_division=0)
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
                raise ValueError(
                    "observation_map must be provided for forecasting mode")

            forecasts = []
            actual_next_values = actual_values[1:
                                               ] if actual_values is not None else None

            for t in range(len(observations) - 1):
                current_probs = torch.zeros(self.S, dtype=torch.float64)
                current_probs[states_seq[t]] = 1.0

                _, prediction = self.predict_one_step_ahead(
                    current_probs, observation_map)
                forecasts.append(prediction)

            forecasts = np.array(forecasts)

            if actual_next_values is not None:
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
