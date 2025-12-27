# source .venv/bin/activate && python train.py --mode classification --states 6 --observations 40 --discr_strategy kmeans --direct_states --feature "sp500 high-low" --target "sp500 close" --steps 40 --final_test
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ..data.data_processor import FinancialDataLoader, discretize_data, map_bins_to_values, Discretizer
from ..models.hmm_model import HiddenMarkovModel

np.random.seed(42)
torch.manual_seed(42)

device = 'cpu'
print(f"Using device: {device}")


def load_financial_data(file_path, normalize=True):
    print(f"Loading data from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    print(f"Loaded data with shape {data.shape}")
    print(f"Columns: {', '.join(data.columns)}")

    return data


def prepare_data(data, features, target_column, test_size=0.2, normalize=True):
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in data. Available columns: {', '.join(data.columns)}")

    for feat in features:
        if feat not in data.columns:
            raise ValueError(
                f"Feature column '{feat}' not found in data. Available columns: {', '.join(data.columns)}")

    subset_cols = [target_column] + features
    data_clean = data.dropna(subset=subset_cols)

    print(f"Dropped {len(data) - len(data_clean)} rows with NaN values")

    X = data_clean[features].values.astype(np.float32)
    y = data_clean[target_column].values.astype(np.float32)

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(
        f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    scaler_params = {}
    if normalize:
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        scaler_params = {'mean': mean, 'std': std}
        print(f"Normalized features with mean={mean} and std={std}")

    return X_train, X_test, y_train, y_test, scaler_params


def initialize_hmm_params(num_states, num_observations):
    T = np.ones((num_states, num_states)) / num_states
    T = T + np.random.uniform(0, 0.1, T.shape)
    T = T / T.sum(axis=1, keepdims=True)

    E = np.ones((num_states, num_observations)) / num_observations
    E = E + np.random.uniform(0, 0.1, E.shape)
    E = E / E.sum(axis=1, keepdims=True)

    T0 = np.ones(num_states) / num_states

    return T, E, T0


def train_hmm(X_train_discrete, num_states, num_observations, max_steps=20):
    print("\n" + "="*50)
    print(
        f"Training HMM with {num_states} states and {num_observations} observations")
    print("="*50)

    T, E, T0 = initialize_hmm_params(num_states, num_observations)

    hmm = HiddenMarkovModel(T, E, T0, device='cpu',
                            epsilon=0.001, maxStep=max_steps)

    if not isinstance(X_train_discrete, torch.Tensor):
        X_train_discrete = torch.tensor(X_train_discrete, dtype=torch.int64)

    print(f"Running Baum-Welch EM algorithm")
    start_time = time.time()

    T0, T, E, converged = hmm.Baum_Welch_EM(X_train_discrete)

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    if converged:
        print("HMM training converged!")
    else:
        print("HMM training did not converge within max_steps")

    return hmm


def run_single_epoch_test(X_train_discrete, num_states, num_observations):
    print("\n" + "="*50)
    print("RUNNING SINGLE EPOCH TEST")
    print("="*50)

    T, E, T0 = initialize_hmm_params(num_states, num_observations)

    hmm = HiddenMarkovModel(T, E, T0, device='cpu', epsilon=0.001, maxStep=1)

    if not isinstance(X_train_discrete, torch.Tensor):
        X_train_discrete = torch.tensor(X_train_discrete, dtype=torch.int64)

    hmm.N = len(X_train_discrete)
    shape = [hmm.N, hmm.S]
    hmm.initialize_forw_back_variables(shape)

    print("Running single epoch (EM step)...")
    start_time = time.time()

    obs_prob_seq = hmm.E[X_train_discrete]

    hmm.forward_backward(obs_prob_seq)

    new_T0, new_T = hmm.re_estimate_transition(X_train_discrete)
    new_E = hmm.re_estimate_emission(X_train_discrete)

    converged = hmm.check_convergence(new_T0, new_T, new_E)

    hmm.T0 = new_T0
    hmm.E = new_E
    hmm.T = new_T

    elapsed_time = time.time() - start_time
    print(f"Single epoch completed in {elapsed_time:.2f} seconds")

    print("\nInitial State Probabilities (T0):")
    print(hmm.T0.numpy())

    print("\nTransition Matrix (T) - First few rows:")
    print(hmm.T[:3].numpy())

    print("\nEmission Matrix (E) - First few rows:")
    print(hmm.E[:3].numpy())

    return hmm, {'T0': hmm.T0, 'T': hmm.T, 'E': hmm.E}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train HMM models for financial time series')
    parser.add_argument('--feature', type=str, default='sp500 close',
                        help='Feature column to use from the dataset')
    parser.add_argument('--target', type=str, default='sp500 close',
                        help='Target column to predict')
    parser.add_argument('--mode', type=str, choices=['classification', 'forecasting'], default='classification',
                        help='HMM task mode: classification or forecasting')
    parser.add_argument('--states', type=int, default=3,
                        help='Number of hidden states for the HMM')
    parser.add_argument('--observations', type=int, default=15,
                        help='Number of observation bins for discretization')
    parser.add_argument('--steps', type=int, default=20,
                        help='Maximum training steps for Baum-Welch algorithm')
    parser.add_argument('--single_epoch', action='store_true',
                        help='Run a single epoch test')
    parser.add_argument('--data_file', type=str, default='financial_data.csv',
                        help='Path to the financial data CSV file')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size (as a fraction of the whole dataset)')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation set size (as a fraction of the training dataset)')
    parser.add_argument('--discr_strategy', type=str, default='equal_freq',
                        choices=['equal_freq', 'equal_width', 'kmeans'],
                        help='Discretization strategy')
    parser.add_argument('--class_threshold', type=float, default=0.5,
                        help='Classification threshold for bull/bear market')
    parser.add_argument('--use_feature_for_hmm', action='store_true',
                        help='Use feature values instead of returns for HMM training')
    parser.add_argument('--direct_states', action='store_true',
                        help='Use direct state correlation for classification instead of majority voting')
    parser.add_argument('--use_validation', action='store_true',
                        help='Use validation set for evaluation instead of test set')
    parser.add_argument('--final_test', action='store_true',
                        help='Train on train+validation data and evaluate on test set')

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Using device: {device}")
    print(
        f"Running in {args.mode} mode with {args.states} states and {args.observations} observation bins")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, args.data_file)
    normalize = True

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = load_financial_data(file_path)

        data_loader = FinancialDataLoader(
            file_path=file_path,
            target_column=args.target,
            features=[args.feature],
            normalize=normalize
        )

        log_returns_col = data_loader.add_log_returns(args.target)

        label_col = data_loader.add_regime_labels(
            log_returns_col, threshold=0.0, window=5)

        # Modified to implement train/validation/test split
        if args.use_validation or args.final_test:
            print(
                f"Using validation split with val_size={args.val_size} and test_size={args.test_size}")
            # First split train+val/test
            total_samples = len(data_loader.data)
            test_size_actual = int(total_samples * args.test_size)
            train_val_size = total_samples - test_size_actual

            indices = np.arange(total_samples)

            train_val_indices = indices[:train_val_size]
            test_indices = indices[train_val_size:]

            # Create test dataset
            test_data = data_loader.data.iloc[test_indices].copy()
            test_loader = FinancialDataLoader(
                file_path=None, target_column=args.target, features=[args.feature],
                normalize=normalize, data=test_data, device=device
            )

            if args.final_test:
                # Use all non-test data for training
                train_data = data_loader.data.iloc[train_val_indices].copy()
                train_loader = FinancialDataLoader(
                    file_path=None, target_column=args.target, features=[args.feature],
                    normalize=normalize, data=train_data, device=device
                )

                # Use test data for evaluation
                X_train = train_loader.data[args.feature].values
                X_eval = test_loader.data[args.feature].values
                y_train = train_loader.data[log_returns_col].values
                y_eval = test_loader.data[log_returns_col].values
                train_labels = train_loader.data[label_col].values
                eval_labels = test_loader.data[label_col].values

                print(
                    f"FINAL TEST: Training on combined train+validation data ({len(train_data)} samples)")
                print(
                    f"FINAL TEST: Evaluating on test data ({len(test_data)} samples)")
            else:
                # Then split train/val from the train_val set
                val_size_actual = int(train_val_size * args.val_size)
                train_size = train_val_size - val_size_actual

                train_indices = train_val_indices[:train_size]
                val_indices = train_val_indices[train_size:]

                # Create datasets
                train_data = data_loader.data.iloc[train_indices].copy()
                val_data = data_loader.data.iloc[val_indices].copy()

                print(
                    f"Train set: {len(train_data)} samples, Validation set: {len(val_data)} samples, Test set: {len(test_data)} samples")

                train_loader = FinancialDataLoader(
                    file_path=None, target_column=args.target, features=[args.feature],
                    normalize=normalize, data=train_data, device=device
                )

                val_loader = FinancialDataLoader(
                    file_path=None, target_column=args.target, features=[args.feature],
                    normalize=normalize, data=val_data, device=device
                )

                # Use validation set for evaluation
                X_train = train_loader.data[args.feature].values
                X_eval = val_loader.data[args.feature].values
                y_train = train_loader.data[log_returns_col].values
                y_eval = val_loader.data[log_returns_col].values
                train_labels = train_loader.data[label_col].values
                eval_labels = val_loader.data[label_col].values
        else:
            # Basic train-test split without validation
            X = data_loader.data[args.feature].values
            y = data_loader.data[log_returns_col].values
            labels = data_loader.data[label_col].values
            
            # Split data into train and test
            split_idx = int(len(X) * (1 - args.test_size))
            X_train, X_eval = X[:split_idx], X[split_idx:]
            y_train, y_eval = y[:split_idx], y[split_idx:]
            train_labels, eval_labels = labels[:split_idx], labels[split_idx:]
            
            print(f"Train set: {len(X_train)} samples, Test set: {len(X_eval)} samples")

        # Data for HMM
        if args.use_feature_for_hmm:
            hmm_train_data = X_train
            hmm_eval_data = X_eval
            print(f"Using feature values for HMM training")
        else:
            hmm_train_data = y_train
            hmm_eval_data = y_eval
            print(f"Using log returns for HMM training")

        # Verify data before discretization
        print(f"HMM train data shape: {hmm_train_data.shape}, range: [{np.min(hmm_train_data)}, {np.max(hmm_train_data)}]")

        # Use Discretizer class to prevent data leakage (fits on train, transforms test)
        try:
            discretizer = Discretizer(
                num_bins=args.observations, 
                strategy=args.discr_strategy,
                random_state=42
            )
            X_train_discrete = discretizer.fit_transform(hmm_train_data)
            X_eval_discrete = discretizer.transform(hmm_eval_data)
                
            print(f"Discretized data: train shape {X_train_discrete.shape}, unique values: {np.unique(X_train_discrete)}")
            print(f"Discretized eval data: shape {X_eval_discrete.shape}, unique values: {np.unique(X_eval_discrete)}")
            
        except Exception as e:
            print(f"Error in discretization: {str(e)}")
            raise

        if args.mode == 'forecasting':
            unique_bins = np.unique(X_train_discrete)
            bin_values = {}

            for bin_idx in unique_bins:
                bin_mask = (X_train_discrete == bin_idx)
                bin_values[bin_idx] = np.mean(hmm_train_data[bin_mask])

            obs_map = np.array([bin_values.get(i, 0)
                               for i in range(args.observations)])
            print(f"Created observation map: {obs_map}")
        else:
            obs_map = None

        if args.single_epoch:
            hmm, params = run_single_epoch_test(
                X_train_discrete, args.states, args.observations)
        else:
            T, E, T0 = initialize_hmm_params(args.states, args.observations)
            hmm = HiddenMarkovModel(T, E, T0, device='cpu', maxStep=args.steps)

            T0, T, E, converged = hmm.Baum_Welch_EM(X_train_discrete)

        print(f"\nEvaluating HMM in {args.mode} mode...")
        eval_metrics = hmm.evaluate(
            X_eval_discrete,
            mode=args.mode,
            actual_values=hmm_eval_data,
            actual_labels=eval_labels,
            observation_map=obs_map,
            class_threshold=args.class_threshold,
            direct_states=args.direct_states
        )

        print("\n" + "="*50)
        print("HMM MODEL EVALUATION REPORT")
        print("="*50)

        if args.mode == 'classification':
            print(f"Classification Metrics:")
            print(f"  Accuracy:  {eval_metrics['accuracy']:.4f}")
            print(f"  Precision: {eval_metrics['precision']:.4f}")
            print(f"  Recall:    {eval_metrics['recall']:.4f}")
            print(f"  F1 Score:  {eval_metrics['f1_score']:.4f}")

            print("\nConfusion Matrix:")
            print(eval_metrics['confusion_matrix'])

            print("\nState Interpretations:")
            for state, interp in eval_metrics['state_interpretations'].items():
                print(f"  State {state}: {interp['type']}")
                print(f"    Bull Ratio: {interp['bull_ratio']:.2f}")
                print(f"    Mean Return: {interp['mean']:.6f}")
                print(f"    Std Deviation: {interp['std']:.6f}")

        else:
            print(f"Forecasting Metrics:")
            print(f"  MSE:         {eval_metrics['mse']:.6f}")
            print(f"  MAE:         {eval_metrics['mae']:.6f}")
            print(f"  Correlation: {eval_metrics['correlation']:.4f}")

        plt.figure(figsize=(15, 12))

        if args.mode == 'classification':
            plt.subplot(3, 1, 1)
            plt.plot(X_eval, label=args.feature, color='blue')
            plt.title(f'{args.feature} with Inferred Market Regimes')

            states = eval_metrics['states_seq']
            unique_states = np.unique(states)
            colors = ['red', 'green', 'yellow', 'orange', 'purple']

            for i, state in enumerate(unique_states):
                state_interp = eval_metrics['state_interpretations'][state]
                state_type = state_interp['type']
                mask = (states == state)
                plt.fill_between(range(len(X_eval)), np.min(X_eval), np.max(X_eval),
                                 where=mask, alpha=0.3, color=colors[i % len(colors)],
                                 label=f"State {state}: {state_type}")
            plt.legend(loc='upper left')

            plt.subplot(3, 1, 2)
            plt.plot(hmm_eval_data, label='Log Returns', color='blue')
            plt.scatter(range(len(hmm_eval_data)), hmm_eval_data, c=eval_metrics['predicted_labels'],
                        cmap='coolwarm', alpha=0.6, label='Predicted Labels')
            plt.title('Log Returns with Bull/Bear Classifications')
            plt.axhline(y=0, color='black', linestyle='--')
            plt.legend()

            plt.subplot(3, 1, 3)
            cm = eval_metrics['confusion_matrix']
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=['Bear', 'Bull'])
            disp.plot(ax=plt.gca(), cmap='Blues', values_format='.0f')
            plt.title('Confusion Matrix (Bear vs Bull Classification)')

        else:
            plt.subplot(3, 1, 1)
            plt.plot(eval_metrics['actual_next_values'],
                     label='Actual Returns', color='blue')
            plt.plot(eval_metrics['forecasts'],
                     label='Predicted Returns', color='red')
            plt.title('One-Step-Ahead Return Forecasts')
            plt.axhline(y=0, color='black', linestyle='--')
            plt.legend()

            plt.subplot(3, 1, 2)
            plt.scatter(eval_metrics['actual_next_values'],
                        eval_metrics['forecasts'], alpha=0.5)
            plt.axhline(y=0, color='black', linestyle='--')
            plt.axvline(x=0, color='black', linestyle='--')
            plt.title(
                f'Actual vs Predicted Returns (Correlation: {eval_metrics["correlation"]:.4f})')
            plt.xlabel('Actual Returns')
            plt.ylabel('Predicted Returns')

            plt.subplot(3, 1, 3)
            plt.plot(X_eval, label=args.feature, color='blue')
            plt.title(f'{args.feature} with Inferred Market Regimes')

            states = eval_metrics['states_seq']
            unique_states = np.unique(states)
            colors = ['red', 'green', 'yellow', 'orange', 'purple']

            for i, state in enumerate(unique_states):
                mask = (states == state)
                plt.fill_between(range(len(X_eval)), np.min(X_eval), np.max(X_eval),
                                 where=mask, alpha=0.3, color=colors[i % len(colors)],
                                 label=f"State {state}")
            plt.legend(loc='upper left')

        plt.tight_layout()
        output_file = f'hmm_{args.mode}_results.png'
        plt.savefig(output_file)
        print(f"\nResults plot saved to '{output_file}'")

        model_file = f'hmm_{args.mode}_model.pt'
        hmm.save_model(model_file)
        print(f"Model saved to '{model_file}'")

        # Now load the model back from disk to inspect parameters
        print("\nLoading the model back from disk to inspect parameters...")
        loaded_hmm = HiddenMarkovModel.load_model(model_file)

        print("\n=== HMM Model Parameters ===")
        print("Initial State Probabilities (T0):")
        print(loaded_hmm.T0)

        print("\nTransition Matrix (T):")
        print(loaded_hmm.T)

        print("\nEmission Matrix (E):")
        print(loaded_hmm.E)
        print("============================")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
