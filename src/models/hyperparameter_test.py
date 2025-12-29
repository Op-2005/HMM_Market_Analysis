# Main training script for HMM model with hyperparameter optimization and structured emission matrix initialization.
import torch
import numpy as np
import pandas as pd
import os
import time
from src.data.data_processor import FinancialDataLoader, Discretizer
from src.models.hmm_model import HiddenMarkovModel

np.random.seed(42)
torch.manual_seed(42)
device = 'cpu'

def run_hyperparameter_test(
    data_file='financial_data.csv',
    feature='sp500 high-low',
    target='sp500 close',
    mode='classification',
    states=3,
    observations=15,
    steps=20,
    discr_strategy='equal_freq',
    class_threshold=0.5,
    use_feature_for_hmm=False,
    direct_states=False,
    test_size=0.2,
    val_size=0.2,
    use_validation=True,
    final_test=False
):
    print("\n" + "="*70)
    print(f"RUNNING TEST: states={states}, observations={observations}, strategy={discr_strategy}")
    print(f"          steps={steps}, direct_states={direct_states}, feature={feature}")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(base_dir, 'data', data_file)
    normalize = True
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data_loader = FinancialDataLoader(
        file_path=file_path,
        target_column=target,
        features=[feature],
        normalize=normalize
    )
    
    log_returns_col = data_loader.add_log_returns(target)
    label_col = data_loader.add_regime_labels(log_returns_col, threshold=0.0, window=5)
    
    # Split data into train/validation/test sets
    total_samples = len(data_loader.data)
    test_size_actual = int(total_samples * test_size)
    train_val_size = total_samples - test_size_actual
    indices = np.arange(total_samples)
    train_val_indices = indices[:train_val_size]
    test_indices = indices[train_val_size:]
    
    test_data = data_loader.data.iloc[test_indices].copy()
    test_loader = FinancialDataLoader(
        file_path=None, target_column=target, features=[feature],
        normalize=normalize, data=test_data, device=device
    )
    
    if final_test:
        # Use all non-test data for training
        train_data = data_loader.data.iloc[train_val_indices].copy()
        train_loader = FinancialDataLoader(
            file_path=None, target_column=target, features=[feature],
            normalize=normalize, data=train_data, device=device
        )
        X_train = train_loader.data[feature].values
        X_eval = test_loader.data[feature].values
        train_labels = train_loader.data[label_col].values
        eval_labels = test_loader.data[label_col].values
    else:
        # Split train/val from the train_val set
        val_size_actual = int(train_val_size * val_size)
        train_size = train_val_size - val_size_actual
        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]
        
        train_data = data_loader.data.iloc[train_indices].copy()
        val_data = data_loader.data.iloc[val_indices].copy()
        
        train_loader = FinancialDataLoader(
            file_path=None, target_column=target, features=[feature],
            normalize=normalize, data=train_data, device=device
        )
        val_loader = FinancialDataLoader(
            file_path=None, target_column=target, features=[feature],
            normalize=normalize, data=val_data, device=device
        )
        X_train = train_loader.data[feature].values
        X_eval = val_loader.data[feature].values
        train_labels = train_loader.data[label_col].values
        eval_labels = val_loader.data[label_col].values
    
    # Choose data source for HMM training
    if use_feature_for_hmm:
        hmm_train_data = X_train
        hmm_eval_data = X_eval
    else:
        hmm_train_data = train_loader.data[log_returns_col].values
        hmm_eval_data = test_loader.data[log_returns_col].values if final_test else val_loader.data[log_returns_col].values
    
    # Discretize data using Discretizer to prevent data leakage
    discretizer = Discretizer(
        num_bins=observations, 
        strategy=discr_strategy,
        random_state=42
    )
    X_train_discrete = discretizer.fit_transform(hmm_train_data)
    X_eval_discrete = discretizer.transform(hmm_eval_data)
    
    # Initialize and train HMM
    T, E, T0 = initialize_structured_hmm_params(states, observations)
    hmm = HiddenMarkovModel(T, E, T0, device='cpu', maxStep=steps)
    
    start_time = time.time()
    T0, T, E, converged = hmm.Baum_Welch_EM(X_train_discrete)
    training_time = time.time() - start_time
    
    # Evaluate model
    eval_metrics = hmm.evaluate(
        X_eval_discrete,
        mode=mode,
        actual_values=hmm_eval_data,
        actual_labels=eval_labels,
        observation_map=None,
        class_threshold=class_threshold,
        direct_states=direct_states
    )
    
    # Add metadata to results
    eval_metrics.update({
        'states': states,
        'observations': observations,
        'steps': steps,
        'discr_strategy': discr_strategy,
        'direct_states': direct_states,
        'feature': feature,
        'training_time': training_time,
        'converged': converged
    })
    
    # Remove large arrays from results
    if 'states_seq' in eval_metrics:
        del eval_metrics['states_seq']
    if 'predicted_labels' in eval_metrics:
        del eval_metrics['predicted_labels']
    
    return eval_metrics

def initialize_structured_hmm_params(num_states, num_observations):
    # Initialize HMM parameters with structured emission matrix for better regime separation
    # Transition matrix with higher self-transition probability for stability
    T = np.ones((num_states, num_states)) / num_states
    for i in range(num_states):
        T[i, i] = 0.4
    T = T / T.sum(axis=1, keepdims=True)
    
    # Structured emission matrix encoding different market regimes
    E = np.ones((num_states, num_observations)) / num_observations
    
    # For 5 states: structure them as bull/mixed/bear regimes based on volatility patterns
    if num_states >= 5 and num_observations >= 10:
        # Strong bull market - low volatility
        E[0, :num_observations//5] = 0.7 / (num_observations//5)
        E[0, num_observations//5:] = 0.3 / (num_observations - num_observations//5)
        
        # Moderate bull market - low-medium volatility
        mid_low = num_observations//5
        mid_high = 2*num_observations//5
        E[1, mid_low:mid_high] = 0.6 / (mid_high - mid_low)
        E[1, :mid_low] = 0.2 / mid_low
        E[1, mid_high:] = 0.2 / (num_observations - mid_high)
        
        # Sideways market - medium volatility
        mid_low = 2*num_observations//5
        mid_high = 3*num_observations//5
        E[2, mid_low:mid_high] = 0.6 / (mid_high - mid_low)
        E[2, :mid_low] = 0.2 / mid_low
        E[2, mid_high:] = 0.2 / (num_observations - mid_high)
        
        # Moderate bear market - medium-high volatility
        mid_low = 3*num_observations//5
        mid_high = 4*num_observations//5
        E[3, mid_low:mid_high] = 0.6 / (mid_high - mid_low)
        E[3, :mid_low] = 0.2 / mid_low
        E[3, mid_high:] = 0.2 / (num_observations - mid_high)
        
        # Strong bear market - high volatility
        E[4, 4*num_observations//5:] = 0.7 / (num_observations - 4*num_observations//5)
        E[4, :4*num_observations//5] = 0.3 / (4*num_observations//5)
    
    E = E / E.sum(axis=1, keepdims=True)
    
    # Initial state distribution with slight bias toward bull states
    T0 = np.ones(num_states) / num_states
    if num_states >= 5:
        T0[:3] = 0.6 / 3
        T0[3:] = 0.4 / (num_states - 3)
    
    return T, E, T0

def run_optimized_test():
    # Run optimized model test with best known hyperparameters
    print("\n" + "="*70)
    print("RUNNING OPTIMIZED MODEL TEST")
    print("="*70)
    
    params = {
        'states': 5,
        'observations': 20,
        'discr_strategy': 'equal_freq',
        'direct_states': True,
        'feature': 'sp500 high-low',
        'steps': 60,
        'mode': 'classification',
        'target': 'sp500 close',
        'test_size': 0.2,
        'val_size': 0.05,
        'final_test': True,
        'class_threshold': 0.4
    }
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(base_dir, 'data', 'financial_data.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data_loader = FinancialDataLoader(
        file_path=file_path,
        target_column=params['target'],
        features=[params['feature']],
        normalize=True
    )
    
    log_returns_col = data_loader.add_log_returns(params['target'])
    label_col = data_loader.add_regime_labels(log_returns_col, threshold=0.0, window=5)
    
    # Create train/test split
    total_samples = len(data_loader.data)
    test_size_actual = int(total_samples * params['test_size'])
    train_val_size = total_samples - test_size_actual
    indices = np.arange(total_samples)
    train_val_indices = indices[:train_val_size]
    test_indices = indices[train_val_size:]
    
    test_data = data_loader.data.iloc[test_indices].copy()
    train_data = data_loader.data.iloc[train_val_indices].copy()
    
    test_loader = FinancialDataLoader(
        file_path=None, target_column=params['target'], features=[params['feature']],
        normalize=True, data=test_data, device=device
    )
    train_loader = FinancialDataLoader(
        file_path=None, target_column=params['target'], features=[params['feature']],
        normalize=True, data=train_data, device=device
    )
    
    X_train = train_loader.data[params['feature']].values
    X_eval = test_loader.data[params['feature']].values
    eval_labels = test_loader.data[label_col].values
    
    # Discretize feature data
    discretizer = Discretizer(
        num_bins=params['observations'], 
        strategy=params['discr_strategy'],
        random_state=42
    )
    X_train_discrete = discretizer.fit_transform(X_train)
    X_eval_discrete = discretizer.transform(X_eval)
    
    # Initialize and train HMM
    T, E, T0 = initialize_structured_hmm_params(params['states'], params['observations'])
    hmm = HiddenMarkovModel(T, E, T0, device='cpu', maxStep=params['steps'])
    
    start_time = time.time()
    T0, T, E, converged = hmm.Baum_Welch_EM(X_train_discrete)
    training_time = time.time() - start_time
    
    # Evaluate model
    eval_metrics = hmm.evaluate(
        X_eval_discrete,
        mode=params['mode'],
        actual_values=X_eval,
        actual_labels=eval_labels,
        class_threshold=params['class_threshold'],
        direct_states=params['direct_states']
    )
    
    # Save model
    model_dir = os.path.join(base_dir, 'results')
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, 'optimized_hmm_classification_model.pt')
    hmm.save_model(model_file)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {eval_metrics['accuracy']:.4f}")
    print(f"Precision: {eval_metrics['precision']:.4f}")
    print(f"Recall:    {eval_metrics['recall']:.4f}")
    print(f"F1 Score:  {eval_metrics['f1_score']:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    eval_metrics.update({
        'states': params['states'],
        'observations': params['observations'],
        'steps': params['steps'],
        'discr_strategy': params['discr_strategy'],
        'direct_states': params['direct_states'],
        'feature': params['feature'],
        'class_threshold': params['class_threshold'],
        'training_time': training_time,
        'converged': converged
    })
    
    return eval_metrics, hmm

if __name__ == "__main__":
    results, model = run_optimized_test()
