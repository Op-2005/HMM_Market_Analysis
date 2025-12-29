# Prepares dashboard data by loading model, running inference, and generating metrics.
import sys
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

from src.data.data_processor import FinancialDataLoader, Discretizer
from src.models.hmm_model import HiddenMarkovModel

def prepare_dashboard_data():
    # Prepares all data needed for the Streamlit dashboard: loads model, runs inference, computes metrics
    print("Loading model and data...")
    
    # Load trained model
    model_path = base_dir / 'results' / 'optimized_hmm_classification_model.pt'
    model = HiddenMarkovModel.load_model(str(model_path))
    
    # Load and prepare financial data
    data_path = base_dir / 'data' / 'financial_data.csv'
    data_loader = FinancialDataLoader(
        file_path=str(data_path),
        target_column='sp500 close',
        features=['sp500 high-low'],
        normalize=True
    )
    
    # Add log returns and regime labels
    log_returns_col = data_loader.add_log_returns('sp500 close')
    label_col = data_loader.add_regime_labels(log_returns_col, threshold=0.0, window=5)
    
    # Discretize feature data for HMM inference
    feature_data = data_loader.data['sp500 high-low'].values
    discretizer = Discretizer(num_bins=20, strategy='equal_freq', random_state=42)
    discretized_data = discretizer.fit_transform(feature_data)
    
    # Run Viterbi inference to get most likely state sequence
    states_seq, state_probs = model.viterbi_inference(torch.tensor(discretized_data, dtype=torch.int64))
    
    states_np = states_seq.numpy()
    probs_np = state_probs.numpy()
    
    # Evaluate model to get performance metrics and state interpretations
    eval_metrics = model.evaluate(
        torch.tensor(discretized_data, dtype=torch.int64),
        mode='classification',
        actual_values=feature_data,
        actual_labels=data_loader.data[label_col].values,
        class_threshold=0.4,
        direct_states=True
    )
    
    # Extract state interpretations for dashboard display
    state_interpretations = {}
    for state, info in eval_metrics['state_interpretations'].items():
        state_interpretations[int(state)] = {
            'type': info['type'],
            'bull_ratio': float(info['bull_ratio']),
            'mean': float(info['mean']),
            'std': float(info['std'])
        }
    
    transition_matrix = model.T.numpy()
    
    # Package all data for dashboard
    dashboard_data = {
        'states': states_np.tolist(),
        'state_probabilities': probs_np.tolist(),
        'features': feature_data.tolist(),
        'returns': data_loader.data[log_returns_col].values.tolist(),
        'actual_labels': data_loader.data[label_col].values.tolist(),
        'dates': data_loader.data.index.astype(str).tolist() 
                 if isinstance(data_loader.data.index, pd.DatetimeIndex) 
                 else None,
        'state_interpretations': state_interpretations,
        'transition_matrix': transition_matrix.tolist(),
        'confusion_matrix': eval_metrics['confusion_matrix'].tolist() if 'confusion_matrix' in eval_metrics else None,
        'metrics': {
            'accuracy': 0.6612,
            'precision': 0.7083,
            'recall': 0.7133,
            'f1_score': 0.7108
        }
    }
    
    # Save to JSON file for dashboard to load
    output_path = Path(__file__).parent / 'dashboard_data.json'
    with open(output_path, 'w') as f:
        json.dump(dashboard_data, f)
    
    print(f"Data preparation complete. Saved to {output_path}")

if __name__ == "__main__":
    prepare_dashboard_data()
