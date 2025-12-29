# Analyzes HMM transition matrices and state-regime correlations, creating visualizations.
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)

from src.models.hmm_model import HiddenMarkovModel
from src.data.data_processor import FinancialDataLoader, Discretizer

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
    output_dir = Path(base_dir) / 'results' / 'visualizations'
output_dir.mkdir(exist_ok=True, parents=True)

def load_model_and_data():
    model_path = Path(base_dir) / 'results' / 'optimized_hmm_classification_model.pt'
    data_path = Path(base_dir) / 'data' / 'financial_data.csv'
    
    print(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu')
    
    # Load model parameters
    T = model_data['T'].numpy() if isinstance(model_data['T'], torch.Tensor) else model_data['T']
    E = model_data['E'].numpy() if isinstance(model_data['E'], torch.Tensor) else model_data['E']
    T0 = model_data['T0'].numpy() if isinstance(model_data['T0'], torch.Tensor) else model_data['T0']
    
    # Load data for correlation analysis
    print(f"Loading data from {data_path}...")
    loader = FinancialDataLoader(
        file_path=str(data_path),
        target_column='sp500 close',
        features=['sp500 high-low'],
        normalize=True
    )
    
    log_returns_col = loader.add_log_returns('sp500 close')
    label_col = loader.add_regime_labels(log_returns_col, threshold=0.0, window=5)
    
    return T, E, T0, loader.data, label_col

def analyze_transition_matrix(T, output_dir):
    print("\n" + "="*60)
    print("ANALYZING TRANSITION MATRIX")
    print("="*60)
    
    num_states = T.shape[0]
    
    # 1. Create transition matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(T, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=[f'State {i}' for i in range(num_states)],
                yticklabels=[f'State {i}' for i in range(num_states)],
                cbar_kws={'label': 'Transition Probability'})
    plt.title('HMM Transition Matrix\n(Probability of transitioning between states)', fontsize=14, fontweight='bold')
    plt.xlabel('To State', fontsize=12)
    plt.ylabel('From State', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'transition_matrix_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved transition matrix heatmap")
    
    # 2. Calculate regime persistence (average duration in each state)
    # Expected duration = 1 / (1 - self-transition probability)
    persistence = []
    for i in range(num_states):
        self_transition = T[i, i]
        expected_duration = 1 / (1 - self_transition) if self_transition < 1 else float('inf')
        persistence.append(expected_duration)
    
    # Create persistence bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(num_states), persistence, color=sns.color_palette("husl", num_states))
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Expected Duration (time steps)', fontsize=12)
    plt.title('Regime Persistence\n(Expected time spent in each state)', fontsize=14, fontweight='bold')
    plt.xticks(range(num_states), [f'State {i}' for i in range(num_states)])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, persistence)):
        if np.isfinite(val):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'regime_persistence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved regime persistence chart")
    
    # 3. Identify most common transition paths
    # Find transitions with highest probability (>0.2)
    transitions = []
    for i in range(num_states):
        for j in range(num_states):
            if i != j and T[i, j] > 0.2:  # Only significant transitions
                transitions.append({
                    'from': i,
                    'to': j,
                    'probability': T[i, j]
                })
    
    transitions_df = pd.DataFrame(transitions).sort_values('probability', ascending=False)
    
    # Create transition paths visualization
    if len(transitions_df) > 0:
        plt.figure(figsize=(12, 8))
        top_transitions = transitions_df.head(10)
        
        # Create a simple network-style visualization
        y_pos = np.arange(len(top_transitions))
        colors = plt.cm.viridis(top_transitions['probability'] / top_transitions['probability'].max())
        
        bars = plt.barh(y_pos, top_transitions['probability'], color=colors)
        plt.yticks(y_pos, [f"State {row['from']} → State {row['to']}" 
                           for _, row in top_transitions.iterrows()])
        plt.xlabel('Transition Probability', fontsize=12)
        plt.title('Top State Transitions\n(Most likely state transitions)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_transitions['probability'])):
            plt.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'top_transitions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved top transitions visualization")
    
    return persistence, transitions_df

def analyze_correlations(data, label_col, output_dir):
    print("\n" + "="*60)
    print("ANALYZING STATE-REGIME CORRELATIONS")
    print("="*60)
    
    # Load model and get state predictions for correlation analysis
    model_path = Path(base_dir) / 'results' / 'optimized_hmm_classification_model.pt'
    model_data = torch.load(model_path, map_location='cpu')
    
    T = model_data['T'].numpy() if isinstance(model_data['T'], torch.Tensor) else model_data['T']
    E = model_data['E'].numpy() if isinstance(model_data['E'], torch.Tensor) else model_data['E']
    T0 = model_data['T0'].numpy() if isinstance(model_data['T0'], torch.Tensor) else model_data['T0']
    
    hmm = HiddenMarkovModel(T, E, T0)
    
    # Get feature data and discretize
    feature_data = data['sp500 high-low'].values
    discretizer = Discretizer(num_bins=20, strategy='equal_freq', random_state=42)
    discrete_data = discretizer.fit_transform(feature_data)
    
    # Get state sequence
    states_seq, _ = hmm.viterbi_inference(torch.tensor(discrete_data, dtype=torch.int64))
    states_seq = states_seq.numpy()
    
    # Create state-regime correlation analysis
    data_with_states = data.copy()
    data_with_states['predicted_state'] = states_seq
    data_with_states['regime_label'] = data[label_col].values
    
    # Calculate state-regime correlations
    state_correlations = []
    for state in range(hmm.S):
        state_mask = data_with_states['predicted_state'] == state
        if state_mask.sum() > 0:
            # Calculate bull ratio (proportion of bull market days in this state)
            bull_ratio = (data_with_states.loc[state_mask, 'regime_label'] == 1).mean()
            
            # Calculate mean return for this state
            if 'sp500 close_log_return' in data_with_states.columns:
                mean_return = data_with_states.loc[state_mask, 'sp500 close_log_return'].mean()
                std_return = data_with_states.loc[state_mask, 'sp500 close_log_return'].std()
            else:
                mean_return = 0
                std_return = 0
            
            # Correlation between state indicator and regime label
            state_indicator = (data_with_states['predicted_state'] == state).astype(int)
            correlation = np.corrcoef(state_indicator, data_with_states['regime_label'])[0, 1]
            
            state_correlations.append({
                'state': state,
                'bull_ratio': bull_ratio,
                'mean_return': mean_return,
                'std_return': std_return,
                'correlation': correlation,
                'count': state_mask.sum()
            })
    
    corr_df = pd.DataFrame(state_correlations)
    
    # Create correlation visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bull ratio by state
    axes[0].bar(corr_df['state'], corr_df['bull_ratio'], color=sns.color_palette("RdYlGn", len(corr_df)))
    axes[0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Neutral (50%)')
    axes[0].set_xlabel('State', fontsize=12)
    axes[0].set_ylabel('Bull Market Ratio', fontsize=12)
    axes[0].set_title('Bull Market Ratio by State', fontsize=13, fontweight='bold')
    axes[0].set_xticks(corr_df['state'])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, row in corr_df.iterrows():
        axes[0].text(row['state'], row['bull_ratio'] + 0.02,
                    f'{row["bull_ratio"]:.2f}', ha='center', fontweight='bold')
    
    # Plot 2: State-regime correlation
    colors_corr = ['red' if x < 0 else 'green' for x in corr_df['correlation']]
    axes[1].barh(corr_df['state'], corr_df['correlation'], color=colors_corr, alpha=0.7)
    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[1].set_xlabel('Correlation with Bull Market', fontsize=12)
    axes[1].set_ylabel('State', fontsize=12)
    axes[1].set_title('State-Regime Correlation', fontsize=13, fontweight='bold')
    axes[1].set_yticks(corr_df['state'])
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, row in corr_df.iterrows():
        axes[1].text(row['correlation'] + (0.05 if row['correlation'] > 0 else -0.05),
                    row['state'], f'{row["correlation"]:.3f}',
                    va='center', ha='left' if row['correlation'] > 0 else 'right',
                    fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'state_regime_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved state-regime correlation analysis")
    
    # Create summary table
    print("\nState-Regime Correlation Summary:")
    print(corr_df.to_string(index=False))
    
    return corr_df

def create_summary_report(persistence, transitions_df, corr_df, output_dir):
    report_path = output_dir / 'regime_transition_analysis_summary.md'
    
    with open(report_path, 'w') as f:
        f.write("# Regime Transition Pattern Analysis Summary\n\n")
        
        f.write("## 1. Regime Persistence\n\n")
        f.write("Expected duration (time steps) spent in each state:\n\n")
        for i, duration in enumerate(persistence):
            f.write(f"- **State {i}**: {duration:.2f} time steps\n")
        f.write("\n")
        
        f.write("## 2. Top State Transitions\n\n")
        f.write("Most probable transitions between states:\n\n")
        if len(transitions_df) > 0:
            f.write("| From | To | Probability |\n")
            f.write("|------|----|----|\n")
            for _, row in transitions_df.head(10).iterrows():
                f.write(f"| State {row['from']} | State {row['to']} | {row['probability']:.3f} |\n")
        f.write("\n")
        
        f.write("## 3. State-Regime Correlations\n\n")
        f.write("Relationship between predicted states and actual market regimes:\n\n")
        f.write("| State | Bull Ratio | Mean Return | Correlation | Count |\n")
        f.write("|-------|------------|-------------|-------------|-------|\n")
        for _, row in corr_df.iterrows():
            f.write(f"| {row['state']} | {row['bull_ratio']:.3f} | {row['mean_return']:.4f} | {row['correlation']:.3f} | {row['count']} |\n")
        f.write("\n")
        
        # Identify regime types
        f.write("## 4. Regime Classification\n\n")
        for _, row in corr_df.iterrows():
            if row['bull_ratio'] > 0.6:
                regime_type = "Bull Market"
            elif row['bull_ratio'] < 0.4:
                regime_type = "Bear Market"
            else:
                regime_type = "Mixed/Transitional"
            f.write(f"- **State {row['state']}**: {regime_type} (Bull ratio: {row['bull_ratio']:.2f})\n")
    
    print(f"✓ Saved summary report to {report_path}")

def main():
    print("="*70)
    print("REGIME TRANSITION PATTERN ANALYSIS")
    print("="*70)
    
    try:
        # Load model and data
        T, E, T0, data, label_col = load_model_and_data()
        
        # Analyze transition matrix
        persistence, transitions_df = analyze_transition_matrix(T, output_dir)
        
        # Analyze correlations
        corr_df = analyze_correlations(data, label_col, output_dir)
        
        # Create summary report
        create_summary_report(persistence, transitions_df, corr_df, output_dir)
        
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nAll visualizations saved to: {output_dir}")
        print("\nGenerated files:")
        print("  - transition_matrix_analysis.png")
        print("  - regime_persistence.png")
        print("  - top_transitions.png")
        print("  - state_regime_correlations.png")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()

