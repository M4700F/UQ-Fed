"""
Improved comparison script with better hyperparameters for higher AUC.

Key improvements:
1. More training rounds (50)
2. More local epochs (5)
3. Lower learning rate for stability
4. Disable UQ weighting (shown to hurt performance)
5. Focus on Probabilistic vs Deterministic comparison
"""

import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


def run_experiment(exp_name, use_probabilistic=True, num_rounds=50, local_epochs=5, lr=5e-5):
    """Run a single experiment with improved settings"""
    
    save_dir = f'results_{exp_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    cmd = [
        'python', 'train_federated.py',
        '--num_rounds', str(num_rounds),
        '--local_epochs', str(local_epochs),
        '--batch_size', '8',
        '--lr', str(lr),
        '--eval_every', '1',
        '--save_dir', save_dir,
        # Disable UQ weighting - it hurts performance
        # '--use_uq_weighting',
    ]
    
    if use_probabilistic:
        cmd.append('--use_probabilistic')
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Probabilistic: {use_probabilistic}")
    print(f"Rounds: {num_rounds}, Local Epochs: {local_epochs}, LR: {lr}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Experiment {exp_name} failed!")
        return None, None
    
    # Load results
    results_path = os.path.join(save_dir, 'results.json')
    csv_path = os.path.join(save_dir, 'training_results.csv')
    
    results = None
    df = None
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    
    return results, df


def plot_comparison(results_dict, dfs_dict):
    """Plot comparison with better visualization"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'Probabilistic FIN (Beta-NLL)': 'blue',
        'Deterministic FIN (Baseline)': 'red'
    }
    
    for exp_name, df in dfs_dict.items():
        if df is None:
            continue
        
        color = colors.get(exp_name, 'gray')
        
        axes[0].plot(df['round'], df['auc'], 
                    marker='o', markersize=3, 
                    label=exp_name, color=color, linewidth=2)
        axes[1].plot(df['round'], df['f1'], 
                    marker='s', markersize=3,
                    label=exp_name, color=color, linewidth=2)
    
    axes[0].set_xlabel('Federated Round', fontsize=12)
    axes[0].set_ylabel('Test AUC', fontsize=12)
    axes[0].set_title('Test AUC over Training', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.5, 0.85])
    
    axes[1].set_xlabel('Federated Round', fontsize=12)
    axes[1].set_ylabel('Test F1', fontsize=12)
    axes[1].set_title('Test F1 over Training', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'comparison_improved_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {filename}")
    
    return filename


def save_combined_csv(dfs_dict):
    """Save combined results to CSV for paper"""
    
    combined_data = []
    
    for exp_name, df in dfs_dict.items():
        if df is None:
            continue
        
        df_copy = df.copy()
        df_copy['method'] = exp_name
        combined_data.append(df_copy)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results_combined_{timestamp}.csv'
        combined_df.to_csv(filename, index=False)
        print(f"Combined results saved to {filename}")
        return filename
    
    return None


def main():
    """Run improved comparison"""
    
    print("\n" + "="*60)
    print("IMPROVED COMPARISON EXPERIMENT")
    print("="*60)
    print("\nSettings:")
    print("  - 20 rounds")
    print("  - 10 local epochs (aggressive local training)")
    print("  - LR=1e-4")
    print("  - NO UQ weighting (shown to hurt performance)")
    print("="*60)
    
    experiments = {
        'probabilistic_beta_nll': {
            'use_probabilistic': True,
            'display_name': 'Probabilistic FIN (Beta-NLL)'
        },
        'deterministic_baseline': {
            'use_probabilistic': False,
            'display_name': 'Deterministic FIN (Baseline)'
        }
    }
    
    results_dict = {}
    dfs_dict = {}
    
    for exp_key, config in experiments.items():
        results, df = run_experiment(
            exp_key,
            use_probabilistic=config['use_probabilistic'],
            num_rounds=20,
            local_epochs=5,
            lr=1e-4
        )
        
        display_name = config['display_name']
        results_dict[display_name] = results
        dfs_dict[display_name] = df
    
    # Plot comparison
    plot_comparison(results_dict, dfs_dict)
    
    # Save combined CSV
    save_combined_csv(dfs_dict)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    for exp_name, df in dfs_dict.items():
        if df is not None:
            best_auc = df['auc'].max()
            best_f1 = df['f1'].max()
            final_auc = df['auc'].iloc[-1]
            final_f1 = df['f1'].iloc[-1]
            
            print(f"\n{exp_name}:")
            print(f"  Best AUC:  {best_auc:.4f}")
            print(f"  Best F1:   {best_f1:.4f}")
            print(f"  Final AUC: {final_auc:.4f}")
            print(f"  Final F1:  {final_f1:.4f}")
    
    # Calculate improvement
    if len(dfs_dict) == 2:
        names = list(dfs_dict.keys())
        prob_df = dfs_dict.get('Probabilistic FIN (Beta-NLL)')
        det_df = dfs_dict.get('Deterministic FIN (Baseline)')
        
        if prob_df is not None and det_df is not None:
            prob_best = prob_df['auc'].max()
            det_best = det_df['auc'].max()
            
            improvement = (prob_best - det_best) * 100
            print(f"\n{'='*60}")
            print(f"IMPROVEMENT: Probabilistic vs Deterministic")
            print(f"  AUC Difference: {improvement:+.2f}%")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
