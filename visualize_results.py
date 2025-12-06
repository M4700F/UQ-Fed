"""
Visualize and analyze results from experiments
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(checkpoint_dir):
    """Load results from checkpoint directory"""
    results_path = os.path.join(checkpoint_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def plot_single_experiment(results, title, save_path=None):
    """Plot results from a single experiment"""

    if results is None or len(results) == 0:
        print(f"No results found for {title}")
        return

    rounds = [r['round'] for r in results]
    aucs = [r['auc'] for r in results]
    f1s = [r['f1'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AUC plot
    ax1.plot(rounds, aucs, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Test AUC', fontsize=12)
    ax1.set_title(f'{title} - Test AUC', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.7, 1.0])

    # Annotate best
    best_auc_idx = np.argmax(aucs)
    ax1.annotate(f'Best: {aucs[best_auc_idx]:.4f}',
                xy=(rounds[best_auc_idx], aucs[best_auc_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # F1 plot
    ax2.plot(rounds, f1s, 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Federated Round', fontsize=12)
    ax2.set_ylabel('Test F1 Score', fontsize=12)
    ax2.set_title(f'{title} - Test F1', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.6, 1.0])

    # Annotate best
    best_f1_idx = np.argmax(f1s)
    ax2.annotate(f'Best: {f1s[best_f1_idx]:.4f}',
                xy=(rounds[best_f1_idx], f1s[best_f1_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def compare_experiments(results_dict, save_path='comparison.png'):
    """Compare multiple experiments"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    for idx, (exp_name, results) in enumerate(results_dict.items()):
        if results is None or len(results) == 0:
            continue

        rounds = [r['round'] for r in results]
        aucs = [r['auc'] for r in results]
        f1s = [r['f1'] for r in results]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        ax1.plot(rounds, aucs, color=color, marker=marker,
                linewidth=2, markersize=6, label=exp_name)
        ax2.plot(rounds, f1s, color=color, marker=marker,
                linewidth=2, markersize=6, label=exp_name)

    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Test AUC', fontsize=12)
    ax1.set_title('Test AUC Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.7, 1.0])

    ax2.set_xlabel('Federated Round', fontsize=12)
    ax2.set_ylabel('Test F1 Score', fontsize=12)
    ax2.set_title('Test F1 Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.6, 1.0])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")


def print_summary_table(results_dict):
    """Print summary table of results"""

    print("\n" + "="*80)
    print(" " * 25 + "RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<40} {'Final AUC':<12} {'Best AUC':<12} {'Final F1':<12} {'Best F1':<12}")
    print("-"*80)

    for exp_name, results in results_dict.items():
        if results is None or len(results) == 0:
            print(f"{exp_name:<40} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue

        aucs = [r['auc'] for r in results]
        f1s = [r['f1'] for r in results]

        final_auc = aucs[-1]
        best_auc = max(aucs)
        final_f1 = f1s[-1]
        best_f1 = max(f1s)

        print(f"{exp_name:<40} {final_auc:<12.4f} {best_auc:<12.4f} {final_f1:<12.4f} {best_f1:<12.4f}")

    print("="*80 + "\n")


def analyze_improvements(prob_results, det_results):
    """Analyze improvement of probabilistic over deterministic"""

    if prob_results is None or det_results is None:
        print("Cannot compute improvements - missing results")
        return

    prob_aucs = [r['auc'] for r in prob_results]
    det_aucs = [r['auc'] for r in det_results]

    prob_f1s = [r['f1'] for r in prob_results]
    det_f1s = [r['f1'] for r in det_results]

    # Final improvements
    final_auc_imp = ((prob_aucs[-1] - det_aucs[-1]) / det_aucs[-1]) * 100
    final_f1_imp = ((prob_f1s[-1] - det_f1s[-1]) / det_f1s[-1]) * 100

    # Best improvements
    best_auc_imp = ((max(prob_aucs) - max(det_aucs)) / max(det_aucs)) * 100
    best_f1_imp = ((max(prob_f1s) - max(det_f1s)) / max(det_f1s)) * 100

    print("\n" + "="*80)
    print(" " * 20 + "IMPROVEMENT ANALYSIS")
    print("="*80)
    print(f"\nProbabilistic FIN vs Deterministic FIN:")
    print(f"  Final AUC improvement: {final_auc_imp:+.2f}%")
    print(f"  Best AUC improvement:  {best_auc_imp:+.2f}%")
    print(f"  Final F1 improvement:  {final_f1_imp:+.2f}%")
    print(f"  Best F1 improvement:   {best_f1_imp:+.2f}%")
    print("="*80 + "\n")


def main():
    """Main function"""

    print("Searching for experiment results...")

    # Find all checkpoint directories
    checkpoint_dirs = [
        'checkpoints_prob',
        'checkpoints_det',
        'checkpoints_probabilistic_fin_uq_weighted',
        'checkpoints_deterministic_fin_standard',
        'checkpoints_probabilistic_fin_standard',
        'checkpoints_deterministic_fin_uq_weighted'
    ]

    results_dict = {}
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            results = load_results(checkpoint_dir)
            if results:
                # Clean name for display
                name = checkpoint_dir.replace('checkpoints_', '').replace('_', ' ').title()
                results_dict[name] = results
                print(f"✓ Found results in {checkpoint_dir}")

    if not results_dict:
        print("\nNo results found! Run training first:")
        print("  python3 train_federated.py --use_probabilistic --use_uq_weighting")
        return

    # Print summary table
    print_summary_table(results_dict)

    # Plot individual experiments
    for name, results in results_dict.items():
        plot_single_experiment(
            results,
            name,
            save_path=f"plot_{name.replace(' ', '_').lower()}.png"
        )

    # Plot comparison if multiple experiments
    if len(results_dict) > 1:
        compare_experiments(results_dict, save_path='comparison_all.png')

    # Analyze improvements
    if 'Prob' in results_dict and 'Det' in results_dict:
        analyze_improvements(results_dict['Prob'], results_dict['Det'])
    elif 'Probabilistic Fin Uq Weighted' in results_dict and 'Deterministic Fin Standard' in results_dict:
        analyze_improvements(
            results_dict['Probabilistic Fin Uq Weighted'],
            results_dict['Deterministic Fin Standard']
        )

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
