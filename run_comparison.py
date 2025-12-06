"""
Script to run comparison between Probabilistic FIN and Original FIN
"""

import subprocess
import json
import matplotlib.pyplot as plt
import os


def run_experiment(exp_name, use_probabilistic=True, use_uq_weighting=True):
    """Run a single experiment"""

    cmd = [
        'python', 'train_federated.py',
        '--num_rounds', '30',
        '--local_epochs', '3',
        '--batch_size', '8',
        '--lr', '1e-4',
        '--eval_every', '1',
        '--save_dir', f'checkpoints_{exp_name}'
    ]

    if use_probabilistic:
        cmd.append('--use_probabilistic')

    if use_uq_weighting:
        cmd.append('--use_uq_weighting')
        cmd.extend(['--uq_alpha', '0.5'])

    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Experiment {exp_name} failed!")
        return None

    # Load results
    results_path = os.path.join(f'checkpoints_{exp_name}', 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)

    return None


def plot_comparison(results_dict):
    """Plot comparison of different methods"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for exp_name, results in results_dict.items():
        if results is None:
            continue

        rounds = [r['round'] for r in results]
        aucs = [r['auc'] for r in results]
        f1s = [r['f1'] for r in results]

        ax1.plot(rounds, aucs, marker='o', label=exp_name)
        ax2.plot(rounds, f1s, marker='s', label=exp_name)

    ax1.set_xlabel('Round')
    ax1.set_ylabel('Test AUC')
    ax1.set_title('Test AUC over Federated Rounds')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Round')
    ax2.set_ylabel('Test F1')
    ax2.set_title('Test F1 over Federated Rounds')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300)
    print("\nComparison plot saved to comparison_results.png")


def main():
    """Run all experiments"""

    experiments = {
        'probabilistic_fin_uq_weighted': {
            'use_probabilistic': True,
            'use_uq_weighting': True,
            'description': 'Probabilistic FIN with UQ-weighted aggregation (OURS)'
        },
        'deterministic_fin_standard': {
            'use_probabilistic': False,
            'use_uq_weighting': False,
            'description': 'Deterministic FIN with standard FedAvg (BASELINE)'
        },
        'probabilistic_fin_standard': {
            'use_probabilistic': True,
            'use_uq_weighting': False,
            'description': 'Probabilistic FIN with standard FedAvg (Ablation 1)'
        },
        'deterministic_fin_uq_weighted': {
            'use_probabilistic': False,
            'use_uq_weighting': True,
            'description': 'Deterministic FIN with UQ-weighted aggregation (Ablation 2)'
        }
    }

    results_dict = {}

    print("\n" + "="*60)
    print("RUNNING ALL EXPERIMENTS")
    print("="*60)

    for exp_name, config in experiments.items():
        print(f"\n{config['description']}")

        results = run_experiment(
            exp_name,
            use_probabilistic=config['use_probabilistic'],
            use_uq_weighting=config['use_uq_weighting']
        )

        results_dict[config['description']] = results

    # Plot comparison
    plot_comparison(results_dict)

    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    for exp_name, results in results_dict.items():
        if results is not None and len(results) > 0:
            final_result = results[-1]
            best_auc = max([r['auc'] for r in results])
            best_f1 = max([r['f1'] for r in results])

            print(f"\n{exp_name}:")
            print(f"  Final AUC: {final_result['auc']:.4f}")
            print(f"  Final F1: {final_result['f1']:.4f}")
            print(f"  Best AUC: {best_auc:.4f}")
            print(f"  Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
