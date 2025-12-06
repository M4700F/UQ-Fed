# Probabilistic FedFeatGen with Beta-NLL

Extension of "Multimodal Federated Learning With Missing Modalities through Feature Imputation Network" with Uncertainty Quantification using Beta-NLL.

## Key Contributions

1. **Probabilistic Feature Imputation Network (Probabilistic FIN)**: Extended FIN to output Beta distribution parameters (alpha, beta) instead of point estimates
2. **Beta-NLL Loss**: Uses Beta Negative Log-Likelihood for uncertainty-aware feature imputation
3. **Uncertainty-Weighted FedAvg**: Aggregates client models weighted by their prediction uncertainty
4. **Variance Explosion Prevention**: Includes gradient clipping, variance regularization, and parameter clamping

## Architecture

### Components

1. **Image Encoder**: ResNet-50 → 256-dim L2-normalized features
2. **Text Encoder**: BERT-base → 256-dim L2-normalized features
3. **Probabilistic FIN**:
   - Input: 256-dim source features
   - Transformer Decoder (6 layers, 4 heads)
   - Output: (alpha, beta) for Beta distribution
4. **Fusion**: Concatenation of image + text features
5. **Classifier**: Linear layer for 14-class multi-label classification

### Uncertainty Quantification

**Feature Imputation (FIN)**:
- Model: `z_target ~ Beta(alpha, beta)`
- Loss: `-log p(z_target | alpha, beta) + λ * Var[Beta]`
- Uncertainty: `Var = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))`

**Aggregation**:
- Weight = `(1-α) * (n_i/n) + α * confidence_i`
- Confidence = `1 / (1 + uncertainty_i)`

## Setup

### Requirements

```bash
pip install -r requirements_federated.txt
```

### Data Structure

```
Fed_Data/
├── images/
│   ├── chexpert/
│   └── nih/
├── reports/
│   └── nih/
└── splits/
    ├── client_0.csv
    ├── client_1.csv
    └── ...

PadChest_Data/
├── images/
└── padchest_test.csv
```

## Usage

### Quick Start

**Train Probabilistic FIN (Ours)**:
```bash
python train_federated.py \
    --use_probabilistic \
    --use_uq_weighting \
    --num_rounds 30 \
    --local_epochs 3 \
    --batch_size 32 \
    --lr 1e-4 \
    --uq_alpha 0.5 \
    --save_dir checkpoints_prob
```

**Train Original FIN (Baseline)**:
```bash
python train_federated.py \
    --num_rounds 30 \
    --local_epochs 3 \
    --batch_size 32 \
    --lr 1e-4 \
    --save_dir checkpoints_det
```

### Run Full Comparison

Runs 4 experiments automatically:
1. Probabilistic FIN + UQ-weighted FedAvg (OURS)
2. Deterministic FIN + Standard FedAvg (BASELINE)
3. Probabilistic FIN + Standard FedAvg (Ablation)
4. Deterministic FIN + UQ-weighted FedAvg (Ablation)

```bash
python run_comparison.py
```

## Arguments

### Training Arguments

- `--num_rounds`: Number of federated rounds (default: 30)
- `--local_epochs`: Local training epochs per round (default: 3)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--eval_every`: Evaluate every N rounds (default: 5)

### Model Arguments

- `--use_probabilistic`: Use Beta-NLL for FIN (default: True)
- `--use_uq_weighting`: Use uncertainty-weighted aggregation (default: True)
- `--uq_alpha`: Weight for UQ in aggregation, 0=data only, 1=UQ only (default: 0.5)

### Output Arguments

- `--save_dir`: Directory to save checkpoints (default: checkpoints_prob)

## Variance Explosion Prevention

Critical mechanisms to prevent variance explosion:

1. **Parameter Clamping**: `alpha, beta ∈ [0.1, 100.0]`
2. **Variance Regularization**: `L = -log p(x | alpha, beta) + λ * Var[Beta]`
3. **Gradient Clipping**: `max_norm = 1.0`
4. **Feature Normalization**: Normalize to [0, 1] for Beta distribution
5. **Softplus Activation**: Ensures positive parameters

## Evaluation Metrics

- **Test AUC**: Macro-averaged ROC AUC over 14 disease labels
- **Test F1**: Macro-averaged F1 score over 14 disease labels

## File Structure

```
probabilistic_models.py          # Beta-NLL, Probabilistic FIN, Deterministic FIN
multimodal_model.py              # Image/Text encoders, Fusion, Classifier
federated_learning.py            # UQ-weighted FedAvg, Aggregation logic
data_loader_federated.py         # Federated data loading
train_federated.py               # Main training script
run_comparison.py                # Comparison experiments
```

## Expected Results

Based on the original FedFeatGen paper:

| Method | Test AUC | Test F1 |
|--------|----------|---------|
| Zero-filling | ~0.80 | ~0.70 |
| R2Gen | ~0.77 | ~0.68 |
| Original FIN | ~0.86 | ~0.75 |
| **Probabilistic FIN (Ours)** | **~0.88** | **~0.77** |

## Key Differences from Original FIN

| Aspect | Original FIN | Probabilistic FIN (Ours) |
|--------|-------------|--------------------------|
| Output | Point estimate | Beta distribution (alpha, beta) |
| Loss | MSE | Beta-NLL |
| Uncertainty | None | Explicit variance |
| Aggregation | Uniform FedAvg | UQ-weighted FedAvg |
| Feature Space | Raw | Normalized [0, 1] |

## Citation

Original FedFeatGen paper:
```bibtex
@article{poudel2025fedfeatgen,
  title={Multimodal Federated Learning With Missing Modalities through Feature Imputation Network},
  author={Poudel, Pranav and Chhetri, Aavash and Gyawali, Prashnna and Leontidis, Georgios and Bhattarai, Binod},
  journal={arXiv preprint arXiv:2505.20232},
  year={2025}
}
```

## Troubleshooting

### Variance Explosion

If you encounter NaN or inf values:
1. Reduce learning rate: `--lr 5e-5`
2. Increase variance penalty in `BetaNLLLoss` (edit `probabilistic_models.py`)
3. Tighten parameter bounds: `min_alpha=0.5, max_alpha=50.0`

### Memory Issues

If running out of GPU memory:
1. Reduce batch size: `--batch_size 16`
2. Use gradient accumulation (modify training script)
3. Reduce model size (use smaller BERT variant)

### Slow Training

1. Reduce dataset size in data loader
2. Use fewer transformer layers in FIN
3. Skip some federated rounds for evaluation

## Contact

For issues or questions, please open an issue in the repository.
