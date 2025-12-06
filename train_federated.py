"""
Main training script for probabilistic FedFeatGen with Beta-NLL
Uses Beta distribution for bounded feature imputation with uncertainty
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from typing import Dict, List, Tuple

from data_loader_federated import create_federated_dataloaders, create_test_dataloader
from multimodal_model import get_model
from probabilistic_models import ProbabilisticFIN, DeterministicFIN, BetaNLLLoss, normalize_features
from federated_learning import FederatedTrainer


class LocalTrainer:
    """Handles local training at each client"""

    def __init__(
        self,
        client_id: int,
        is_multimodal: bool,
        model: nn.Module,
        fin_i2t: nn.Module,
        fin_t2i: nn.Module,
        use_probabilistic: bool = True,
        device: str = 'cuda'
    ):
        self.client_id = client_id
        self.is_multimodal = is_multimodal
        self.model = model.to(device)
        self.fin_i2t = fin_i2t.to(device)
        self.fin_t2i = fin_t2i.to(device)
        self.use_probabilistic = use_probabilistic
        self.device = device

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        if use_probabilistic:
            self.beta_nll_loss = BetaNLLLoss()  # Beta NLL for bounded uncertainty
        self.mse_loss = nn.MSELoss()

        # Track uncertainty
        self.mean_uncertainty = 0.0

    def train_multimodal_client(
        self,
        dataloader: DataLoader,
        num_epochs: int = 3,
        lr: float = 1e-4
    ) -> float:
        """
        Train multimodal client:
        1. Train main model on local data
        2. Train FIN networks

        Returns:
            mean_uncertainty: Average uncertainty for this client
        """

        # Optimizers
        model_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        fin_optimizer = optim.Adam(
            list(self.fin_i2t.parameters()) + list(self.fin_t2i.parameters()),
            lr=lr
        )

        self.model.train()
        self.fin_i2t.train()
        self.fin_t2i.train()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            for batch in dataloader:
                images = batch['image'].to(self.device)
                texts = batch['text']
                labels = batch['labels'].to(self.device)

                # === Step 1: Train main model ===
                model_optimizer.zero_grad()

                img_feat, txt_feat, outputs = self.model(images, texts)
                logits = outputs['logits']

                # Classification loss
                cls_loss = self.bce_loss(logits, labels)

                # Uncertainty loss (if using probabilistic)
                if self.use_probabilistic and 'alpha' in outputs:
                    alpha = outputs['alpha']
                    beta = outputs['beta']

                    # Beta-NLL loss on predictions (optional)
                    # For simplicity, we primarily use it on FIN
                    uncertainty_loss = 0.0
                else:
                    uncertainty_loss = 0.0

                loss = cls_loss + uncertainty_loss
                loss.backward()

                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                model_optimizer.step()

                # === Step 2: Train FIN networks ===
                fin_optimizer.zero_grad()

                # Get fresh features (detach to not affect main model)
                with torch.no_grad():
                    img_feat_target = self.model.encode_image(images)
                    txt_feat_target = self.model.encode_text(texts)

                # Train I2T FIN (image -> text)
                if self.use_probabilistic:
                    # Pass target_features to update normalizer, get normalized output for loss
                    alpha_t, beta_t, _ = self.fin_i2t(img_feat_target, target_features=txt_feat_target, return_normalized=True)
                    # Normalize target for Beta-NLL loss
                    txt_feat_norm = normalize_features(txt_feat_target)
                    loss_i2t = self.beta_nll_loss(alpha_t, beta_t, txt_feat_norm)
                else:
                    pred_t = self.fin_i2t(img_feat_target)
                    loss_i2t = self.mse_loss(pred_t, txt_feat_target)

                # Train T2I FIN (text -> image)
                if self.use_probabilistic:
                    alpha_i, beta_i, _ = self.fin_t2i(txt_feat_target, target_features=img_feat_target, return_normalized=True)
                    img_feat_norm = normalize_features(img_feat_target)
                    loss_t2i = self.beta_nll_loss(alpha_i, beta_i, img_feat_norm)
                else:
                    pred_i = self.fin_t2i(txt_feat_target)
                    loss_t2i = self.mse_loss(pred_i, img_feat_target)

                fin_loss = loss_i2t + loss_t2i
                fin_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.fin_i2t.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.fin_t2i.parameters(), max_norm=1.0)

                fin_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Track uncertainty
                if self.use_probabilistic:
                    with torch.no_grad():
                        unc_t = self.fin_i2t.get_uncertainty(alpha_t, beta_t).mean().item()
                        unc_i = self.fin_t2i.get_uncertainty(alpha_i, beta_i).mean().item()
                        self.mean_uncertainty = (unc_t + unc_i) / 2

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return self.mean_uncertainty

    def train_unimodal_client(
        self,
        dataloader: DataLoader,
        missing_modality: str = 'text',  # 'text' or 'image'
        num_epochs: int = 3,
        lr: float = 1e-4
    ) -> float:
        """
        Train unimodal client:
        Use FIN to impute missing modality

        Returns:
            mean_uncertainty: Average uncertainty for this client
        """

        model_optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        self.fin_i2t.eval()  # FIN is only for inference
        self.fin_t2i.eval()

        total_loss = 0.0
        num_batches = 0
        uncertainties = []

        for epoch in range(num_epochs):
            for batch in dataloader:
                images = batch['image'].to(self.device)
                texts = batch['text']  # May be empty
                labels = batch['labels'].to(self.device)

                model_optimizer.zero_grad()

                if missing_modality == 'text':
                    # Have image, impute text
                    with torch.no_grad():
                        img_feat = self.model.encode_image(images)

                        if self.use_probabilistic:
                            # Get denormalized features for classifier (return_normalized=False)
                            alpha_t, beta_t, txt_feat = self.fin_i2t(img_feat, return_normalized=False)
                            # Track uncertainty
                            unc = self.fin_i2t.get_uncertainty(alpha_t, beta_t).mean().item()
                            uncertainties.append(unc)
                        else:
                            txt_feat = self.fin_i2t(img_feat)

                    # Get image features again for gradient
                    img_feat = self.model.encode_image(images)

                else:
                    # Have text, impute image
                    with torch.no_grad():
                        txt_feat = self.model.encode_text(texts)

                        if self.use_probabilistic:
                            alpha_i, beta_i, img_feat = self.fin_t2i(txt_feat, return_normalized=False)
                            unc = self.fin_t2i.get_uncertainty(alpha_i, beta_i).mean().item()
                            uncertainties.append(unc)
                        else:
                            img_feat = self.fin_t2i(txt_feat)

                    # Get text features again for gradient
                    txt_feat = self.model.encode_text(texts)

                # Classification with imputed features
                outputs = self.model.fusion_classifier(img_feat, txt_feat)
                logits = outputs['logits']

                loss = self.bce_loss(logits, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                model_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.mean_uncertainty = np.mean(uncertainties) if uncertainties else 0.0

        return self.mean_uncertainty


def evaluate_model(
    model: nn.Module,
    fin_i2t: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate model on test set

    Returns:
        Dictionary with AUC and F1 scores
    """

    model = model.to(device)
    fin_i2t = fin_i2t.to(device)
    model.eval()
    fin_i2t.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            texts = batch['text']
            labels = batch['labels'].to(device)

            # Forward pass
            img_feat, txt_feat, outputs = model(images, texts)
            logits = outputs['logits']

            # Sigmoid for probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute metrics
    # AUC (macro average)
    try:
        auc_scores = []
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:  # Skip if only one class
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                auc_scores.append(auc)
        macro_auc = np.mean(auc_scores) if auc_scores else 0.0
    except:
        macro_auc = 0.0

    # F1 (macro average)
    try:
        preds_binary = (all_preds > 0.5).astype(int)
        macro_f1 = f1_score(all_labels, preds_binary, average='macro', zero_division=0)
    except:
        macro_f1 = 0.0

    return {
        'auc': macro_auc,
        'f1': macro_f1
    }


def save_checkpoint(
    round_num: int,
    model: nn.Module,
    fin_i2t: nn.Module,
    fin_t2i: nn.Module,
    metrics: Dict,
    save_dir: str = 'checkpoints'
):
    """Save checkpoint"""

    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'round': round_num,
        'model_state': model.state_dict(),
        'fin_i2t_state': fin_i2t.state_dict(),
        'fin_t2i_state': fin_t2i.state_dict(),
        'metrics': metrics
    }

    path = os.path.join(save_dir, f'checkpoint_round_{round_num}.pth')
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def main(args):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create dataloaders
    # Based on actual data: clients 0-7 are unimodal (CheXpert), clients 8-9 are multimodal (NIH)
    print("\n=== Creating Federated Dataloaders ===")
    client_loaders, is_multimodal_list = create_federated_dataloaders(
        num_multimodal_clients=2,
        num_unimodal_clients=8,
        batch_size=args.batch_size
    )

    test_loader = create_test_dataloader(batch_size=args.batch_size)

    # Create global models
    print("\n=== Creating Models ===")
    global_model = get_model(use_uncertainty=args.use_probabilistic, pretrained=True)

    if args.use_probabilistic:
        global_fin_i2t = ProbabilisticFIN()
        global_fin_t2i = ProbabilisticFIN()
        print("Using Probabilistic FIN with Beta-NLL")
    else:
        global_fin_i2t = DeterministicFIN()
        global_fin_t2i = DeterministicFIN()
        print("Using Deterministic FIN with MSE")

    # Create federated trainer
    fed_trainer = FederatedTrainer(
        global_model=global_model,
        global_fin_i2t=global_fin_i2t,
        global_fin_t2i=global_fin_t2i,
        num_clients=10,
        use_uq_weighting=args.use_uq_weighting,
        uq_alpha=args.uq_alpha,
        uq_temperature=args.uq_temperature
    )

    # Training loop
    print(f"\n=== Starting Federated Training for {args.num_rounds} rounds ===")

    best_auc = 0.0
    results = []

    for round_num in range(args.num_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{args.num_rounds}")
        print(f"{'='*60}")

        # Distribute models to clients
        client_models, client_fins_i2t, client_fins_t2i = fed_trainer.distribute_models()

        # Train each client
        for client_id in range(10):
            is_multimodal = is_multimodal_list[client_id]

            print(f"\nClient {client_id} ({'Multimodal' if is_multimodal else 'Unimodal'}):")

            trainer = LocalTrainer(
                client_id=client_id,
                is_multimodal=is_multimodal,
                model=client_models[client_id],
                fin_i2t=client_fins_i2t[client_id],
                fin_t2i=client_fins_t2i[client_id],
                use_probabilistic=args.use_probabilistic,
                device=device
            )

            if is_multimodal:
                uncertainty = trainer.train_multimodal_client(
                    client_loaders[client_id],
                    num_epochs=args.local_epochs,
                    lr=args.lr
                )
            else:
                uncertainty = trainer.train_unimodal_client(
                    client_loaders[client_id],
                    missing_modality='text',  # Assume image-only clients
                    num_epochs=args.local_epochs,
                    lr=args.lr
                )

            # Update client stats
            data_size = len(client_loaders[client_id].dataset)
            fed_trainer.update_client_stats(client_id, uncertainty, data_size)

            print(f"  Uncertainty: {uncertainty:.6f}, Data size: {data_size}")

        # Aggregate models
        print("\n--- Aggregating models ---")
        multimodal_indices = [i for i, is_mm in enumerate(is_multimodal_list) if is_mm]
        fed_trainer.aggregate_round(
            client_models,
            client_fins_i2t,
            client_fins_t2i,
            multimodal_indices
        )

        # Evaluate
        if (round_num + 1) % args.eval_every == 0:
            print("\n--- Evaluating on test set ---")
            metrics = evaluate_model(
                fed_trainer.global_model,
                fed_trainer.global_fin_i2t,
                test_loader,
                device=device
            )

            print(f"Test AUC: {metrics['auc']:.4f}")
            print(f"Test F1: {metrics['f1']:.4f}")

            results.append({
                'round': round_num + 1,
                'auc': metrics['auc'],
                'f1': metrics['f1']
            })
            
            # Save results to CSV after each evaluation (incremental saving)
            os.makedirs(args.save_dir, exist_ok=True)
            csv_path = os.path.join(args.save_dir, 'training_results.csv')
            with open(csv_path, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(['round', 'test_auc', 'test_f1'])
                for r in results:
                    writer.writerow([r['round'], f"{r['auc']:.4f}", f"{r['f1']:.4f}"])
            print(f"Results saved to {csv_path}")

            # Save best model
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                save_checkpoint(
                    round_num + 1,
                    fed_trainer.global_model,
                    fed_trainer.global_fin_i2t,
                    fed_trainer.global_fin_t2i,
                    metrics,
                    save_dir=args.save_dir
                )

    # Save final results
    results_path = os.path.join(args.save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training completed! Best AUC: {best_auc:.4f}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--num_rounds', type=int, default=30, help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=3, help='Local epochs per round')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    # Model settings
    parser.add_argument('--use_probabilistic', action='store_true', default=True,
                       help='Use probabilistic FIN with Beta-NLL')
    parser.add_argument('--use_uq_weighting', action='store_true', default=True,
                       help='Use uncertainty-weighted aggregation')
    parser.add_argument('--uq_alpha', type=float, default=0.5,
                       help='Weight for UQ in aggregation (0=data only, 1=UQ only)')
    parser.add_argument('--uq_temperature', type=float, default=0.1,
                       help='Temperature for UQ softmax scaling (higher = smoother weights)')

    # Evaluation
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every N rounds')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints_prob',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    main(args)
