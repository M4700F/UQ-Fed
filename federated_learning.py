"""
Federated Learning Framework with Uncertainty-Weighted Aggregation
Implements UQ-weighted FedAvg
"""

import torch
import torch.nn as nn
import copy
from typing import List, Dict, Tuple, Optional
import numpy as np


def fedavg_aggregate(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_weights: Optional[List[float]] = None
) -> nn.Module:
    """
    Standard FedAvg aggregation

    Args:
        global_model: Global model to update
        client_models: List of client models
        client_weights: Optional weights for each client (default: uniform)

    Returns:
        Updated global model
    """

    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)

    # Normalize weights
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]

    # Get global model state dict and device
    global_dict = global_model.state_dict()
    device = next(iter(global_dict.values())).device

    # Aggregate parameters
    for key in global_dict.keys():
        # Skip non-float parameters (e.g., embeddings, position IDs)
        if not global_dict[key].dtype.is_floating_point:
            # For non-float parameters, just copy from first client
            global_dict[key] = client_models[0].state_dict()[key].clone().to(device)
            continue

        global_dict[key] = torch.zeros_like(global_dict[key])

        for client_model, weight in zip(client_models, client_weights):
            client_dict = client_model.state_dict()
            global_dict[key] += weight * client_dict[key].to(device)

    # Update global model
    global_model.load_state_dict(global_dict)

    return global_model


def uq_weighted_fedavg_aggregate(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_uncertainties: List[float],
    client_data_sizes: List[int],
    alpha: float = 0.5,
    temperature: float = 1.0
) -> nn.Module:
    """
    Uncertainty-weighted FedAvg aggregation with temperature scaling

    Weight = (data_size / total_data) * confidence
    where confidence = exp(-uncertainty / temperature) for softmax-like weighting

    Args:
        global_model: Global model to update
        client_models: List of client models
        client_uncertainties: List of mean uncertainties for each client
        client_data_sizes: List of data sizes for each client
        alpha: Trade-off between uncertainty and data size (0 = only data size, 1 = only uncertainty)
        temperature: Temperature for uncertainty scaling (higher = smoother weights)

    Returns:
        Updated global model
    """
    import math

    # Compute data-based weights
    total_data = sum(client_data_sizes)
    data_weights = [n / total_data for n in client_data_sizes]

    # Compute uncertainty-based confidence with temperature scaling
    # Using softmax-like weighting: confidence = exp(-uncertainty / temperature)
    # This gives smoother and more stable weights than 1/(1+unc)
    scaled_unc = [math.exp(-unc / temperature) for unc in client_uncertainties]
    
    # Normalize confidences
    total_conf = sum(scaled_unc)
    norm_confidences = [c / total_conf for c in scaled_unc]

    # Combine data weight and confidence
    client_weights = [
        (1 - alpha) * dw + alpha * conf
        for dw, conf in zip(data_weights, norm_confidences)
    ]

    # Normalize final weights
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]

    print(f"\nClient weights (data + UQ, temp={temperature}):")
    for i, (dw, conf, w, unc) in enumerate(
        zip(data_weights, norm_confidences, client_weights, client_uncertainties)
    ):
        print(f"  Client {i}: data_wt={dw:.3f}, conf={conf:.3f}, "
              f"final_wt={w:.3f}, unc={unc:.4f}")

    # Aggregate using weighted average
    return fedavg_aggregate(global_model, client_models, client_weights)


def aggregate_fins(
    global_fin_i2t: nn.Module,
    global_fin_t2i: nn.Module,
    client_fins_i2t: List[nn.Module],
    client_fins_t2i: List[nn.Module],
    multimodal_indices: List[int]
) -> Tuple[nn.Module, nn.Module]:
    """
    Aggregate FIN models (only from multimodal clients)

    Args:
        global_fin_i2t: Global image-to-text FIN
        global_fin_t2i: Global text-to-image FIN
        client_fins_i2t: List of client image-to-text FINs
        client_fins_t2i: List of client text-to-image FINs
        multimodal_indices: Indices of multimodal clients

    Returns:
        Updated global FIN models
    """

    if len(multimodal_indices) == 0:
        return global_fin_i2t, global_fin_t2i

    # Get only multimodal clients' FINs
    mm_fins_i2t = [client_fins_i2t[i] for i in multimodal_indices]
    mm_fins_t2i = [client_fins_t2i[i] for i in multimodal_indices]

    # Uniform aggregation for FINs
    num_mm_clients = len(multimodal_indices)
    weights = [1.0 / num_mm_clients] * num_mm_clients

    # Aggregate image-to-text FIN
    global_fin_i2t = fedavg_aggregate(global_fin_i2t, mm_fins_i2t, weights)

    # Aggregate text-to-image FIN
    global_fin_t2i = fedavg_aggregate(global_fin_t2i, mm_fins_t2i, weights)

    return global_fin_i2t, global_fin_t2i


class FederatedTrainer:
    """Handles federated training logic"""

    def __init__(
        self,
        global_model: nn.Module,
        global_fin_i2t: nn.Module,
        global_fin_t2i: nn.Module,
        num_clients: int,
        use_uq_weighting: bool = True,
        uq_alpha: float = 0.5,
        uq_temperature: float = 0.1,
        warmup_rounds: int = 5  # Warmup period for UQ weighting
    ):
        self.global_model = global_model
        self.global_fin_i2t = global_fin_i2t
        self.global_fin_t2i = global_fin_t2i
        self.num_clients = num_clients
        self.use_uq_weighting = use_uq_weighting
        self.uq_alpha = uq_alpha
        self.uq_temperature = uq_temperature
        self.warmup_rounds = warmup_rounds
        self.current_round = 0

        # Track client statistics
        self.client_uncertainties = [0.0] * num_clients
        self.client_data_sizes = [0] * num_clients

    def distribute_models(self) -> Tuple[List[nn.Module], List[nn.Module], List[nn.Module]]:
        """
        Distribute global models to all clients
        Uses state_dict copying for memory efficiency

        Returns:
            client_models: List of client models (copies of global)
            client_fins_i2t: List of client I2T FINs
            client_fins_t2i: List of client T2I FINs
        """
        import gc
        
        client_models = []
        client_fins_i2t = []
        client_fins_t2i = []
        
        # Get device
        device = next(self.global_model.parameters()).device

        for i in range(self.num_clients):
            # Create new model instances and load state dict (more memory efficient than deepcopy)
            # For main model
            client_model = type(self.global_model)(
                **{k: v for k, v in self.global_model.__dict__.items() 
                   if not k.startswith('_') and k in ['image_encoder', 'text_encoder', 'fusion', 'classifier'] == False}
            ) if hasattr(self.global_model, '__dict__') else copy.deepcopy(self.global_model)
            
            # Just use deepcopy but clear cache first
            if i > 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            client_models.append(copy.deepcopy(self.global_model))
            client_fins_i2t.append(copy.deepcopy(self.global_fin_i2t))
            client_fins_t2i.append(copy.deepcopy(self.global_fin_t2i))

        return client_models, client_fins_i2t, client_fins_t2i

    def aggregate_round(
        self,
        client_models: List[nn.Module],
        client_fins_i2t: List[nn.Module],
        client_fins_t2i: List[nn.Module],
        multimodal_indices: List[int]
    ):
        """
        Aggregate models after a federated round

        Args:
            client_models: List of trained client models
            client_fins_i2t: List of trained client I2T FINs
            client_fins_t2i: List of trained client T2I FINs
            multimodal_indices: Indices of multimodal clients
        """
        self.current_round += 1

        # Aggregate main model with warmup for UQ weighting
        if self.use_uq_weighting and self.current_round > self.warmup_rounds:
            # After warmup, use full UQ weighting
            effective_alpha = self.uq_alpha
            print(f"  [UQ] Using uncertainty weighting (alpha={effective_alpha:.2f})")
            self.global_model = uq_weighted_fedavg_aggregate(
                self.global_model,
                client_models,
                self.client_uncertainties,
                self.client_data_sizes,
                alpha=effective_alpha,
                temperature=self.uq_temperature
            )
        elif self.use_uq_weighting and self.current_round <= self.warmup_rounds:
            # During warmup, gradually increase UQ influence
            warmup_progress = self.current_round / self.warmup_rounds
            effective_alpha = self.uq_alpha * warmup_progress
            print(f"  [UQ] Warmup round {self.current_round}/{self.warmup_rounds}, alpha={effective_alpha:.2f}")
            self.global_model = uq_weighted_fedavg_aggregate(
                self.global_model,
                client_models,
                self.client_uncertainties,
                self.client_data_sizes,
                alpha=effective_alpha,
                temperature=self.uq_temperature
            )
        else:
            self.global_model = fedavg_aggregate(
                self.global_model,
                client_models,
                client_weights=None  # Uniform weighting
            )

        # Aggregate FIN models (only from multimodal clients)
        self.global_fin_i2t, self.global_fin_t2i = aggregate_fins(
            self.global_fin_i2t,
            self.global_fin_t2i,
            client_fins_i2t,
            client_fins_t2i,
            multimodal_indices
        )

    def update_client_stats(
        self,
        client_id: int,
        uncertainty: float,
        data_size: int
    ):
        """Update client statistics for UQ-weighted aggregation"""
        self.client_uncertainties[client_id] = uncertainty
        self.client_data_sizes[client_id] = data_size


if __name__ == "__main__":
    # Test aggregation
    print("Testing Federated Aggregation...")

    from multimodal_model import get_model
    from probabilistic_models import ProbabilisticFIN

    # Create models
    global_model = get_model()
    global_fin_i2t = ProbabilisticFIN()
    global_fin_t2i = ProbabilisticFIN()

    # Create trainer
    trainer = FederatedTrainer(
        global_model=global_model,
        global_fin_i2t=global_fin_i2t,
        global_fin_t2i=global_fin_t2i,
        num_clients=10,
        use_uq_weighting=True
    )

    # Distribute models
    client_models, client_fins_i2t, client_fins_t2i = trainer.distribute_models()

    print(f"Distributed {len(client_models)} models to clients")

    # Simulate client updates with different uncertainties
    for i in range(10):
        trainer.update_client_stats(
            client_id=i,
            uncertainty=np.random.uniform(0.01, 0.1),
            data_size=1000 + i * 100
        )

    # Aggregate
    multimodal_indices = [8, 9]  # Last 2 are multimodal (clients 8 and 9 have reports)
    trainer.aggregate_round(
        client_models,
        client_fins_i2t,
        client_fins_t2i,
        multimodal_indices
    )

    print("\nAggregation successful!")
