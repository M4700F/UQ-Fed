"""
Probabilistic FIN with Input-Dependent Uncertainty (Fixed)

Key fixes:
1. Alpha/beta now depend on INPUT features, not just learned queries
2. Input features are pooled and combined with transformer output
3. Proper initialization to ensure variance in outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureNormalizer:
    """Normalizes features to [0,1] range for Beta distribution."""
    
    def __init__(self, momentum=0.1, eps=1e-5):
        self.running_min = None
        self.running_max = None
        self.momentum = momentum
        self.eps = eps
        
    def update_stats(self, features):
        """Update running statistics during training."""
        with torch.no_grad():
            batch_min = features.min(dim=0)[0]
            batch_max = features.max(dim=0)[0]
            
            if self.running_min is None:
                self.running_min = batch_min.clone()
                self.running_max = batch_max.clone()
            else:
                if self.running_min.device != batch_min.device:
                    self.running_min = self.running_min.to(batch_min.device)
                    self.running_max = self.running_max.to(batch_max.device)
                    
                self.running_min = (1 - self.momentum) * self.running_min + self.momentum * batch_min
                self.running_max = (1 - self.momentum) * self.running_max + self.momentum * batch_max
    
    def normalize(self, features):
        """Normalize features to [0,1] range."""
        if self.running_min is None:
            feat_min = features.min(dim=0, keepdim=True)[0]
            feat_max = features.max(dim=0, keepdim=True)[0]
        else:
            if self.running_min.device != features.device:
                self.running_min = self.running_min.to(features.device)
                self.running_max = self.running_max.to(features.device)
            feat_min = self.running_min.unsqueeze(0)
            feat_max = self.running_max.unsqueeze(0)
        
        normalized = (features - feat_min) / (feat_max - feat_min + self.eps)
        return torch.clamp(normalized, self.eps, 1 - self.eps)
    
    def denormalize(self, normalized_features):
        """Convert normalized features back to original scale."""
        if self.running_min is None:
            return normalized_features
            
        if self.running_min.device != normalized_features.device:
            self.running_min = self.running_min.to(normalized_features.device)
            self.running_max = self.running_max.to(normalized_features.device)
            
        feat_min = self.running_min.unsqueeze(0)
        feat_max = self.running_max.unsqueeze(0)
        
        return normalized_features * (feat_max - feat_min + self.eps) + feat_min


class BetaNLLLoss(nn.Module):
    """Beta-NLL loss from the 111.pdf paper."""
    
    def __init__(self, beta_weight=1.0, mse_weight=0.5):
        super().__init__()
        self.beta_weight = beta_weight
        self.mse_weight = mse_weight
        
    def forward(self, alpha, beta, target):
        eps = 1e-6
        target = torch.clamp(target, eps, 1 - eps)
        
        log_beta_fn = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        nll = log_beta_fn - (alpha - 1) * torch.log(target) - (beta - 1) * torch.log(1 - target)
        
        mean_pred = alpha / (alpha + beta)
        mse = F.mse_loss(mean_pred, target, reduction='none')
        
        loss = self.beta_weight * nll + self.mse_weight * mse
        
        return loss.mean()


class InputDependentUncertaintyHead(nn.Module):
    """Produces input-dependent alpha/beta parameters."""
    
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=256):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.alpha_head, self.beta_head]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        nn.init.constant_(self.alpha_head[-1].bias, 0.5)
        nn.init.constant_(self.beta_head[-1].bias, 0.5)
        
    def forward(self, transformer_output, input_features):
        input_cond = self.input_proj(input_features)
        combined = torch.cat([transformer_output, input_cond], dim=-1)
        
        alpha = F.softplus(self.alpha_head(combined)) + 1.01
        beta = F.softplus(self.beta_head(combined)) + 1.01
        
        return alpha, beta


class ProbabilisticFIN(nn.Module):
    """Probabilistic FIN with INPUT-DEPENDENT uncertainty."""
    
    def __init__(self, input_dim=256, output_dim=256, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.uncertainty_head = InputDependentUncertaintyHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        self.normalizer = FeatureNormalizer()
        
    def forward(self, x, target_features=None, return_normalized=True, return_samples=False, num_samples=1):
        """
        Forward pass with input-dependent uncertainty.
        
        Args:
            x: Input features [B, input_dim]
            target_features: Target features for normalizer update (training only)
            return_normalized: If True, return normalized mean; if False, return denormalized
            return_samples: If True, return samples from the distribution
            num_samples: Number of samples to draw
        """
        batch_size = x.shape[0]
        original_input = x
        
        # Update normalizer statistics during training
        if target_features is not None:
            self.normalizer.update_stats(target_features)
        
        x_proj = self.input_proj(x).unsqueeze(1)
        query = self.query_token.expand(batch_size, -1, -1)
        sequence = torch.cat([query, x_proj], dim=1)
        sequence = sequence + self.pos_embedding
        
        encoded = self.transformer(sequence)
        transformer_output = encoded[:, 0, :]
        
        alpha, beta = self.uncertainty_head(transformer_output, original_input)
        mean = alpha / (alpha + beta)
        
        # Return normalized or denormalized features
        if return_normalized:
            output_features = mean
        else:
            output_features = self.normalizer.denormalize(mean)
        
        if return_samples:
            dist = torch.distributions.Beta(alpha, beta)
            samples = dist.rsample((num_samples,))
            return mean, alpha, beta, samples
        
        return alpha, beta, output_features
    
    def get_uncertainty(self, alpha, beta):
        """Compute variance of Beta distribution."""
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        return variance
    
    def impute(self, x, use_mean=True):
        _, _, features = self.forward(x, return_normalized=False)
        return features


class DeterministicFIN(nn.Module):
    """Deterministic FIN (baseline)."""
    
    def __init__(self, input_dim=256, output_dim=256, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x_proj = self.input_proj(x).unsqueeze(1)
        query = self.query_token.expand(batch_size, -1, -1)
        sequence = torch.cat([query, x_proj], dim=1)
        sequence = sequence + self.pos_embedding
        
        encoded = self.transformer(sequence)
        output = self.output_proj(encoded[:, 0, :])
        
        return output
    
    def impute(self, x):
        return self.forward(x)


def test_input_dependent_uncertainty():
    """Test that uncertainty varies with input."""
    print("Testing input-dependent uncertainty...")
    
    model = ProbabilisticFIN(input_dim=256, output_dim=256)
    model.eval()
    
    torch.manual_seed(42)
    x1 = torch.randn(1, 256)
    x2 = torch.randn(1, 256) * 2
    x3 = torch.zeros(1, 256)
    x4 = torch.ones(1, 256)
    
    with torch.no_grad():
        alpha1, beta1, _ = model(x1)
        alpha2, beta2, _ = model(x2)
        alpha3, beta3, _ = model(x3)
        alpha4, beta4, _ = model(x4)
        
        var1 = model.get_uncertainty(alpha1, beta1).mean().item()
        var2 = model.get_uncertainty(alpha2, beta2).mean().item()
        var3 = model.get_uncertainty(alpha3, beta3).mean().item()
        var4 = model.get_uncertainty(alpha4, beta4).mean().item()
    
    print(f"  Input 1 (random):  uncertainty = {var1:.6f}")
    print(f"  Input 2 (scaled):  uncertainty = {var2:.6f}")
    print(f"  Input 3 (zeros):   uncertainty = {var3:.6f}")
    print(f"  Input 4 (ones):    uncertainty = {var4:.6f}")
    
    uncertainties = [var1, var2, var3, var4]
    if len(set([f"{u:.6f}" for u in uncertainties])) > 1:
        print("SUCCESS: Uncertainties vary across inputs!")
    else:
        print("WARNING: Uncertainties are still constant!")
    
    batch_inputs = torch.randn(32, 256)
    with torch.no_grad():
        alpha_batch, beta_batch, _ = model(batch_inputs)
        var_batch = model.get_uncertainty(alpha_batch, beta_batch).mean(dim=1)
    
    print(f"\n  Batch variance of uncertainty: {var_batch.std().item():.6f}")
    print(f"  Min uncertainty in batch: {var_batch.min().item():.6f}")
    print(f"  Max uncertainty in batch: {var_batch.max().item():.6f}")
    
    return True


def normalize_features(features, eps=1e-5):
    """Normalize features to [0,1] range for Beta distribution."""
    feat_min = features.min(dim=0, keepdim=True)[0]
    feat_max = features.max(dim=0, keepdim=True)[0]
    normalized = (features - feat_min) / (feat_max - feat_min + eps)
    return torch.clamp(normalized, eps, 1 - eps)


if __name__ == "__main__":
    test_input_dependent_uncertainty()
