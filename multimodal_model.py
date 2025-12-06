"""
Multimodal Model with Uncertainty Quantification
Includes image encoder, text encoder, fusion, and classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, BertTokenizer
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ImageEncoder(nn.Module):
    """ResNet-50 based image encoder"""

    def __init__(self, output_dim: int = 256, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)

        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Project to output_dim and L2-normalize (as per paper)
        self.projection = nn.Linear(2048, output_dim)

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images [batch_size, 3, 224, 224]

        Returns:
            features: L2-normalized features [batch_size, output_dim]
        """

        features = self.features(x)  # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 2048]

        features = self.projection(features)  # [B, output_dim]

        # L2-normalize (as per paper)
        features = F.normalize(features, p=2, dim=-1)

        return features


class TextEncoder(nn.Module):
    """BERT-base text encoder"""

    def __init__(self, output_dim: int = 256):
        super().__init__()

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Project to output_dim and L2-normalize (as per paper)
        self.projection = nn.Linear(768, output_dim)  # BERT hidden size = 768

        self.output_dim = output_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            features: L2-normalized features [batch_size, output_dim]
        """

        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        # Project
        features = self.projection(cls_output)  # [B, output_dim]

        # L2-normalize (as per paper)
        features = F.normalize(features, p=2, dim=-1)

        return features


class MultimodalFusionClassifier(nn.Module):
    """
    Multimodal fusion + classifier with uncertainty
    Fusion: Simple concatenation (as per paper)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 14,
        use_uncertainty: bool = True
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty

        # Fusion: concatenate image + text features
        fused_dim = 2 * feature_dim

        if use_uncertainty:
            # Output logits + uncertainty parameters (alpha, beta)
            self.fc_logits = nn.Linear(fused_dim, num_classes)
            self.fc_alpha = nn.Linear(fused_dim, num_classes)
            self.fc_beta = nn.Linear(fused_dim, num_classes)
        else:
            # Standard classifier
            self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_features: [batch_size, feature_dim]
            text_features: [batch_size, feature_dim]

        Returns:
            Dictionary with logits and optionally uncertainty parameters
        """

        # Concatenate features
        fused = torch.cat([image_features, text_features], dim=-1)  # [B, 2*feature_dim]

        if self.use_uncertainty:
            logits = self.fc_logits(fused)
            alpha = F.softplus(self.fc_alpha(fused)) + 0.1
            beta = F.softplus(self.fc_beta(fused)) + 0.1

            return {
                'logits': logits,
                'alpha': alpha,
                'beta': beta
            }
        else:
            logits = self.classifier(fused)
            return {
                'logits': logits
            }


class MultimodalModel(nn.Module):
    """
    Complete multimodal model
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 14,
        use_uncertainty: bool = True,
        pretrained: bool = True
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty

        # Encoders
        self.image_encoder = ImageEncoder(output_dim=feature_dim, pretrained=pretrained)
        self.text_encoder = TextEncoder(output_dim=feature_dim)

        # Fusion + Classifier
        self.fusion_classifier = MultimodalFusionClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            use_uncertainty=use_uncertainty
        )

        # Tokenizer (needed for text encoding)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to features"""
        return self.image_encoder(images)

    def encode_text(self, texts: list) -> torch.Tensor:
        """Encode texts to features"""

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(next(self.text_encoder.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(self.text_encoder.parameters()).device)

        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        images: torch.Tensor,
        texts: list
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass

        Args:
            images: [batch_size, 3, 224, 224]
            texts: List of text strings

        Returns:
            image_features: [batch_size, feature_dim]
            text_features: [batch_size, feature_dim]
            outputs: Dictionary with logits and uncertainty (if enabled)
        """

        # Encode modalities
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        # Fusion + Classification
        outputs = self.fusion_classifier(image_features, text_features)

        return image_features, text_features, outputs


def get_model(
    feature_dim: int = 256,
    num_classes: int = 14,
    use_uncertainty: bool = True,
    pretrained: bool = True
) -> MultimodalModel:
    """Factory function to create model"""

    model = MultimodalModel(
        feature_dim=feature_dim,
        num_classes=num_classes,
        use_uncertainty=use_uncertainty,
        pretrained=pretrained
    )

    return model


if __name__ == "__main__":
    # Test model
    print("Testing Multimodal Model...")

    model = get_model(use_uncertainty=True)

    # Random inputs
    images = torch.randn(2, 3, 224, 224)
    texts = ["This is a test report.", "Another medical report here."]

    # Forward pass
    img_feat, txt_feat, outputs = model(images, texts)

    print(f"Image features: {img_feat.shape}")
    print(f"Text features: {txt_feat.shape}")
    print(f"Logits: {outputs['logits'].shape}")

    if 'alpha' in outputs:
        print(f"Alpha (uncertainty): {outputs['alpha'].shape}")
        print(f"Beta (uncertainty): {outputs['beta'].shape}")

    print("\nModel created successfully!")
