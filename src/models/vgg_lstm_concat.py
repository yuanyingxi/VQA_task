"""
VGG16 + LSTM + Concat baseline model for VizWiz-VQA.

Architecture:
  - Image: VGG16 (fc7 features, 4096-d), conv blocks frozen
  - Question: Embedding(300) + LSTM(1024), last hidden state
  - Fusion: Concat(4096+1024=5120) → Linear(5120, 1024) + ReLU + Dropout → Linear(1024, num_answers)
"""
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights

from src.configs.vgg_lstm_concat_config import (
    IMG_FEAT_DIM, EMBED_DIM, LSTM_HIDDEN,
    FUSION_DIM, MLP_HIDDEN, DROPOUT, FREEZE_CONV,
)

logger = logging.getLogger(__name__)


def _vgg16_weights_available() -> bool:
    """Check if VGG16 ImageNet weights are already cached on disk."""
    import torchvision
    hub_dir = os.path.expanduser(os.path.join("~", ".cache", "torch", "hub", "checkpoints"))
    if not os.path.isdir(hub_dir):
        return False
    for fname in os.listdir(hub_dir):
        if fname.startswith("vgg16-") and fname.endswith(".pth") and "partial" not in fname:
            return True
    return False


class VGG16Encoder(nn.Module):
    """
    VGG16 feature extractor: returns fc7 (4096-d) features.

    Conv layers are frozen by default; only the classifier is trained.
    """

    def __init__(self, freeze_conv: bool = FREEZE_CONV) -> None:
        super().__init__()
        if _vgg16_weights_available():
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            logger.info("VGG16 loaded with ImageNet pretrained weights.")
        else:
            logger.warning(
                "VGG16 ImageNet weights not found in cache. "
                "Using random initialization (download blocked/slow). "
                "To use pretrained weights, run once with network access: "
                "import torchvision; torchvision.models.vgg16(weights='IMAGENET1K_V1')"
            )
            vgg = models.vgg16(weights=None)

        # Feature extractor (conv layers)
        self.features = vgg.features

        # Classifier up to fc7 (classifier[0]=Linear, [1]=ReLU, [2]=Dropout, [3]=Linear=fc7)
        # Actually VGG classifier: Linear(25088, 4096) -> ReLU -> Dropout -> Linear(4096, 4096) -> ReLU -> Dropout -> Linear(4096, 1000)
        # fc7 is the output after classifier[4] (the second ReLU after the second Linear)
        # Let's take the first 5 layers: Linear, ReLU, Dropout, Linear, ReLU → output 4096 fc7
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:5])

        # Freeze conv layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        self.output_dim = IMG_FEAT_DIM  # 4096

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract 4096-d fc7 features.

        Args:
            images: (B, 3, 224, 224) normalized image tensor.

        Returns:
            features: (B, 4096) float tensor.
        """
        x = self.features(images)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


class LSTMEncoder(nn.Module):
    """
    LSTM question encoder: returns last hidden state (1024-d).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = LSTM_HIDDEN,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.hidden_dim = hidden_dim

    def forward(
        self,
        question_indices: torch.Tensor,
        question_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode question into a 1024-d vector.

        Args:
            question_indices: (B, L) long tensor, padded indices.
            question_lengths: (B,) int tensor, original lengths before padding.

        Returns:
            last_hidden: (B, 1024) float tensor, last layer's final hidden state.
        """
        emb = self.embedding(question_indices)  # (B, L, embed_dim)

        # Pack padded sequence for efficient LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, question_lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)  # h_n: (1, B, hidden_dim)
        last_hidden = h_n[-1]  # (B, hidden_dim)
        return last_hidden


class VGG_LSTM_Concat(nn.Module):
    """
    VGG16 + LSTM + Concat baseline for VQA.

    Image features (fc7, 4096) and question features (LSTM last, 1024)
    are concatenated → MLP → answer logits.
    """

    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        freeze_conv: bool = FREEZE_CONV,
        embed_dim: int = EMBED_DIM,
        lstm_hidden: int = LSTM_HIDDEN,
        fusion_dim: int = FUSION_DIM,
        mlp_hidden: int = MLP_HIDDEN,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.image_encoder = VGG16Encoder(freeze_conv=freeze_conv)
        self.text_encoder = LSTMEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=lstm_hidden,
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_answers),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layers with small values."""
        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        images: torch.Tensor,
        question_indices: torch.Tensor,
        question_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images:            (B, 3, 224, 224)
            question_indices:  (B, L) padded token ids
            question_lengths:  (B,)  original lengths

        Returns:
            logits: (B, num_answers) classification logits.
        """
        img_feat = self.image_encoder(images)           # (B, 4096)
        qst_feat = self.text_encoder(question_indices, question_lengths)  # (B, 1024)
        fusion_in = torch.cat([img_feat, qst_feat], dim=1)  # (B, 5120)
        logits = self.fusion(fusion_in)
        return logits
