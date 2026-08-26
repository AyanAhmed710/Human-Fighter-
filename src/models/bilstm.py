"""
Baseline model (protocol section 7: "build first"): BiLSTM + attention over
the per-frame feature vectors (xyz + bone + velocity + visibility, from
src.data.preprocess.to_feature_vector). No pretrained weights -- this is the
from-scratch sanity model that validates the full pipeline end-to-end before
the 2s-AGCN (which does use a pretrained checkpoint, hence addendum #4's
discriminative fine-tuning applies there, not here).
"""
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Learned additive attention over the time axis -- lets the model
    downweight noisy/low-visibility frames itself instead of a hard cutoff
    (consistent with addendum #2's soft-confidence-weighting decision)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        # x: (B, T, H)
        weights = torch.tanh(self.proj(x))
        weights = self.score(weights).squeeze(-1)         # (B, T)
        weights = torch.softmax(weights, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, H)
        return pooled, weights


class BiLSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = AttentionPool(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        out, _ = self.lstm(x)          # (B, T, 2*hidden_dim)
        pooled, attn_weights = self.attn(out)
        logits = self.classifier(pooled)
        return logits, attn_weights
