from dataclasses import dataclass
import torch.nn as nn

@dataclass
class Config:
    hidden_dim: int = 256
    embedding_dim: int = 128
    n_layers: int = 2
    vocab_size: int = 132

class MLLNetwork(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.gru = nn.GRU(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
        )
        self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, x):
        # x: [B, T]
        x = self.embedding(x)  # [B, T, d_model]
        x, _ = self.gru(x)  # [B, T, hidden_size]
        logits = self.head(x)  # [B, T, vocab_size]
        return logits