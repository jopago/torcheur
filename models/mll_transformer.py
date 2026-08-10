from torch import nn
import torch

from models.mll_network import Config


class MLLTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.embedding_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.embedding_dim)
        self.head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

    def forward(self, tokens):
        # tokens: [B, T]
        B, T = tokens.shape

        positions = torch.arange(T, device=tokens.device)

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)[None, :, :]
        )

        # Prevent token t from seeing future tokens
        causal_mask = torch.triu(
            torch.ones(T, T, device=tokens.device, dtype=torch.bool), diagonal=1
        )

        x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)
        logits = self.head(x)

        return logits
