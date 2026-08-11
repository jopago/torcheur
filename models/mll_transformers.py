import torch
from torch import nn

from models.configs import Config, StateActionConfig


class FullProofTransformer(nn.Module):
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
        _, T = tokens.shape

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


class StateActionTransformer(nn.Module):
    def __init__(self, cfg: StateActionConfig):
        super().__init__()

        self.cfg = cfg

        # Encode tokens inside each formula
        self.token_embedding = nn.Embedding(
            cfg.vocab_size,
            cfg.embedding_dim,
            padding_idx=cfg.pad_token_id,
        )

        self.token_position_embedding = nn.Embedding(
            cfg.max_formula_len,
            cfg.embedding_dim,
        )

        formula_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.formula_transformer = nn.TransformerEncoder(
            formula_layer,
            num_layers=cfg.n_formula_layers,
        )

        # Encode relations between formulas in the sequent
        self.formula_position_embedding = nn.Embedding(
            cfg.max_n_formulas,
            cfg.embedding_dim,
        )

        sequent_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.sequent_transformer = nn.TransformerEncoder(
            sequent_layer,
            num_layers=cfg.n_sequent_layers,
        )

        self.norm = nn.LayerNorm(cfg.embedding_dim)

        # Which formula do we split?
        self.split_head = nn.Linear(cfg.embedding_dim, 1)

        # For each formula: LEFT or RIGHT
        self.side_head = nn.Linear(cfg.embedding_dim, 2)

    def forward(self, tokens):
        # tokens: [B, N, L]

        B, N, L = tokens.shape

        token_mask = tokens == self.cfg.pad_token_id
        formula_mask = token_mask.all(dim=-1)  # [B, N]

        # ---- Formula encoding ----

        x = self.token_embedding(tokens)

        positions = torch.arange(L, device=tokens.device)
        x = x + self.token_position_embedding(positions)[None, None, :, :]

        # [B*N, L, D]
        x = x.reshape(B * N, L, -1)
        flat_mask = token_mask.reshape(B * N, L)

        attention_mask = flat_mask.clone()
        fully_padded = attention_mask.all(dim=1)
        attention_mask[fully_padded, 0] = False

        x = self.formula_transformer(
            x,
            src_key_padding_mask=attention_mask,
        )

        # Pour le pooling, on utilise le vrai masque
        valid = (~flat_mask).unsqueeze(-1)

        formula_embeddings = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        # Les slots correspondant à des formules inexistantes valent exactement 0
        formula_embeddings[fully_padded] = 0.0

        formula_embeddings = formula_embeddings.reshape(B, N, -1)
        # ---- Sequent encoding ----

        formula_positions = torch.arange(N, device=tokens.device)

        x = (
            formula_embeddings
            + self.formula_position_embedding(formula_positions)[None, :, :]
        )

        x = self.sequent_transformer(
            x,
            src_key_padding_mask=formula_mask,
        )

        x = self.norm(x)

        # [B, N]
        split_logits = self.split_head(x).squeeze(-1)

        # don't allow selecting padding
        split_logits = split_logits.masked_fill(
            formula_mask,
            -1e6,
        )

        # [B, N, 2]
        side_logits = self.side_head(x)

        return split_logits, side_logits
