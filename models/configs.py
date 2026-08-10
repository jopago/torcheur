from dataclasses import dataclass


@dataclass
class Config:
    hidden_dim: int = 256
    embedding_dim: int = 32
    n_layers: int = 3
    vocab_size: int = 186
    n_heads: int = 4
    ff_dim: int = 256
    max_seq_len: int = 128

@dataclass
class StateActionConfig:
    vocab_size: int

    embedding_dim: int = 128
    n_heads: int = 4
    ff_dim: int = 512

    max_formula_len: int = 64
    max_n_formulas: int = 32

    n_formula_layers: int = 2
    n_sequent_layers: int = 2

    pad_token_id: int = 0