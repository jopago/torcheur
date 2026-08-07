from dataclasses import dataclass

import torch
import torch.nn as nn

from tokenizer import FormulaTokenizer

@dataclass
class Config:
    hidden_dim: int = 256
    embedding_dim: int = 128
    n_layers: int = 3
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


with open("mll_positional.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
lines = lines[:50_000]
tokenizer_training_text = lines[:10_000]

tokenizer = FormulaTokenizer()
tokenizer.train("\n".join(tokenizer_training_text), n_merges=100)
vocab_size = len(tokenizer.vocab)
print("Vocab size:", vocab_size)

config = Config(vocab_size=vocab_size)
device = "mps"
raw_model = MLLNetwork(config).to(device)
model = torch.compile(raw_model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss = nn.CrossEntropyLoss(ignore_index=-1)

context_size = 128
batch_size = 64

encoded = [
    torch.tensor(tokenizer.encode(line)[:context_size + 1], dtype=torch.long)
    for line in lines
]

print("training...")
for step in range(10_000):
    seqs = [encoded[i] for i in torch.randint(len(encoded), (batch_size,))]

    max_len = max(len(s) for s in seqs)

    x = torch.zeros(batch_size, max_len - 1, dtype=torch.long)
    y = torch.full((batch_size, max_len - 1), -1, dtype=torch.long)

    for i, s in enumerate(seqs):
        x[i, :len(s) - 1] = s[:-1]
        y[i, :len(s) - 1] = s[1:]

    x = x.to(device)
    y = y.to(device)

    logits = model(x)

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=-1,
    )

    pred = logits.argmax(dim=-1)
    mask = y != -1

    accuracy = (pred[mask] == y[mask]).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(step, loss.item(), " accuracy = ", accuracy.item())
