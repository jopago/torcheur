import torch
from torch import nn

from mll_network import Config, MLLNetwork
from tokenizer import FormulaTokenizer

with open("mll_positional.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

lines = lines[:100_000]
test_lines = lines[100_000:150_000]

# Train the tokenizer on 10k lines
tokenizer_training_text = lines[:10_000]

"""
# Train tokenizer
tokenizer = FormulaTokenizer()
tokenizer.train("\n".join(tokenizer_training_text), n_merges=100)
tokenizer.save("tokenzier_mll.json")
"""

tokenizer = FormulaTokenizer.load("tokenizer_mll.json")
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

    # x_t = seq_t, y_t = seq_{t+1}
    for i, s in enumerate(seqs):
        x[i, :len(s) - 1] = s[:-1]
        y[i, :len(s) - 1] = s[1:]

    x = x.to(device)
    y = y.to(device)

    logits = model(x)

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=-1,  # padding
    )

    pred = logits.argmax(dim=-1)
    mask = y != -1

    accuracy = (pred[mask] == y[mask]).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(step, loss.item(), " accuracy = ", accuracy.item())
    if step % 500 == 0:
        torch.save(model.state_dict(), f"mll_model_{step}.pt")
