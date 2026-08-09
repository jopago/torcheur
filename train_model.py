import torch

from models.mll_network import Config
from models.mll_transformer import MLLTransformer
from tokenizer import FormulaTokenizer

with open("mll_positional.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

test_lines = lines[130_000:200_000]
lines = lines[:130_000]

# Train the tokenizer on 10k lines
tokenizer_training_text = lines[:10_000]

# Train tokenizer
tokenizer = FormulaTokenizer()
# tokenizer.train("\n".join(tokenizer_training_text), n_merges=30)
# tokenizer.save("tokenizer_mll.json")

tokenizer = FormulaTokenizer.load("tokenizer_mll.json")

vocab_size = len(tokenizer.vocab)
print("Vocab size:", vocab_size)
print("Vocab: ", tokenizer.vocab)

context_size = 250
config = Config(vocab_size=vocab_size,
                max_seq_len=context_size,
                n_layers=4,
                embedding_dim=256,
                n_heads=4,
                ff_dim=512)
device = "mps"
model = MLLTransformer(config).to(device)
# model = torch.compile(raw_model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10_000,
    eta_min=1e-5,
)

batch_size = 64

# encoded = [
#    torch.tensor(tokenizer.encode(line)[:context_size + 1], dtype=torch.long)
#    for line in lines
# ]
# encoded_test = [
#    torch.tensor(tokenizer.encode(line)[:context_size + 1], dtype=torch.long)
#    for line in test_lines
# ]
# torch.save(encoded, f"encoded_training_set.pt")
# torch.save(encoded_test, f"encoded_test_set.pt")

encoded = torch.load("encoded_training_set.pt")
encoded_test = torch.load("encoded_test_set.pt")

print("training...")


def make_xy(seqs):
    max_len = max(len(s) for s in seqs)

    x = torch.zeros(batch_size, max_len - 1, dtype=torch.long)
    y = torch.full((batch_size, max_len - 1), -1, dtype=torch.long)

    # x_t = seq_t, y_t = seq_{t+1}
    for i, s in enumerate(seqs):
        x[i, :len(s) - 1] = s[:-1]
        y[i, :len(s) - 1] = s[1:]

    x = x.to(device)
    y = y.to(device)

    return x, y


for step in range(20_000):
    seqs = [encoded[i] for i in torch.randint(len(encoded), (batch_size,))]

    x, y = make_xy(seqs)
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
    scheduler.step()

    # test acc
    with torch.no_grad():
        seqs_test = [encoded_test[i] for i in torch.randint(len(encoded_test), (batch_size,))]
        x_test, y_test = make_xy(seqs_test)
        logits = model(x_test)
        pred = logits.argmax(dim=-1)
        mask = y_test != -1
        accuracy_test = (pred[mask] == y_test[mask]).float().mean()

    if step % 10 == 0:
        print(step, loss.item(), " train accuracy = ", accuracy.item(), " test accuracy = ", accuracy_test.item())
    if step % 500 == 0:
        torch.save(model.state_dict(), f"checkpoints/mll_transformer_{step}.pt")
