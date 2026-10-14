from pathlib import Path

import torch

from mll.parse import parse_normalized_line
from models.configs import StateActionConfig
from models.mll_transformers import SplitTransformer
from tokenizer import FormulaTokenizer

# Caps in number of dataset LINES (not proofs) to keep encoding in RAM.
TRAIN_N = 800_000
VAL_N = 200_000


def read_n_lines(path: str, n: int) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if len(lines) >= n:
                break
    return lines


lines = read_n_lines("mll_positional_train.txt", TRAIN_N)
val_lines = read_n_lines("mll_positional_val.txt", VAL_N)
print(f"Using {len(lines)} train lines, {len(val_lines)} val lines")

tokenizer_path = Path("tokenizer_mll_state_action.json")
if tokenizer_path.exists():
    tokenizer = FormulaTokenizer.load(str(tokenizer_path))
else:
    tokenizer_training_text = [
        "".join(parse_normalized_line(line)[0]) for line in lines[:10_000]
    ]
    tokenizer = FormulaTokenizer()
    tokenizer.train("\n".join(tokenizer_training_text), n_merges=20)
    tokenizer.save(str(tokenizer_path))

vocab_size = len(tokenizer.vocab)
print("Vocab size:", vocab_size)

config = StateActionConfig(
    vocab_size=vocab_size + 1,
    embedding_dim=128,
    n_heads=4,
    ff_dim=512,
    max_formula_len=64,
    max_n_formulas=32,
    n_formula_layers=3,
    n_sequent_layers=3,
    pad_token_id=vocab_size,
)
device = "cpu"
model = SplitTransformer(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=3000,
    eta_min=1e-5,
)

batch_size = 64


def encode_formula_labels(dataset: list[str], config: StateActionConfig):
    encoded = []

    for line in dataset:
        formulas, labels = parse_normalized_line(line)

        formula_tokens = [
            torch.tensor(
                tokenizer.encode(formula)[: config.max_formula_len],
                dtype=torch.long,
            )
            for formula in formulas[: config.max_n_formulas]
        ]

        labels = labels[: config.max_n_formulas]

        if len(labels) > 1:
            encoded.append((formula_tokens, labels))

    return encoded


train_path = Path(f"encoded_train_{TRAIN_N}.pt")
val_path = Path(f"encoded_val_{VAL_N}.pt")
if train_path.exists() and val_path.exists() and train_path.stat().st_size > 0:
    print("Loading encoded sets from disk...")
    encoded_train = torch.load(train_path)
    encoded_val = torch.load(val_path)
else:
    print("Encoding training and val set...")
    encoded_train = encode_formula_labels(lines, config)
    encoded_val = encode_formula_labels(val_lines, config)
    torch.save(encoded_train, train_path)
    torch.save(encoded_val, val_path)

print("training split model...")


def make_xy(batch):
    B = len(batch)

    N = max(len(formulas) for formulas, _ in batch)
    L = max(len(formula) for formulas, _ in batch for formula in formulas)

    N = min(N, config.max_n_formulas)
    L = min(L, config.max_formula_len)

    x = torch.full(
        (B, N, L),
        config.pad_token_id,
        dtype=torch.long,
    )
    split_target = torch.empty(B, dtype=torch.long)

    for b, (formulas, labels) in enumerate(batch):
        formulas = formulas[:N]
        labels = labels[:N]

        for i, formula in enumerate(formulas):
            length = min(len(formula), L)
            x[b, i, :length] = formula[:length]

        labels_t = torch.tensor(labels, dtype=torch.long)
        split_target[b] = (labels_t == 2).nonzero(as_tuple=True)[0].item()

    return x.to(device), split_target.to(device)


eval_size = 4096
eval_batch_size = 128
eval_indices = torch.randperm(len(encoded_val))[:eval_size]
fixed_eval = [encoded_val[i] for i in eval_indices]


@torch.no_grad()
def evaluate():
    model.eval()
    split_correct = 0
    split_total = 0

    for start in range(0, len(fixed_eval), eval_batch_size):
        batch = fixed_eval[start : start + eval_batch_size]
        x, split_target = make_xy(batch)
        split_logits = model(x)
        split_pred = split_logits.argmax(dim=-1)
        split_correct += (split_pred == split_target).sum().item()
        split_total += split_target.numel()

    model.train()
    return split_correct / split_total


for step in range(501):
    indices = torch.randint(len(encoded_train), (batch_size,))
    batch = [encoded_train[i] for i in indices]

    x, split_target = make_xy(batch)
    split_logits = model(x)
    loss = torch.nn.functional.cross_entropy(split_logits, split_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 10 == 0:
        print("loss = ", loss.item())

    if step % 30 == 0:
        print(step, "loss =", loss.item(), "eval split acc =", evaluate())

    if step % 500 == 0:
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/mll_split_transformer_{step}.pt")
