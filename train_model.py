import ast

import torch

from mll.parse import parse_line
from models.configs import StateActionConfig
from models.mll_transformers import StateActionTransformer
from tokenizer import FormulaTokenizer

with open("mll_positional.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]


test_lines = lines[200_000:300_000]
lines = lines[:200_000]

# Train the tokenizer on 10k formulas
tokenizer_training_text = ["".join(parse_line(line)[0]) for line in lines[:10_000]]

# Train tokenizer
tokenizer = FormulaTokenizer()
tokenizer.train("\n".join(tokenizer_training_text), n_merges=20)
tokenizer.save("tokenizer_mll_state_action.json")

# tokenizer = FormulaTokenizer.load("tokenizer_mll.json")

vocab_size = len(tokenizer.vocab)
print("Vocab size:", vocab_size)
print("Vocab: ", tokenizer.vocab)

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
model = StateActionTransformer(config).to(device)
# state = torch.load("checkpoints/mll_transformer_8000.pt")
# model.load_state_dict(state)

# model = torch.compile(raw_model)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=3000,
    eta_min=1e-5,
)

batch_size = 64

print("Encoding training and test set...")


def encode_formula_labels(dataset: list[str], config: StateActionConfig):
    encoded = []

    for line in dataset:
        formulas, labels = parse_line(line)

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


# encoded_train = encode_formula_labels(lines, config)
# encoded_test = encode_formula_labels(test_lines, config)

# torch.save(encoded_train, "encoded_training_set.pt")
# torch.save(encoded_test, "encoded_test_set.pt")

encoded_train = torch.load("encoded_training_set.pt")
encoded_test = torch.load("encoded_test_set.pt")

print("training...")


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

    # 0 = RIGHT
    # 1 = LEFT
    # -1 = ignore (split formula + padding)
    side_target = torch.full(
        (B, N),
        -1,
        dtype=torch.long,
    )

    for b, (formulas, labels) in enumerate(batch):
        formulas = formulas[:N]
        labels = labels[:N]

        for i, formula in enumerate(formulas):
            length = min(len(formula), L)
            x[b, i, :length] = formula[:length]

        labels = torch.tensor(labels, dtype=torch.long)

        # Exactly one formula should have label 2
        split_target[b] = (labels == 2).nonzero(as_tuple=True)[0].item()

        # labels 0/1 already correspond to RIGHT/LEFT
        for i, label in enumerate(labels):
            if label != 2:
                side_target[b, i] = label

    return (
        x.to(device),
        split_target.to(device),
        side_target.to(device),
    )


eval_size = 4096
eval_batch_size = 128

eval_indices = torch.randperm(len(encoded_test))[:eval_size]
fixed_eval = [encoded_test[i] for i in eval_indices]


@torch.no_grad()
def evaluate():
    model.eval()

    split_correct = 0
    split_total = 0

    side_correct = 0
    side_total = 0

    for start in range(0, len(fixed_eval), eval_batch_size):
        batch = fixed_eval[start : start + eval_batch_size]

        x, split_target, side_target = make_xy(batch)
        split_logits, side_logits = model(x)

        # Split accuracy
        split_pred = split_logits.argmax(dim=-1)
        split_correct += (split_pred == split_target).sum().item()
        split_total += split_target.numel()

        # Side accuracy
        side_pred = side_logits.argmax(dim=-1)
        mask = side_target != -1

        side_correct += (side_pred[mask] == side_target[mask]).sum().item()
        side_total += mask.sum().item()

    model.train()

    return (
        split_correct / split_total,
        side_correct / side_total,
    )


for step in range(20_000):
    indices = torch.randint(len(encoded_train), (batch_size,))
    batch = [encoded_train[i] for i in indices]

    x, split_target, side_target = make_xy(batch)
    split_logits, side_logits = model(x)

    split_loss = torch.nn.functional.cross_entropy(
        split_logits,
        split_target,
    )

    side_loss = torch.nn.functional.cross_entropy(
        side_logits.reshape(-1, 2),
        side_target.reshape(-1),
        ignore_index=-1,
    )

    loss = split_loss + side_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 10 == 0:
        print("loss = ", loss.item())

    if step % 30 == 0:
        eval_split_acc, eval_side_acc = evaluate()
        print(
            step,
            "loss =",
            loss.item(),
            "eval split acc =",
            eval_split_acc,
            "eval side acc =",
            eval_side_acc,
        )
    if step % 500 == 0:
        torch.save(model.state_dict(), f"checkpoints/mll_state_transformer_{step}.pt")
