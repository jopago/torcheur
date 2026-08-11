import torch

from mll.check import check_proof, check_syntax
from mll.parse import parse_line
from models.configs import Config, StateActionConfig
from models.mll_transformers import FullProofTransformer, StateActionTransformer
from tokenizer import FormulaTokenizer

with open("mll_positional.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
print(lines[266_722])

SEED = 42
# torch.manual_seed(SEED)
tokenizer = FormulaTokenizer.load("tokenizer_mll_state_action.json")
vocab_size = len(tokenizer.vocab)

device = "cpu"

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

model = StateActionTransformer(config)

state = torch.load("checkpoints/mll_state_transformer_500.pt")
model.load_state_dict(state)
model = model.to(device)

prompt = "⊢ f; ⊗(¬f,⊗(⊥,⅋(a,¬a))); ⊗(¬i,⅋(h,¬h)); ⊗(⅋(⊗(¬f,𝟙),⊗(⅋(¬f,f),f)),⊗(i,𝟙)) || [1, 2, 0, 0]."
print("\nPROMPT:")
print(prompt)

# Encode prompt

formulas, labels = parse_line(prompt)
formula_tokens = [
    torch.tensor(
        tokenizer.encode(formula)[: config.max_formula_len],
        dtype=torch.long,
    )
    for formula in formulas[: config.max_n_formulas]
]

print("Tokens input:", formula_tokens)

# Build [B=1, N, L] input tensor
N = len(formula_tokens)
L = max(len(t) for t in formula_tokens)

x = torch.full(
    (1, N, L),
    config.pad_token_id,
    dtype=torch.long,
)

for i, tokens in enumerate(formula_tokens):
    x[0, i, :len(tokens)] = tokens

x = x.to(device)

model.eval()

with torch.no_grad():
    split_logits, side_logits = model(x)

    # Probabilities only for display
    split_probs = torch.softmax(split_logits, dim=-1)[0]      # [N]
    side_probs = torch.softmax(side_logits, dim=-1)[0]        # [N, 2]

    predicted_split = split_probs.argmax().item()

print("\nRESULT:")

for i, formula in enumerate(formulas):
    right_prob = side_probs[i, 0].item()
    left_prob = side_probs[i, 1].item()

    print(
        f"{i}: {formula}"
        f" | split={split_probs[i].item():.3f}"
        f" | RIGHT={right_prob:.3f}"
        f" | LEFT={left_prob:.3f}"
        f" | target={labels[i]}"
    )

print("\nPredicted split index:", predicted_split)
print("Predicted split formula:", formulas[predicted_split])

predicted_labels = []

for i in range(N):
    if i == predicted_split:
        predicted_labels.append(2)
    else:
        predicted_labels.append(side_probs[i].argmax().item())

print("Predicted:", predicted_labels)
print("Target:   ", labels)



