import torch

from build_proof import ProofFromSplitterBuilder
from mll.check import check_proof
from mll.parse import ParseError, Parser
from models.configs import StateActionConfig
from models.mll_transformers import StateActionTransformer
from tensor_splitter import (
    tensor_splitter_from_transformer,
)
from tokenizer import FormulaTokenizer

# Use "val" while comparing archis / hyperparams.
# Switch to "test" once at the end for the holdout score (do not tune on it).
EVAL_SPLIT = "val"  # "val" | "test"
EVAL_PATH = f"mll_positional_{EVAL_SPLIT}.txt"
# Caps in number of dataset LINES (aligned with train_* scripts).
SPLIT_CAPS = {"val": 200_000, "test": 200_000}
# Further cap for CPU runtime of proof search; set to None to use full SPLIT_CAPS.
MAX_EVAL = 2000


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


eval_lines_raw = read_n_lines(EVAL_PATH, SPLIT_CAPS[EVAL_SPLIT])

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
tensor_splitter = tensor_splitter_from_transformer(model, tokenizer, config)
builder = ProofFromSplitterBuilder(tensor_splitter)
state = torch.load("checkpoints/mll_state_transformer_500.pt")
model.load_state_dict(state)
model = model.to(device)
model.eval()

# # --- one-shot demo on an explicit formula (kept for debugging) ---
# from mll.parse import parse_normalized_line
#
# prompt = "⊢ ⊗(⅋(¬a,a),⊗(¬b,⅋(f,¬f))); ⊗(⅋(⊥,𝟙),b); ⊗(⊥,⊗(⊗(⅋(¬e,e),𝟙),𝟙)) || [0, 2, 1]."
# print("\nPROMPT:")
# print(prompt)
#
# formulas, labels = parse_normalized_line(prompt)
# formula_tokens = [
#     torch.tensor(
#         tokenizer.encode(formula)[: config.max_formula_len],
#         dtype=torch.long,
#     )
#     for formula in formulas[: config.max_n_formulas]
# ]
#
# N = len(formula_tokens)
# L = max(len(t) for t in formula_tokens)
#
# x = torch.full(
#     (1, N, L),
#     config.pad_token_id,
#     dtype=torch.long,
# )
#
# for i, tokens in enumerate(formula_tokens):
#     x[0, i, : len(tokens)] = tokens
#
# x = x.to(device)
#
# with torch.no_grad():
#     split_logits, side_logits = model(x)
#     split_probs = torch.softmax(split_logits, dim=-1)[0]
#     side_probs = torch.softmax(side_logits, dim=-1)[0]
#     predicted_split = split_probs.argmax().item()
#
# print("\nPredicted split index:", predicted_split)
# print("Predicted split formula:", formulas[predicted_split])
#
# predicted_labels = []
# for i in range(N):
#     if i == predicted_split:
#         predicted_labels.append(2)
#     else:
#         predicted_labels.append(side_probs[i].argmax().item())
#
# print("Predicted:", predicted_labels)
# print("Target:   ", labels)
#
# builder = ProofFromSplitterBuilder(transformer_tensor_splitter)
# statement = prompt.split("||")[0]
# parser = Parser(statement)
# sequent = parser.parse_sequent()
# proof = builder.build_proof(sequent)
# print("Proof valid", check_proof(proof) and proof.sequent == sequent)

eval_lines = eval_lines_raw if MAX_EVAL is None else eval_lines_raw[:MAX_EVAL]
n_total = len(eval_lines)
n_valid = 0

print(f"Evaluating proof success on {n_total} {EVAL_SPLIT} sequents ({EVAL_PATH})...")

for i, line in enumerate(eval_lines):
    statement = line.split("||")[0]
    try:
        sequent = Parser(statement).parse_sequent()
        proof = builder.build_proof(sequent)
        if check_proof(proof) and proof is not None and proof.sequent == sequent:
            n_valid += 1
    except (ParseError, ValueError, TypeError, IndexError) as e:
        print(f"  [{i + 1}] failed: {type(e).__name__}: {e}")

    if (i + 1) % 100 == 0 or (i + 1) == n_total:
        print(
            f"  [{i + 1}/{n_total}] valid proofs = {n_valid} ({n_valid / (i + 1):.3%})"
        )

print(f"\nProof accuracy: {n_valid}/{n_total} = {n_valid / n_total:.3%}")
