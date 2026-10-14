import random

import torch

from build_proof import ProofFromSplitterBuilder
from mll import proofs as P
from mll.check import check_proof
from mll.formulas import Tensor
from mll.parse import Parser
from models.configs import StateActionConfig
from models.mll_transformers import SideTransformer, SplitTransformer
from tokenizer import FormulaTokenizer

# Use "val" while comparing archis / hyperparams.
# Switch to "test" once at the end for the holdout score (do not tune on it).
EVAL_SPLIT = "val"  # "val" | "test"
EVAL_PATH = f"mll_positional_{EVAL_SPLIT}.txt"
SPLIT_CAPS = {"val": 200_000, "test": 200_000}
MAX_EVAL = 2000

SPLIT_CKPT = "checkpoints/mll_split_transformer_500.pt"
SIDE_CKPT = "checkpoints/mll_side_transformer_500.pt"


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

split_model = SplitTransformer(config)
split_model.load_state_dict(torch.load(SPLIT_CKPT))
split_model = split_model.to(device)
split_model.eval()

side_model = SideTransformer(config)
side_model.load_state_dict(torch.load(SIDE_CKPT))
side_model = side_model.to(device)
side_model.eval()


def transformer_tensor_splitter(sequent: P.Sequent) -> tuple[int, tuple[int, ...]]:
    if not sequent:
        return -1, ()

    formula_tokens = [
        torch.tensor(
            tokenizer.encode(P.fstr(formula))[: config.max_formula_len],
            dtype=torch.long,
        )
        for formula in sequent
    ]

    N = len(formula_tokens)
    L = max((len(t) for t in formula_tokens), default=0)
    if L == 0:
        return -1, ()

    x = torch.full(
        (1, N, L),
        config.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    for i, tokens in enumerate(formula_tokens):
        x[0, i, : len(tokens)] = tokens.to(device)

    with torch.no_grad():
        split_logits = split_model(x)
        principal_index = split_logits[0].argmax().item()

        # If the model did not pick a ⊗, fall back to a random tensor in the sequent.
        if not isinstance(sequent[principal_index], Tensor):
            tensor_indices = [
                i for i, formula in enumerate(sequent) if isinstance(formula, Tensor)
            ]
            if not tensor_indices:
                return -1, ()
            principal_index = random.choice(tensor_indices)

        # Recompute sides conditioned on the (possibly corrected) principal.
        split_index = torch.tensor([principal_index], device=device)
        side_logits = side_model(x, split_index)
        side_pred = side_logits[0].argmax(dim=-1)

    left_positions = tuple(
        i for i in range(N) if i != principal_index and side_pred[i].item() == 1
    )
    return principal_index, left_positions


builder = ProofFromSplitterBuilder(transformer_tensor_splitter)

eval_lines = eval_lines_raw if MAX_EVAL is None else eval_lines_raw[:MAX_EVAL]
n_total = len(eval_lines)
n_valid = 0

print(
    f"Evaluating split→side proof success on {n_total} {EVAL_SPLIT} sequents "
    f"({EVAL_PATH})..."
)

for i, line in enumerate(eval_lines):
    statement = line.split("||")[0]
    try:
        sequent = Parser(statement).parse_sequent()
        proof = builder.build_proof(sequent)
        if check_proof(proof) and proof is not None and proof.sequent == sequent:
            n_valid += 1
    except Exception as e:
        print(f"  [{i + 1}] failed: {type(e).__name__}: {e}")

    if (i + 1) % 100 == 0 or (i + 1) == n_total:
        print(
            f"  [{i + 1}/{n_total}] valid proofs = {n_valid} ({n_valid / (i + 1):.3%})"
        )

print(f"\nProof accuracy: {n_valid}/{n_total} = {n_valid / n_total:.3%}")
