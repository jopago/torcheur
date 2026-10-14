import random

from build_proof import ProofFromSplitterBuilder
from mll import proofs as P
from mll.check import check_proof
from mll.formulas import Tensor
from mll.parse import Parser

# Use "val" while comparing archis / hyperparams.
# Switch to "test" once at the end for the holdout score (do not tune on it).
EVAL_SPLIT = "val"  # "val" | "test"
EVAL_PATH = f"mll_positional_{EVAL_SPLIT}.txt"
SPLIT_CAPS = {"val": 200_000, "test": 200_000}
MAX_EVAL = 2000
SEED = 0


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


def random_tensor_splitter(sequent: P.Sequent) -> tuple[int, tuple[int, ...]]:
    """Baseline: pick a random ⊗ and a random LEFT/RIGHT partition."""
    if not sequent:
        return -1, ()

    tensor_indices = [
        i for i, formula in enumerate(sequent) if isinstance(formula, Tensor)
    ]
    if not tensor_indices:
        return -1, ()

    principal_index = random.choice(tensor_indices)
    left_positions = tuple(
        sorted(
            i
            for i in range(len(sequent))
            if i != principal_index and random.random() < 0.5
        )
    )
    return principal_index, left_positions


random.seed(SEED)
eval_lines_raw = read_n_lines(EVAL_PATH, SPLIT_CAPS[EVAL_SPLIT])
eval_lines = eval_lines_raw if MAX_EVAL is None else eval_lines_raw[:MAX_EVAL]

builder = ProofFromSplitterBuilder(random_tensor_splitter)

n_total = len(eval_lines)
n_valid = 0

print(
    f"Evaluating random splitter on {n_total} {EVAL_SPLIT} sequents ({EVAL_PATH})..."
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
            f"  [{i + 1}/{n_total}] valid proofs = {n_valid} "
            f"({n_valid / (i + 1):.3%})"
        )

print(f"\nProof accuracy: {n_valid}/{n_total} = {n_valid / n_total:.3%}")
