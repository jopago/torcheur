import torch

from build_proof import ProofFromSplitterBuilder
from mll.check import check_proof
from mll.parse import ParseError, Parser
from models.configs import StateActionConfig
from models.mll_transformers import SideTransformer, SplitTransformer
from tensor_splitter import tensor_splitter_from_split_side
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

tensor_splitter = tensor_splitter_from_split_side(
    split_model, side_model, tokenizer, config
)
builder = ProofFromSplitterBuilder(tensor_splitter)

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
    except (ParseError, ValueError, TypeError, IndexError) as e:
        print(f"  [{i + 1}] failed: {type(e).__name__}: {e}")

    if (i + 1) % 100 == 0 or (i + 1) == n_total:
        print(
            f"  [{i + 1}/{n_total}] valid proofs = {n_valid} ({n_valid / (i + 1):.3%})"
        )

print(f"\nProof accuracy: {n_valid}/{n_total} = {n_valid / n_total:.3%}")
