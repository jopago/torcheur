import torch

from build_proof import ProofFromSplitterBuilder
from mll.parse import Parser
from models.configs import StateActionConfig
from models.mll_transformers import StateActionTransformer
from tokenizer import FormulaTokenizer

with open("mll_positional.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
print(lines[289_125])

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
from mll import proofs as P


def transformer_tensor_splitter(
    transformer: StateActionTransformer, sequent: P.Sequent
) -> tuple[int, tuple[int, ...]]:
    formula_tokens = [
        torch.tensor(
            tokenizer.encode(P.fstr(formula))[: config.max_formula_len],
            dtype=torch.long,
        )
        for formula in sequent
    ]

    N = len(formula_tokens)
    L = max(len(t) for t in formula_tokens)

    x = torch.full(
        (1, N, L),
        config.pad_token_id,
        dtype=torch.long,
        device=device,
    )

    for i, tokens in enumerate(formula_tokens):
        x[0, i, : len(tokens)] = tokens.to(device)

    with torch.no_grad():
        split_logits, side_logits = transformer(x)

    # Which tensor to split
    principal_index = split_logits[0].argmax().item()

    # Formulas assigned to LEFT
    side_pred = side_logits[0].argmax(dim=-1)

    left_positions = tuple(
        i for i in range(N) if i != principal_index and side_pred[i].item() == 1
    )

    return principal_index, left_positions


model = StateActionTransformer(config)

state = torch.load("checkpoints/mll_state_transformer_500.pt")
model.load_state_dict(state)
model = model.to(device)

prompt = "⊢ 𝟙; ⊗(⊥,𝟙); ⊗(⊥,¬j); j || [1, 2, 0, 0]."
print("\nPROMPT:")
print(prompt)


statement = prompt.split("||")[0]
parser = Parser(statement)
sequent = parser.parse_sequent()

predicted_split, left_positions = transformer_tensor_splitter(model, sequent)

print("\nPredicted split index:", predicted_split)
print("Predicted left positions:", left_positions)
print("Target:", prompt.split("||")[1])

builder = ProofFromSplitterBuilder(lambda seq: transformer_tensor_splitter(model, seq))

from mll.check import check_proof

try:
    proof = builder.build_proof(sequent)
    if check_proof(proof) and proof.sequent == sequent:
        print("proof valid")
except:
    print("proof invalid")
