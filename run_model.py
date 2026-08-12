import torch

from build_proof import ProofFromSplitterBuilder
from mll.parse import Parser
from models.configs import StateActionConfig
from models.mll_transformers import StateActionTransformer
from tensor_splitter import (
    random_tensor_splitter,
    tensor_splitter_from_transformer,
)
from tokenizer import FormulaTokenizer

with open("mll_positional.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]


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
tensor_splitter = tensor_splitter_from_transformer(model, tokenizer, config)
random_splitter = random_tensor_splitter()

state = torch.load("checkpoints/mll_state_transformer_500.pt")
model.load_state_dict(state)
model = model.to(device)
proof_builder = ProofFromSplitterBuilder(tensor_splitter)
random_proof_builder = ProofFromSplitterBuilder(random_splitter)

test_lines = lines[220_000:222_000]
count_valid = 0
count_valid_random = 0
total = 0


for line in test_lines:
    statement = line.split("||")[0]
    parser = Parser(statement)
    sequent = parser.parse_sequent()

    if len(sequent) == 1:
        continue

    total += 1

    # predicted_split, left_positions = tensor_splitter(sequent)
    # print("\nPredicted split index:", predicted_split)
    # print("Predicted left positions:", left_positions)
    # print("Target:", prompt.split("||")[1])

    from mll.check import check_proof

    try:
        proof = proof_builder.build_proof(sequent)
        if check_proof(proof) and proof.sequent == sequent:
            # print("proof valid")
            count_valid += 1
    except:
        pass

    try:
        proof = random_proof_builder.build_proof(sequent)
        if check_proof(proof) and proof.sequent == sequent:
            count_valid_random += 1
    except:
        pass


print(f"Valid proofs= {count_valid} / {total} - ({int(100 * count_valid / total)}%)")
print(
    f"Valid random proofs= {count_valid_random} / {total} - ({int(100 * count_valid_random / total)}%)"
)
