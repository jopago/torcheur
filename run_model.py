import torch

from mll.check import check_proof, check_syntax
from models.mll_network import Config, MLLNetwork
from models.mll_transformer import MLLTransformer
from tokenizer import FormulaTokenizer

with open("mll_positional.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
print(lines[165_712])

SEED = 42
# torch.manual_seed(SEED)
tokenizer = FormulaTokenizer.load("tokenizer_mll.json")
vocab_size = len(tokenizer.vocab)

device = "mps"

config = Config(
    vocab_size=vocab_size,
    max_seq_len=250,
    n_layers=4,
    embedding_dim=256,
    n_heads=4,
    ff_dim=512,
)

model = MLLTransformer(config)

state = torch.load("checkpoints/mll_transformer_8000.pt")
model.load_state_dict(state)
model = model.to(device)
model.eval()

prompt = "⊢ ⊗(⊗(⅋(a,¬a),c),¬f), ¬f, a, ⊗(¬a,¬j), ⅋(⊗(a,¬a),⊗(⅋(⊗(¬c,j),¬j),⊗(j,¬a))), ⊥, ⊗(f,f), ⊗(a,¬i), ⊗(i,⅋(⊗(h,⊗(𝟙,h)),⅋(⅋(⊗(¬h,⅋(h,¬h)),⅋(⊥,¬h)),⊥)))"
print("\nPROMPT:")
print(prompt)

# Encode prompt
base_tokens = tokenizer.encode(prompt)
max_length = 250
temperature = 0.2

num_proof_try = 10
correct_syntax = 0
correct_proof = 0
for k in range(num_proof_try):
    tokens = base_tokens.copy()
    with torch.no_grad():
        while len(tokens) < max_length:
            x = torch.tensor(
                [tokens],
                dtype=torch.long,
                device=device,
            )

            logits = model(x)

            next_logits = logits[0, -1] / temperature
            probs = torch.softmax(next_logits, dim=-1)

            # next_token = torch.argmax(probs).item()
            # sample
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)

            decoded = tokenizer.decode(tokens)

            if "." in decoded:
                break

    generated = tokenizer.decode(tokens)
    print(generated)
    if check_syntax(generated):
        correct_syntax += 1
    if check_proof(generated):
        correct_proof += 1

print("Correct syntax: ", correct_syntax, " / ", num_proof_try)
print("Correct proof: ", correct_proof, " / ", num_proof_try)
