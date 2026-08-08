import torch
from models.mll_network import MLLNetwork, Config
from models.mll_transformer import MLLTransformer
from tokenizer import FormulaTokenizer

with open("mll_positional.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(lines[142_359])
tokenizer = FormulaTokenizer.load("tokenizer_mll.json")
print(tokenizer.vocab)
vocab_size = len(tokenizer.vocab)

device = "mps"

config = Config(vocab_size=vocab_size,
                max_seq_len=250,
                n_layers=2,
                embedding_dim=64,
                hidden_dim=256)
model = MLLNetwork(config)

state = torch.load("checkpoints/mll_network_3000.pt")
"""state = {
    k.removeprefix("_orig_mod."): v
    for k, v in state.items()
}"""

model.load_state_dict(state)
model = model.to(dce)
model.eval()

prompt = "\"⊢ c, (c⊥ ⊗ d), ((d⊥ ⊗ d) ⊗ j), d⊥, j⊥\""
print("\nPROMPT:")
print(prompt)

# Encode prompt
tokens = tokenizer.encode(prompt)
max_length = 200

# Greedy generation
with torch.no_grad():
    while len(tokens) < max_length:

        x = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=device,
        )

        logits = model(x)

        next_token = logits[0, -1].argmax().item()
        tokens.append(next_token)

        decoded_token = tokenizer.decode(tokens)
        print(decoded_token)
        if "." in decoded_token:
            break

generated = tokenizer.decode(tokens)

print("\nGENERATED:")
print(generated)
