import torch

from models.mll_network import Config, MLLNetwork
from tokenizer import FormulaTokenizer
from mll.check import check_proof, check_syntax
with open("mll_positional.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
print(lines[142_359])

SEED = 42
torch.manual_seed(SEED)
tokenizer = FormulaTokenizer.load("tokenizer_mll.json")
vocab_size = len(tokenizer.vocab)

device = "mps"

config = Config(vocab_size=vocab_size,
                max_seq_len=250,
                n_layers=2,
                embedding_dim=64,
                hidden_dim=256)

model = MLLNetwork(config)

state = torch.load("checkpoints/mll_network_1000.pt")
"""state = {
    k.removeprefix("_orig_mod."): v
    for k, v in state.items()
}"""

model.load_state_dict(state)
model = model.to(device)
model.eval()

prompt = "⊢ ⊗(⅋(¬b,b),¬a), ¬e, ⊗(e,⊗(a,¬e)), e"
print("\nPROMPT:")
print(prompt)

# Encode prompt
tokens = tokenizer.encode(prompt)
max_length = 250
temperature = 0.01
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

        next_token = torch.argmax(probs).item()
        # sample
        # next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)

        decoded = tokenizer.decode(tokens)
        print(decoded)

        if "." in decoded:
            break

generated = tokenizer.decode(tokens)

print("\nGENERATED:")
print(generated)
print(check_syntax("generated"))
