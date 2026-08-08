import torch
from mll_network import MLLNetwork, Config
from tokenizer import FormulaTokenizer

tokenizer = FormulaTokenizer.load("tokenizer_mll.json")
vocab_size = len(tokenizer.vocab)

device = "mps"

config = Config(vocab_size=vocab_size)
model = MLLNetwork(config)

state = torch.load("checkpoints/mll_model_9500.pt")
state = {
    k.removeprefix("_orig_mod."): v
    for k, v in state.items()
}

model.load_state_dict(state)
model = model.to(device)
model.eval()

prompt = "\"⊢ i⊥, ((g⊥ ⅋ g) ⊗ i)\""

print("\nPROMPT:")
print(prompt)

# Encode prompt
tokens = tokenizer.encode(prompt)
max_length = 128

# Greedy generation
with torch.no_grad():
    while len(tokens) < max_length:

        x = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=device,
        )

        logits = model(x)

        # Prediction after the last token
        next_token = logits[0, -1].argmax().item()

        tokens.append(next_token)


generated = tokenizer.decode(tokens)

print("\nGENERATED:")
print(generated)