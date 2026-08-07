import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

plt.style.use("ggplot")
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

p = 103

a = torch.arange(p, device=device)
b = torch.arange(p, device=device)
A, B = torch.meshgrid(a, b)

X = torch.stack(
    [A.flatten(), B.flatten()],
    dim=1,
)

Y = ((A + B) % p).flatten()

class ModularAdder(nn.Module):
    def __init__(self, p, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(p, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, p)
        )

    def forward(self, x):
        a = self.embedding(x[:, 0])
        b = self.embedding(x[:, 1])
        return self.mlp(torch.cat([a, b], dim=1))

raw_model = ModularAdder(p).to(device)

n_samples = X.shape[0]
n_train = int(n_samples * 0.5)

# Train / test split
perm = torch.randperm(n_samples)
train_idx = perm[:n_train]
test_idx = perm[n_train:]

X_train = X[train_idx]
Y_train = Y[train_idx]

X_test = X[test_idx]
Y_test = Y[test_idx]

model = torch.compile(raw_model)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
loss_fn = nn.CrossEntropyLoss()

n_epochs = 25_000

train_acc = []
test_acc = []

for epoch in range(1, n_epochs):
    optimizer.zero_grad()

    y_pred = model(X_train)
    loss = loss_fn(y_pred, Y_train)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        train_accuracy = (y_pred.argmax(dim=1) == Y_train).float().mean()
        train_acc.append(train_accuracy.item())

        y_pred_test = model(X_test)
        test_loss = loss_fn(y_pred_test, Y_test)
        test_accuracy = (y_pred_test.argmax(dim=1) == Y_test).float().mean()
        test_acc.append(test_accuracy.item())

    # Print progress every 200 epochs
    if epoch % 200 == 0:
        print(
            f'Epoch {epoch}/{n_epochs}, Train accuray: {train_accuracy.item():.5f}, Test accuracy: {test_accuracy.item():.5f}')

plt.plot(np.log(range(1, n_epochs)), train_acc, label="Train Accuracy (%)")
plt.plot(np.log(range(1, n_epochs)), test_acc, label="Test Accuracy (%)", linestyle="dashed")
plt.title("Modular Addition")
plt.legend(loc='upper left')
plt.savefig("modular_grokking.svg",dpi=200,bbox_inches="tight")
plt.show()

W = raw_model.mlp[-1].weight.detach().cpu()   # [p, hidden_dim]
F = torch.fft.fft(W, dim=0)
power = (F.abs() ** 2).mean(dim=1)
freqs = torch.arange(p)

plt.plot(freqs[:p//2], power[:p//2])
plt.xlabel("Fourier frequency k")
plt.ylabel("Mean power")
plt.title("Fourier spectrum of output weights")
plt.show()