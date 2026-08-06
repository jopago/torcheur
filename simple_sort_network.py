import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import mse_loss

torch.manual_seed(0)

n_features = 5
n_samples = 1_000_000

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

X = torch.randn(n_samples, n_features).to(device)
Y, _ = torch.sort(X)

# Initialize the model, loss function, and optimizer
input_dim = n_features
hidden_dim = 128
output_dim = n_features
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim))
model = model.to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
batch_size = 20_000

# Training loop
n_epochs = 5000
for epoch in range(1, n_epochs + 1):
    # Batch subset
    batch_idx = np.random.choice(n_samples, batch_size)
    X_batch = X[batch_idx]
    Y_batch = Y[batch_idx]

    # Forward pass
    Y_pred = model(X_batch)
    loss = loss_fn(Y_pred, Y_batch)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress every 200 epochs
    if epoch % 200 == 0:
        print(f'Epoch {epoch}/{n_epochs}, Loss: {loss.item():.5f}')

# Test the trained model
with torch.no_grad():
    X_test = torch.tensor([[-1, 2, 1.6, 0, 2.5]], dtype=torch.float32).to(device)
    Y_test_pred = model(X_test)
    print("Predictions for test inputs:")
    print("Input:", X_test)
    print("Output:", Y_test_pred[0])

X_test = torch.randn(100_000, n_features, device=device)
Y_test = torch.sort(X_test, dim=1).values

with torch.no_grad():
    mse = loss_fn(model(X_test), Y_test)

print("Test MSE:", mse.item())