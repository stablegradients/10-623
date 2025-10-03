# additive_attention_dotproduct.py
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Hyperparameters
# ----------------------------
d_k = 32          # projection width in the LaTeX spec
num_samples = 500000
batch_size = 10240
epochs = 500
lr = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

g = torch.Generator().manual_seed(42)

# ----------------------------
# Synthetic data: pairs (k, q) and target = k · q
# ----------------------------
K = torch.randn(num_samples, d_k, generator=g)
Q = torch.randn(num_samples, d_k, generator=g)
y = (K * Q).sum(dim=1, keepdim=True)  # dot product target, shape [N, 1]

# Simple dataset/loader
dataset = torch.utils.data.TensorDataset(K, Q, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------------------
# Additive attention score module:
#   s_hat = w_s^T tanh(W_s [k; q])
# with W_s ∈ R^{d_k x 2d_k}, w_s ∈ R^{d_k}
# (no bias terms to match the LaTeX)
# ----------------------------
class AdditiveScore(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.W_s = nn.Parameter(torch.randn(d_k, 2 * d_k) / (2 * d_k) ** 0.5)
        self.w_s = nn.Parameter(torch.randn(d_k) / (d_k) ** 0.5)

    def forward(self, k: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        k: [B, d_k], q: [B, d_k]
        returns s_hat: [B, 1]
        """
        x = torch.cat([k, q], dim=-1)               # [B, 2d_k]
        h = torch.tanh(x @ self.W_s.T)              # [B, d_k]
        s_hat = h @ self.w_s                        # [B]
        return s_hat.unsqueeze(-1)                  # [B, 1]


model = AdditiveScore(d_k).to(device)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_fn = nn.L1Loss()  # absolute difference

# ----------------------------
# Training loop
# ----------------------------
model.train()
for epoch in range(1, epochs + 1):
    running = 0.0
    for k_batch, q_batch, y_batch in loader:
        k_batch = k_batch.to(device)
        q_batch = q_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(k_batch, q_batch)
        loss = loss_fn(pred, y_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running += loss.item() * k_batch.size(0)

    print(f"Epoch {epoch:02d} | L1 loss: {running / num_samples:.6f}")

# ----------------------------
# Quick sanity check
# ----------------------------
model.eval()
with torch.no_grad():
    k_test = torch.randn(8, d_k, generator=g).to(device)
    q_test = torch.randn(8, d_k, generator=g).to(device)
    y_true = (k_test * q_test).sum(dim=1, keepdim=True)
    y_pred = model(k_test, q_test)
    print("\nTrue dot products:\n", y_true.squeeze().cpu().numpy())
    print("Predicted scores:\n",   y_pred.squeeze().cpu().numpy())
