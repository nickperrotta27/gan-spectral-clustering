from google.colab import files

uploaded = files.upload()
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

# === Load target distribution ===
with open("gan_target_distribution.pkl", "rb") as f:
    target_data = pickle.load(f)  # shape: (N, 2)
target_data = torch.tensor(target_data, dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Define Generator ===
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 2)
        )

    def forward(self, z):
        return self.net(z)

# === Define Discriminator ===
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# === Initialize models and optimizers ===
generator = Generator().to(device)
discriminator = Discriminator().to(device)
opt_g = optim.Adam(generator.parameters(), lr=1e-4)
opt_d = optim.Adam(discriminator.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()

# === Training loop ===
num_epochs = 5000
batch_size = 128

for epoch in range(num_epochs):
    # ---- Train discriminator for 10 steps ----
    for _ in range(10):
        real_idx = torch.randint(0, target_data.shape[0], (batch_size,))
        real_samples = target_data[real_idx].to(device)
        z = torch.randn(batch_size, 2).to(device)
        fake_samples = generator(z).detach()

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        d_real = discriminator(real_samples)
        d_fake = discriminator(fake_samples)
        loss_d = loss_fn(d_real, real_labels) + loss_fn(d_fake, fake_labels)

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

    # ---- Train generator for 1 step ----
    z = torch.randn(batch_size, 2).to(device)
    fake_samples = generator(z)
    d_fake = discriminator(fake_samples)
    loss_g = loss_fn(d_fake, torch.ones(batch_size, 1).to(device))

    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()

    if epoch % 500 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] | D_loss: {loss_d.item():.4f} | G_loss: {loss_g.item():.4f}")


X, Y = torch.meshgrid(torch.linspace(0, 8, 50), torch.linspace(0, 8, 50))
xgrid = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
discGrid = discriminator(xgrid.to(device))
discGrid = discGrid.detach().cpu().numpy()

plt.figure(figsize=(6,6))
plt.scatter(xgrid[:,0], xgrid[:,1], c=discGrid, cmap="viridis")
plt.colorbar(label="Discriminator Output")
plt.title("Discriminator Output over 2D Grid")
plt.show()

z = torch.randn((2000, 2))
xhat = generator(z.to(device)).detach().cpu()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(z[:, 0], z[:, 1], s=5)
ax[0].set_title("Network Input (Noise)")
ax[1].scatter(xhat[:, 0], xhat[:, 1], s=5)
ax[1].set_title("Network Output (Generated Samples)")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

# ============================================================
# 1. LOAD DATA
# ============================================================
with open("spectral_clustering_data.pkl", "rb") as f:
    X = pickle.load(f)

X = np.array(X)
N = X.shape[0]

# ============================================================
# 2. BUILD AFFINITY MATRIX (RBF)
# ============================================================

# Automatic sigma (median of pairwise distances)
dist = pairwise_distances(X)
sigma = np.median(dist[dist > 0])

W = np.exp(-(dist**2) / (2 * sigma**2))
np.fill_diagonal(W, 0)

# ============================================================
# 3. NORMALIZED LAPLACIAN
# ============================================================

deg = W.sum(axis=1)
D_inv_sqrt = np.diag(1 / np.sqrt(deg + 1e-12))

L = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

# ============================================================
# 4. EIGENDECOMPOSITION
# ============================================================

eigvals, eigvecs = np.linalg.eigh(L)

# k = 3 clusters → use eigenvectors 1, 2, 3
U = eigvecs[:, 1:4]

# Row-normalize
U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

# ============================================================
# 5. RUN CLUSTERING
# ============================================================

# --- Spectral ---
kmeans_spec = KMeans(n_clusters=3, n_init=20, random_state=0)
labels_spectral = kmeans_spec.fit_predict(U_norm)

# --- Raw KMeans ---
kmeans_raw = KMeans(n_clusters=3, n_init=20, random_state=1)
labels_kmeans = kmeans_raw.fit_predict(X)

# ============================================================
# 6. DIAGNOSTICS (PROVES THEY ARE DIFFERENT)
# ============================================================

print("Spectral label counts:", np.bincount(labels_spectral))
print("KMeans label counts:", np.bincount(labels_kmeans))
print("Label arrays identical?:", np.all(labels_spectral == labels_kmeans))

# ============================================================
# 7. PLOTS
# ============================================================

plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=labels_spectral, cmap="Set2", s=20)
plt.title("Spectral Clustering (k=3)")
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=labels_kmeans, cmap="Set1", s=20)
plt.title("KMeans Directly on Input Data (k=3)")
plt.show()

