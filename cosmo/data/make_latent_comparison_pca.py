import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# === Last inn Ψ(x,t) for gravitasjonsbølge og fMRI ===
ψ_gw = np.load("Ψ_xt_GW150914_H1.npy")  # shape: (T, 1)
ψ_neuro = np.load("Ψ_xt_sub-01_task-checkerboard_run-1_bold.npy")  # shape: (T, N)

# === Samkjør lengdene ===
T = min(ψ_gw.shape[0], ψ_neuro.shape[0])
ψ_gw = ψ_gw[:T]
ψ_neuro = ψ_neuro[:T]

# === PCA (2 komponenter) for begge systemer ===
latent_gw = PCA(n_components=2).fit_transform(ψ_gw)
latent_neuro = PCA(n_components=2).fit_transform(ψ_neuro)

# === Plot ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(latent_gw[:, 0], latent_gw[:, 1], color='navy')
plt.title("Latent PCA – GW150914")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(latent_neuro[:, 0], latent_neuro[:, 1], color='forestgreen')
plt.title("Latent PCA – fMRI Subject 01")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

plt.tight_layout()
plt.savefig("latent_comparison_pca.png")
plt.show()