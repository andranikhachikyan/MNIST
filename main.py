import streamlit as st
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

st.title("Manifold Learning Playground (MVP)")

# Data
n = st.sidebar.slider("Points", 1000, 3000, 1500, step=250)
noise = st.sidebar.slider("Noise", 0.0, 0.2, 0.05, step=0.01)
X, t = make_swiss_roll(n_samples=n, noise=noise)
X = X[:, [0,2,1]]  # just to make it look nicer

# PCA
pca = PCA(n_components=2).fit_transform(X)

# --- Diffusion Maps (simple version) ---
k = st.sidebar.slider("k-NN (graph)", 5, 50, 15, step=1)
alpha = st.sidebar.slider("Kernel α (0 = standard)", 0.0, 1.0, 0.5, step=0.1)
t_step = st.sidebar.slider("Diffusion time (t)", 1, 10, 1, step=1)

# k-NN graph distances (use Euclidean)
G = kneighbors_graph(X, k, mode="distance", include_self=False)
D2 = G.multiply(G)  # squared distances on edges

# Heat kernel weights: exp(-d^2 / eps). Estimate eps from median edge distance.
edge_vals = D2.data**0.5
eps = np.median(edge_vals)**2 if edge_vals.size else 1.0
W = D2.copy()
W.data = np.exp(-D2.data / (eps + 1e-9))

# Anisotropic normalization (Coifman-Lafon)
deg = np.array(W.sum(axis=1)).ravel()
d_alpha = np.power(deg, -alpha)
D_alpha_inv = d_alpha
# W_alpha = D^{-alpha} W D^{-alpha}
W = W.tocsr()
W = W.multiply(D_alpha_inv[:,None])
W = W.multiply(D_alpha_inv[None,:])

# Markov normalization
deg2 = np.array(W.sum(axis=1)).ravel()
P = W.multiply(1.0 / (deg2 + 1e-9)[:,None]).tocsr()

# Leading eigenvectors of P (right eigenvectors of transpose if needed)
# Compute top m+1 to skip trivial eigenvector when needed
m = 3
vals, vecs = eigsh(P.T, k=m+1)  # largest eigenvalues
idx = np.argsort(-vals)          # sort descending
vals, vecs = vals[idx], vecs[:, idx]
phi = vecs[:, 1:3] * (vals[1:3]**t_step)  # diffusion coords at time t

# Plot
tab1, tab2 = st.tabs(["PCA", "Diffusion Maps"])
with tab1:
    fig, ax = plt.subplots()
    sc = ax.scatter(pca[:,0], pca[:,1], s=6, c=t, cmap="viridis")
    ax.set_title("PCA (2D)")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots()
    sc = ax.scatter(phi[:,0], phi[:,1], s=6, c=t, cmap="viridis")
    ax.set_title("Diffusion Maps (2D)")
    st.pyplot(fig)

st.caption("Tip: increase k for more connected graphs; tweak α for anisotropy; raise t to emphasize large-scale structure.")

# Export
st.download_button("Download PCA (CSV)", pca.astype(float).tobytes(), file_name="pca.bin")
st.download_button("Download Diffusion Maps (CSV)", phi.astype(float).tobytes(), file_name="diffmap.bin")
