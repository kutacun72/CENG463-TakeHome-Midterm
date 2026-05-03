# Question 3 - Dimensionality Reduction
# CENG 463 Machine Learning Take-Home Midterm
import tensorflow as tf
from tensorflow.keras import layers, models
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score


# -----------------------------
# Step 2: Load MNIST dataset
# -----------------------------

print("Loading MNIST dataset...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

print("MNIST loaded successfully.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", np.unique(y))
print("Pixel value range before scaling:", X.min(), X.max())

# -----------------------------
# Step 3: Preprocessing and sampling
# -----------------------------

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Normalize pixel values from [0, 255] to [0, 1]
X = X / 255.0

# Use a stratified subset for dimensionality reduction experiments
# This keeps all digit classes balanced while reducing computational cost.
X_sample, _, y_sample, _ = train_test_split(
    X,
    y,
    train_size=5000,
    stratify=y,
    random_state=RANDOM_STATE
)

# Split the sampled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    stratify=y_sample,
    random_state=RANDOM_STATE
)

print("\nPreprocessing completed.")
print("Pixel value range after scaling:", X.min(), X.max())
print("Sample shape:", X_sample.shape)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Sample class distribution:")
print(pd.Series(y_sample).value_counts().sort_index())

# -----------------------------
# Step 4: PCA dimensionality reduction
# -----------------------------

print("\nRunning PCA...")

start_time = time.time()

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

pca_runtime = time.time() - start_time

# Reconstruction from 2D PCA space
X_test_pca_reconstructed = pca.inverse_transform(X_test_pca)
pca_mse = mean_squared_error(X_test, X_test_pca_reconstructed)

# Downstream k-NN classification on reduced space
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_pca = knn_pca.predict(X_test_pca)
pca_knn_accuracy = accuracy_score(y_test, y_pred_pca)

print("PCA completed.")
print("PCA runtime:", round(pca_runtime, 4), "seconds")
print("PCA reconstruction MSE:", round(pca_mse, 6))
print("PCA k-NN accuracy:", round(pca_knn_accuracy, 4))
print("PCA explained variance ratio:", pca.explained_variance_ratio_)
print("PCA total explained variance:", round(np.sum(pca.explained_variance_ratio_), 4))

# Save PCA 2D embedding plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=y_test,
    cmap="tab10",
    s=10,
    alpha=0.7
)
plt.colorbar(scatter, label="Digit Class")
plt.title("PCA 2D Embedding of MNIST")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("q3_pca_embedding.png", dpi=300)
plt.close()

print("PCA embedding plot saved as q3_pca_embedding.png")

# -----------------------------
# Step 5: Kernel PCA dimensionality reduction
# -----------------------------

print("\nRunning Kernel PCA...")

start_time = time.time()

kpca = KernelPCA(
    n_components=2,
    kernel="rbf",
    gamma=0.01,
    fit_inverse_transform=True,
    random_state=RANDOM_STATE
)

X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)

kpca_runtime = time.time() - start_time

# Reconstruction from Kernel PCA space
X_test_kpca_reconstructed = kpca.inverse_transform(X_test_kpca)
kpca_mse = mean_squared_error(X_test, X_test_kpca_reconstructed)

# Downstream k-NN classification on reduced space
knn_kpca = KNeighborsClassifier(n_neighbors=5)
knn_kpca.fit(X_train_kpca, y_train)
y_pred_kpca = knn_kpca.predict(X_test_kpca)
kpca_knn_accuracy = accuracy_score(y_test, y_pred_kpca)

print("Kernel PCA completed.")
print("Kernel PCA runtime:", round(kpca_runtime, 4), "seconds")
print("Kernel PCA reconstruction MSE:", round(kpca_mse, 6))
print("Kernel PCA k-NN accuracy:", round(kpca_knn_accuracy, 4))

# Save Kernel PCA 2D embedding plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_test_kpca[:, 0],
    X_test_kpca[:, 1],
    c=y_test,
    cmap="tab10",
    s=10,
    alpha=0.7
)
plt.colorbar(scatter, label="Digit Class")
plt.title("Kernel PCA 2D Embedding of MNIST")
plt.xlabel("Kernel Principal Component 1")
plt.ylabel("Kernel Principal Component 2")
plt.tight_layout()
plt.savefig("q3_kernel_pca_embedding.png", dpi=300)
plt.close()

print("Kernel PCA embedding plot saved as q3_kernel_pca_embedding.png")

# -----------------------------
# Step 6: t-SNE dimensionality reduction
# -----------------------------

print("\nRunning t-SNE with perplexity grid search...")

tsne_results = []

def compute_kruskal_stress(X_high, X_low):
    """
    Computes Kruskal's stress between pairwise distances in original and reduced spaces.
    To reduce computation time, this function uses a subset if the input is large.
    """
    from sklearn.metrics import pairwise_distances

    max_points = 1000
    if X_high.shape[0] > max_points:
        idx = np.random.choice(X_high.shape[0], max_points, replace=False)
        X_high_sub = X_high[idx]
        X_low_sub = X_low[idx]
    else:
        X_high_sub = X_high
        X_low_sub = X_low

    D_high = pairwise_distances(X_high_sub)
    D_low = pairwise_distances(X_low_sub)

    stress = np.sqrt(np.sum((D_high - D_low) ** 2) / np.sum(D_high ** 2))
    return stress


for perplexity in [5, 30, 50]:
    print(f"\nRunning t-SNE with perplexity={perplexity}...")

    start_time = time.time()

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE
    )

    X_sample_tsne = tsne.fit_transform(X_sample)

    tsne_runtime = time.time() - start_time

    # Use the same 80/20 split indices logic by splitting the embedding consistently
    X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(
        X_sample_tsne,
        y_sample,
        test_size=0.2,
        stratify=y_sample,
        random_state=RANDOM_STATE
    )

    knn_tsne = KNeighborsClassifier(n_neighbors=5)
    knn_tsne.fit(X_train_tsne, y_train_tsne)
    y_pred_tsne = knn_tsne.predict(X_test_tsne)
    tsne_knn_accuracy = accuracy_score(y_test_tsne, y_pred_tsne)

    tsne_trustworthiness = trustworthiness(X_sample, X_sample_tsne, n_neighbors=5)
    tsne_stress = compute_kruskal_stress(X_sample, X_sample_tsne)

    tsne_results.append({
        "Method": f"t-SNE perplexity={perplexity}",
        "Runtime": tsne_runtime,
        "Trustworthiness": tsne_trustworthiness,
        "Kruskal Stress": tsne_stress,
        "kNN Accuracy": tsne_knn_accuracy
    })

    print(f"t-SNE perplexity={perplexity} completed.")
    print("Runtime:", round(tsne_runtime, 4), "seconds")
    print("Trustworthiness:", round(tsne_trustworthiness, 4))
    print("Kruskal stress:", round(tsne_stress, 4))
    print("k-NN accuracy:", round(tsne_knn_accuracy, 4))

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_sample_tsne[:, 0],
        X_sample_tsne[:, 1],
        c=y_sample,
        cmap="tab10",
        s=10,
        alpha=0.7
    )
    plt.colorbar(scatter, label="Digit Class")
    plt.title(f"t-SNE 2D Embedding of MNIST, Perplexity={perplexity}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(f"q3_tsne_perplexity_{perplexity}.png", dpi=300)
    plt.close()

    print(f"t-SNE plot saved as q3_tsne_perplexity_{perplexity}.png")


tsne_results_df = pd.DataFrame(tsne_results)
print("\nt-SNE results summary:")
print(tsne_results_df)

# -----------------------------
# Step 7: UMAP dimensionality reduction
# -----------------------------

print("\nRunning UMAP with parameter tuning...")

umap_results = []

umap_param_grid = [
    {"n_neighbors": 10, "min_dist": 0.0},
    {"n_neighbors": 10, "min_dist": 0.5},
    {"n_neighbors": 30, "min_dist": 0.0},
    {"n_neighbors": 30, "min_dist": 0.5},
]

for params in umap_param_grid:
    n_neighbors = params["n_neighbors"]
    min_dist = params["min_dist"]

    print(f"\nRunning UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")

    start_time = time.time()

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=RANDOM_STATE
    )

    X_sample_umap = reducer.fit_transform(X_sample)

    umap_runtime = time.time() - start_time

    X_train_umap, X_test_umap, y_train_umap, y_test_umap = train_test_split(
        X_sample_umap,
        y_sample,
        test_size=0.2,
        stratify=y_sample,
        random_state=RANDOM_STATE
    )

    knn_umap = KNeighborsClassifier(n_neighbors=5)
    knn_umap.fit(X_train_umap, y_train_umap)
    y_pred_umap = knn_umap.predict(X_test_umap)
    umap_knn_accuracy = accuracy_score(y_test_umap, y_pred_umap)

    umap_trustworthiness = trustworthiness(X_sample, X_sample_umap, n_neighbors=5)
    umap_stress = compute_kruskal_stress(X_sample, X_sample_umap)

    umap_results.append({
        "Method": f"UMAP n={n_neighbors}, min_dist={min_dist}",
        "Runtime": umap_runtime,
        "Trustworthiness": umap_trustworthiness,
        "Kruskal Stress": umap_stress,
        "kNN Accuracy": umap_knn_accuracy
    })

    print(f"UMAP n_neighbors={n_neighbors}, min_dist={min_dist} completed.")
    print("Runtime:", round(umap_runtime, 4), "seconds")
    print("Trustworthiness:", round(umap_trustworthiness, 4))
    print("Kruskal stress:", round(umap_stress, 4))
    print("k-NN accuracy:", round(umap_knn_accuracy, 4))

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_sample_umap[:, 0],
        X_sample_umap[:, 1],
        c=y_sample,
        cmap="tab10",
        s=10,
        alpha=0.7
    )
    plt.colorbar(scatter, label="Digit Class")
    plt.title(f"UMAP 2D Embedding of MNIST, n={n_neighbors}, min_dist={min_dist}")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(f"q3_umap_n{n_neighbors}_mindist{min_dist}.png", dpi=300)
    plt.close()

    print(f"UMAP plot saved as q3_umap_n{n_neighbors}_mindist{min_dist}.png")


umap_results_df = pd.DataFrame(umap_results)
print("\nUMAP results summary:")
print(umap_results_df)

# -----------------------------
# Step 8: Undercomplete Autoencoder
# -----------------------------

print("\nRunning Autoencoder...")

tf.random.set_seed(RANDOM_STATE)

start_time = time.time()

input_dim = X_train.shape[1]
bottleneck_dim = 2

# Encoder
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(256, activation="relu")(input_layer)
encoded = layers.Dense(64, activation="relu")(encoded)
latent = layers.Dense(bottleneck_dim, activation="linear", name="latent_space")(encoded)

# Decoder
decoded = layers.Dense(64, activation="relu")(latent)
decoded = layers.Dense(256, activation="relu")(decoded)
output_layer = layers.Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = models.Model(inputs=input_layer, outputs=output_layer)

encoder = models.Model(inputs=input_layer, outputs=latent)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

ae_runtime = time.time() - start_time

# Latent representations
X_train_ae = encoder.predict(X_train)
X_test_ae = encoder.predict(X_test)

# Reconstruction error on test set
X_test_ae_reconstructed = autoencoder.predict(X_test)
ae_mse = mean_squared_error(X_test, X_test_ae_reconstructed)

# Downstream k-NN classification on latent space
knn_ae = KNeighborsClassifier(n_neighbors=5)
knn_ae.fit(X_train_ae, y_train)
y_pred_ae = knn_ae.predict(X_test_ae)
ae_knn_accuracy = accuracy_score(y_test, y_pred_ae)

print("Autoencoder completed.")
print("Autoencoder runtime:", round(ae_runtime, 4), "seconds")
print("Autoencoder reconstruction MSE:", round(ae_mse, 6))
print("Autoencoder k-NN accuracy:", round(ae_knn_accuracy, 4))
print("Final training loss:", round(history.history["loss"][-1], 6))
print("Final validation loss:", round(history.history["val_loss"][-1], 6))

# Save Autoencoder latent space plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_test_ae[:, 0],
    X_test_ae[:, 1],
    c=y_test,
    cmap="tab10",
    s=10,
    alpha=0.7
)
plt.colorbar(scatter, label="Digit Class")
plt.title("Autoencoder 2D Latent Space of MNIST")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.tight_layout()
plt.savefig("q3_autoencoder_latent_space.png", dpi=300)
plt.close()

print("Autoencoder latent space plot saved as q3_autoencoder_latent_space.png")

# Save Autoencoder loss curve
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("q3_autoencoder_loss_curve.png", dpi=300)
plt.close()

print("Autoencoder loss curve saved as q3_autoencoder_loss_curve.png")
# -----------------------------
# Step 9: Continuity score and final summary tables
# -----------------------------

print("\nComputing continuity scores and 5-fold CV accuracies...")

from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold


def continuity_score(X_high, X_low, n_neighbors=5):
    """
    Computes continuity score.
    Trustworthiness measures whether neighbors in low-dimensional space
    are also neighbors in the original space.
    Continuity measures the opposite direction: whether original neighbors
    are preserved in the low-dimensional space.
    """
    return trustworthiness(X_low, X_high, n_neighbors=n_neighbors)


def knn_cv_accuracy(X_reduced, y_labels, k=5, cv=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(
        knn,
        X_reduced,
        y_labels,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy"
    )
    return scores.mean(), scores.std()


# PCA and Kernel PCA CV accuracy
pca_cv_mean, pca_cv_std = knn_cv_accuracy(
    np.vstack((X_train_pca, X_test_pca)),
    np.hstack((y_train, y_test))
)

kpca_cv_mean, kpca_cv_std = knn_cv_accuracy(
    np.vstack((X_train_kpca, X_test_kpca)),
    np.hstack((y_train, y_test))
)

ae_cv_mean, ae_cv_std = knn_cv_accuracy(
    np.vstack((X_train_ae, X_test_ae)),
    np.hstack((y_train, y_test))
)

# Re-run/store best t-SNE and best UMAP embeddings for final comparable metrics
# Based on previous results:
# Best t-SNE: perplexity=30
# Best UMAP: n_neighbors=10, min_dist=0.0

print("\nRecomputing best t-SNE embedding for continuity and CV...")
best_tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=RANDOM_STATE
)
X_best_tsne = best_tsne.fit_transform(X_sample)

best_tsne_trust = trustworthiness(X_sample, X_best_tsne, n_neighbors=5)
best_tsne_cont = continuity_score(X_sample, X_best_tsne, n_neighbors=5)
best_tsne_stress = compute_kruskal_stress(X_sample, X_best_tsne)
best_tsne_cv_mean, best_tsne_cv_std = knn_cv_accuracy(X_best_tsne, y_sample)

print("Best t-SNE continuity:", round(best_tsne_cont, 4))
print("Best t-SNE 5-fold k-NN accuracy:", round(best_tsne_cv_mean, 4), "+/-", round(best_tsne_cv_std, 4))

print("\nRecomputing best UMAP embedding for continuity and CV...")
best_umap = umap.UMAP(
    n_components=2,
    n_neighbors=10,
    min_dist=0.0,
    metric="euclidean",
    random_state=RANDOM_STATE
)
X_best_umap = best_umap.fit_transform(X_sample)

best_umap_trust = trustworthiness(X_sample, X_best_umap, n_neighbors=5)
best_umap_cont = continuity_score(X_sample, X_best_umap, n_neighbors=5)
best_umap_stress = compute_kruskal_stress(X_sample, X_best_umap)
best_umap_cv_mean, best_umap_cv_std = knn_cv_accuracy(X_best_umap, y_sample)

print("Best UMAP continuity:", round(best_umap_cont, 4))
print("Best UMAP 5-fold k-NN accuracy:", round(best_umap_cv_mean, 4), "+/-", round(best_umap_cv_std, 4))


reconstruction_df = pd.DataFrame({
    "Method": ["PCA", "Kernel PCA", "Autoencoder"],
    "Reconstruction MSE": [pca_mse, kpca_mse, ae_mse],
    "Runtime (s)": [pca_runtime, kpca_runtime, ae_runtime],
    "5-fold kNN Accuracy Mean": [pca_cv_mean, kpca_cv_mean, ae_cv_mean],
    "5-fold kNN Accuracy Std": [pca_cv_std, kpca_cv_std, ae_cv_std]
})

manifold_df = pd.DataFrame({
    "Method": ["t-SNE perplexity=30", "UMAP n=10, min_dist=0.0"],
    "Trustworthiness": [best_tsne_trust, best_umap_trust],
    "Continuity": [best_tsne_cont, best_umap_cont],
    "Kruskal Stress": [best_tsne_stress, best_umap_stress],
    "5-fold kNN Accuracy Mean": [best_tsne_cv_mean, best_umap_cv_mean],
    "5-fold kNN Accuracy Std": [best_tsne_cv_std, best_umap_cv_std]
})

print("\nReconstruction-based methods summary:")
print(reconstruction_df)

print("\nBest manifold methods summary:")
print(manifold_df)

reconstruction_df.to_csv("q3_reconstruction_methods_summary.csv", index=False)
manifold_df.to_csv("q3_manifold_methods_summary.csv", index=False)

print("\nSummary tables saved as:")
print("q3_reconstruction_methods_summary.csv")
print("q3_manifold_methods_summary.csv")

# -----------------------------
# Step 10: Final comparison plots
# -----------------------------

print("\nCreating final comparison plots...")

# 1) Reconstruction MSE comparison
plt.figure(figsize=(7, 5))
plt.bar(
    reconstruction_df["Method"],
    reconstruction_df["Reconstruction MSE"]
)
plt.title("Reconstruction MSE Comparison")
plt.ylabel("Mean Squared Error")
plt.xlabel("Method")
plt.tight_layout()
plt.savefig("q3_reconstruction_mse_comparison.png", dpi=300)
plt.close()

# 2) 5-fold k-NN accuracy comparison
all_methods = [
    "PCA",
    "Kernel PCA",
    "Autoencoder",
    "t-SNE",
    "UMAP"
]

all_knn_means = [
    pca_cv_mean,
    kpca_cv_mean,
    ae_cv_mean,
    best_tsne_cv_mean,
    best_umap_cv_mean
]

all_knn_stds = [
    pca_cv_std,
    kpca_cv_std,
    ae_cv_std,
    best_tsne_cv_std,
    best_umap_cv_std
]

plt.figure(figsize=(8, 5))
plt.bar(
    all_methods,
    all_knn_means,
    yerr=all_knn_stds,
    capsize=5
)
plt.title("5-fold k-NN Accuracy on 2D Reduced Space")
plt.ylabel("Accuracy")
plt.xlabel("Dimensionality Reduction Method")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("q3_knn_accuracy_comparison.png", dpi=300)
plt.close()

# 3) Trustworthiness and continuity comparison
manifold_metrics = ["Trustworthiness", "Continuity"]
tsne_values = [best_tsne_trust, best_tsne_cont]
umap_values = [best_umap_trust, best_umap_cont]

x = np.arange(len(manifold_metrics))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width / 2, tsne_values, width, label="t-SNE")
plt.bar(x + width / 2, umap_values, width, label="UMAP")
plt.xticks(x, manifold_metrics)
plt.ylim(0.9, 1.0)
plt.ylabel("Score")
plt.title("Trustworthiness and Continuity Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("q3_trustworthiness_continuity_comparison.png", dpi=300)
plt.close()

print("Final comparison plots saved:")
print("q3_reconstruction_mse_comparison.png")
print("q3_knn_accuracy_comparison.png")
print("q3_trustworthiness_continuity_comparison.png")