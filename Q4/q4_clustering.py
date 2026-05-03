# Question 4 - Clustering
# CENG 463 Machine Learning Take-Home Midterm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# Step 1: Load dataset
# -----------------------------

print("Loading optdigits-like digits dataset...")

digits = load_digits()

X = digits.data
y = digits.target

print("Dataset loaded successfully.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of classes:", len(np.unique(y)))
print("Classes:", np.unique(y))
print("Feature value range before scaling:", X.min(), X.max())

# Standardize features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature value range after scaling:")
print("Mean approx:", round(X_scaled.mean(), 4))
print("Std approx:", round(X_scaled.std(), 4))

# -----------------------------
# Step 2: K-Means model selection
# -----------------------------

print("\nRunning K-Means model selection...")

k_values = range(2, 16)
kmeans_selection_results = []

for k in k_values:
    start_time = time.time()

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = kmeans.fit_predict(X_scaled)
    runtime = time.time() - start_time

    inertia = kmeans.inertia_
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    kmeans_selection_results.append({
        "k": k,
        "Inertia": inertia,
        "Silhouette": sil,
        "Calinski-Harabasz": ch,
        "Davies-Bouldin": db,
        "Runtime": runtime
    })

kmeans_selection_df = pd.DataFrame(kmeans_selection_results)

print("\nK-Means model selection results:")
print(kmeans_selection_df)

# Save elbow plot
plt.figure(figsize=(7, 5))
plt.plot(kmeans_selection_df["k"], kmeans_selection_df["Inertia"], marker="o")
plt.title("K-Means Elbow Plot")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("q4_kmeans_elbow.png", dpi=300)
plt.close()

# Save silhouette plot
plt.figure(figsize=(7, 5))
plt.plot(kmeans_selection_df["k"], kmeans_selection_df["Silhouette"], marker="o")
plt.title("K-Means Silhouette Scores")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("q4_kmeans_silhouette.png", dpi=300)
plt.close()

# Select k=10 because the dataset contains ten digit classes.
# The model-selection metrics are still reported for comparison.
best_k = 10

print("\nK-Means plots saved:")
print("q4_kmeans_elbow.png")
print("q4_kmeans_silhouette.png")
print("Selected k for final K-Means:", best_k)

# -----------------------------
# Step 3: Final K-Means evaluation
# -----------------------------

print("\nRunning final K-Means evaluation...")

start_time = time.time()

kmeans_final = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=20
)

kmeans_labels = kmeans_final.fit_predict(X_scaled)
kmeans_runtime = time.time() - start_time

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_ch = calinski_harabasz_score(X_scaled, kmeans_labels)
kmeans_db = davies_bouldin_score(X_scaled, kmeans_labels)

kmeans_ari = adjusted_rand_score(y, kmeans_labels)
kmeans_nmi = normalized_mutual_info_score(y, kmeans_labels)
kmeans_fmi = fowlkes_mallows_score(y, kmeans_labels)

print("Final K-Means completed.")
print("Runtime:", round(kmeans_runtime, 4), "seconds")
print("Silhouette:", round(kmeans_silhouette, 4))
print("Calinski-Harabasz:", round(kmeans_ch, 4))
print("Davies-Bouldin:", round(kmeans_db, 4))
print("ARI:", round(kmeans_ari, 4))
print("NMI:", round(kmeans_nmi, 4))
print("Fowlkes-Mallows:", round(kmeans_fmi, 4))

# -----------------------------
# Step 4: GMM model selection with AIC/BIC
# -----------------------------

print("\nRunning GMM model selection...")

gmm_components = range(2, 16)
gmm_selection_results = []

for n in gmm_components:
    start_time = time.time()

    gmm = GaussianMixture(
        n_components=n,
        covariance_type="diag",
        random_state=RANDOM_STATE,
        max_iter=300
    )

    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    runtime = time.time() - start_time

    aic = gmm.aic(X_scaled)
    bic = gmm.bic(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    gmm_selection_results.append({
        "n_components": n,
        "AIC": aic,
        "BIC": bic,
        "Silhouette": sil,
        "Calinski-Harabasz": ch,
        "Davies-Bouldin": db,
        "Runtime": runtime
    })

gmm_selection_df = pd.DataFrame(gmm_selection_results)

print("\nGMM model selection results:")
print(gmm_selection_df)

# Save AIC/BIC plot
plt.figure(figsize=(7, 5))
plt.plot(gmm_selection_df["n_components"], gmm_selection_df["AIC"], marker="o", label="AIC")
plt.plot(gmm_selection_df["n_components"], gmm_selection_df["BIC"], marker="o", label="BIC")
plt.title("GMM AIC/BIC Model Selection")
plt.xlabel("Number of Components")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("q4_gmm_aic_bic.png", dpi=300)
plt.close()

# For fair comparison with true digit classes, use 10 components.
# AIC/BIC results are still reported for model selection discussion.
best_gmm_components = 10

print("\nGMM AIC/BIC plot saved:")
print("q4_gmm_aic_bic.png")
print("Selected components for final GMM:", best_gmm_components)

# -----------------------------
# Step 5: Final GMM evaluation
# -----------------------------

print("\nRunning final GMM evaluation...")

start_time = time.time()

gmm_final = GaussianMixture(
    n_components=best_gmm_components,
    covariance_type="diag",
    random_state=RANDOM_STATE,
    max_iter=300
)

gmm_labels = gmm_final.fit_predict(X_scaled)
gmm_runtime = time.time() - start_time

gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
gmm_ch = calinski_harabasz_score(X_scaled, gmm_labels)
gmm_db = davies_bouldin_score(X_scaled, gmm_labels)

gmm_ari = adjusted_rand_score(y, gmm_labels)
gmm_nmi = normalized_mutual_info_score(y, gmm_labels)
gmm_fmi = fowlkes_mallows_score(y, gmm_labels)

gmm_aic = gmm_final.aic(X_scaled)
gmm_bic = gmm_final.bic(X_scaled)

print("Final GMM completed.")
print("Runtime:", round(gmm_runtime, 4), "seconds")
print("AIC:", round(gmm_aic, 4))
print("BIC:", round(gmm_bic, 4))
print("Silhouette:", round(gmm_silhouette, 4))
print("Calinski-Harabasz:", round(gmm_ch, 4))
print("Davies-Bouldin:", round(gmm_db, 4))
print("ARI:", round(gmm_ari, 4))
print("NMI:", round(gmm_nmi, 4))
print("Fowlkes-Mallows:", round(gmm_fmi, 4))

# -----------------------------
# Step 6: DBSCAN eps selection using k-distance graph
# -----------------------------

print("\nRunning DBSCAN eps selection...")

from sklearn.neighbors import NearestNeighbors

min_samples = 5

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Sort distances to the kth nearest neighbor
k_distances = np.sort(distances[:, min_samples - 1])

plt.figure(figsize=(7, 5))
plt.plot(k_distances)
plt.title("DBSCAN k-distance Graph")
plt.xlabel("Sorted Data Points")
plt.ylabel(f"{min_samples}-Nearest Neighbor Distance")
plt.tight_layout()
plt.savefig("q4_dbscan_kdistance.png", dpi=300)
plt.close()

print("DBSCAN k-distance graph saved as q4_dbscan_kdistance.png")

# Try several eps values around a reasonable range for standardized high-dimensional data
eps_values = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
dbscan_selection_results = []

for eps in eps_values:
    start_time = time.time()

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = dbscan.fit_predict(X_scaled)
    runtime = time.time() - start_time

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.mean(labels == -1)

    # Internal metrics require at least 2 clusters and cannot meaningfully include only noise
    if n_clusters >= 2:
        mask = labels != -1
        if len(np.unique(labels[mask])) >= 2:
            sil = silhouette_score(X_scaled[mask], labels[mask])
            ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
            db = davies_bouldin_score(X_scaled[mask], labels[mask])
        else:
            sil, ch, db = np.nan, np.nan, np.nan
    else:
        sil, ch, db = np.nan, np.nan, np.nan

    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    fmi = fowlkes_mallows_score(y, labels)

    dbscan_selection_results.append({
        "eps": eps,
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "Silhouette": sil,
        "Calinski-Harabasz": ch,
        "Davies-Bouldin": db,
        "ARI": ari,
        "NMI": nmi,
        "Fowlkes-Mallows": fmi,
        "Runtime": runtime
    })

dbscan_selection_df = pd.DataFrame(dbscan_selection_results)

print("\nDBSCAN eps selection results:")
print(dbscan_selection_df)

# Select eps based on balance between number of clusters, noise ratio, and external metrics
best_dbscan_eps = dbscan_selection_df.sort_values(
    by=["ARI", "NMI"],
    ascending=False
).iloc[0]["eps"]

print("\nSelected eps for final DBSCAN:", best_dbscan_eps)
print("Selected min_samples for final DBSCAN:", min_samples)

# -----------------------------
# Step 7: Final DBSCAN evaluation
# -----------------------------

print("\nRunning final DBSCAN evaluation...")

start_time = time.time()

dbscan_final = DBSCAN(
    eps=best_dbscan_eps,
    min_samples=min_samples
)

dbscan_labels = dbscan_final.fit_predict(X_scaled)
dbscan_runtime = time.time() - start_time

dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
dbscan_noise_ratio = np.mean(dbscan_labels == -1)

mask = dbscan_labels != -1

if dbscan_n_clusters >= 2 and len(np.unique(dbscan_labels[mask])) >= 2:
    dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
    dbscan_ch = calinski_harabasz_score(X_scaled[mask], dbscan_labels[mask])
    dbscan_db = davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
else:
    dbscan_silhouette = np.nan
    dbscan_ch = np.nan
    dbscan_db = np.nan

dbscan_ari = adjusted_rand_score(y, dbscan_labels)
dbscan_nmi = normalized_mutual_info_score(y, dbscan_labels)
dbscan_fmi = fowlkes_mallows_score(y, dbscan_labels)

print("Final DBSCAN completed.")
print("Runtime:", round(dbscan_runtime, 4), "seconds")
print("Number of clusters:", dbscan_n_clusters)
print("Noise ratio:", round(dbscan_noise_ratio, 4))
print("Silhouette:", round(dbscan_silhouette, 4))
print("Calinski-Harabasz:", round(dbscan_ch, 4))
print("Davies-Bouldin:", round(dbscan_db, 4))
print("ARI:", round(dbscan_ari, 4))
print("NMI:", round(dbscan_nmi, 4))
print("Fowlkes-Mallows:", round(dbscan_fmi, 4))

# -----------------------------
# Step 8: Agglomerative Clustering with Ward linkage
# -----------------------------

print("\nRunning Agglomerative Clustering...")

from scipy.cluster.hierarchy import dendrogram, linkage

# Dendrogram on a subset for readability
dendrogram_sample_size = 150
dendrogram_indices = np.random.choice(
    X_scaled.shape[0],
    dendrogram_sample_size,
    replace=False
)

X_dendrogram = X_scaled[dendrogram_indices]

Z = linkage(X_dendrogram, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(
    Z,
    truncate_mode="level",
    p=5,
    no_labels=True
)
plt.title("Agglomerative Clustering Dendrogram, Ward Linkage")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("q4_agglomerative_dendrogram.png", dpi=300)
plt.close()

print("Agglomerative dendrogram saved as q4_agglomerative_dendrogram.png")

start_time = time.time()

agg_final = AgglomerativeClustering(
    n_clusters=10,
    linkage="ward"
)

agg_labels = agg_final.fit_predict(X_scaled)
agg_runtime = time.time() - start_time

agg_silhouette = silhouette_score(X_scaled, agg_labels)
agg_ch = calinski_harabasz_score(X_scaled, agg_labels)
agg_db = davies_bouldin_score(X_scaled, agg_labels)

agg_ari = adjusted_rand_score(y, agg_labels)
agg_nmi = normalized_mutual_info_score(y, agg_labels)
agg_fmi = fowlkes_mallows_score(y, agg_labels)

print("Final Agglomerative Clustering completed.")
print("Runtime:", round(agg_runtime, 4), "seconds")
print("Silhouette:", round(agg_silhouette, 4))
print("Calinski-Harabasz:", round(agg_ch, 4))
print("Davies-Bouldin:", round(agg_db, 4))
print("ARI:", round(agg_ari, 4))
print("NMI:", round(agg_nmi, 4))
print("Fowlkes-Mallows:", round(agg_fmi, 4))

# -----------------------------
# Step 9: Final clustering comparison table
# -----------------------------

print("\nCreating final clustering comparison table...")

final_results = pd.DataFrame({
    "Algorithm": [
        "K-Means",
        "GMM",
        "DBSCAN",
        "Agglomerative"
    ],
    "Selected Parameters": [
        f"k={best_k}",
        f"n_components={best_gmm_components}, covariance=diag",
        f"eps={best_dbscan_eps}, min_samples={min_samples}",
        "n_clusters=10, linkage=ward"
    ],
    "Runtime (s)": [
        kmeans_runtime,
        gmm_runtime,
        dbscan_runtime,
        agg_runtime
    ],
    "Number of Clusters": [
        best_k,
        best_gmm_components,
        dbscan_n_clusters,
        10
    ],
    "Noise Ratio": [
        0.0,
        0.0,
        dbscan_noise_ratio,
        0.0
    ],
    "Silhouette": [
        kmeans_silhouette,
        gmm_silhouette,
        dbscan_silhouette,
        agg_silhouette
    ],
    "Calinski-Harabasz": [
        kmeans_ch,
        gmm_ch,
        dbscan_ch,
        agg_ch
    ],
    "Davies-Bouldin": [
        kmeans_db,
        gmm_db,
        dbscan_db,
        agg_db
    ],
    "ARI": [
        kmeans_ari,
        gmm_ari,
        dbscan_ari,
        agg_ari
    ],
    "NMI": [
        kmeans_nmi,
        gmm_nmi,
        dbscan_nmi,
        agg_nmi
    ],
    "Fowlkes-Mallows": [
        kmeans_fmi,
        gmm_fmi,
        dbscan_fmi,
        agg_fmi
    ]
})

print("\nFinal clustering comparison:")
print(final_results)

final_results.to_csv("q4_final_clustering_comparison.csv", index=False)

print("\nFinal clustering comparison saved as q4_final_clustering_comparison.csv")

# -----------------------------
# Step 10: Cluster stability analysis
# -----------------------------

print("\nRunning cluster stability analysis...")

def clustering_stability(model_func, X_data, n_runs=10, sample_fraction=0.8):
    """
    Computes clustering stability using repeated 80% subsampling.
    Stability is measured as ARI between cluster assignments on overlapping samples.
    """
    n_samples = X_data.shape[0]
    subsample_size = int(sample_fraction * n_samples)

    sampled_indices_list = []
    labels_list = []

    for run in range(n_runs):
        indices = np.random.choice(
            n_samples,
            subsample_size,
            replace=False
        )

        X_sub = X_data[indices]
        labels_sub = model_func(X_sub)

        sampled_indices_list.append(indices)
        labels_list.append(labels_sub)

    pairwise_ari_scores = []

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            common_indices = np.intersect1d(
                sampled_indices_list[i],
                sampled_indices_list[j]
            )

            if len(common_indices) > 0:
                pos_i = np.array([
                    np.where(sampled_indices_list[i] == idx)[0][0]
                    for idx in common_indices
                ])
                pos_j = np.array([
                    np.where(sampled_indices_list[j] == idx)[0][0]
                    for idx in common_indices
                ])

                labels_i_common = labels_list[i][pos_i]
                labels_j_common = labels_list[j][pos_j]

                ari = adjusted_rand_score(labels_i_common, labels_j_common)
                pairwise_ari_scores.append(ari)

    return np.mean(pairwise_ari_scores), np.std(pairwise_ari_scores)


def kmeans_model(X_sub):
    return KMeans(
        n_clusters=10,
        random_state=RANDOM_STATE,
        n_init=20
    ).fit_predict(X_sub)


def gmm_model(X_sub):
    return GaussianMixture(
        n_components=10,
        covariance_type="diag",
        random_state=RANDOM_STATE,
        max_iter=300
    ).fit_predict(X_sub)


def dbscan_model(X_sub):
    return DBSCAN(
        eps=best_dbscan_eps,
        min_samples=min_samples
    ).fit_predict(X_sub)


def agglomerative_model(X_sub):
    return AgglomerativeClustering(
        n_clusters=10,
        linkage="ward"
    ).fit_predict(X_sub)


stability_results = []

for name, model_func in [
    ("K-Means", kmeans_model),
    ("GMM", gmm_model),
    ("DBSCAN", dbscan_model),
    ("Agglomerative", agglomerative_model)
]:
    print(f"Computing stability for {name}...")

    mean_stability, std_stability = clustering_stability(
        model_func,
        X_scaled,
        n_runs=10,
        sample_fraction=0.8
    )

    stability_results.append({
        "Algorithm": name,
        "Stability ARI Mean": mean_stability,
        "Stability ARI Std": std_stability
    })

stability_df = pd.DataFrame(stability_results)

print("\nCluster stability results:")
print(stability_df)

stability_df.to_csv("q4_cluster_stability_results.csv", index=False)

print("\nCluster stability results saved as q4_cluster_stability_results.csv")

# -----------------------------
# Step 11: Cluster ensemble using co-association matrix
# -----------------------------

print("\nRunning cluster ensemble using co-association matrix...")

base_clusterings = [
    kmeans_labels,
    gmm_labels,
    dbscan_labels
]

n_samples = X_scaled.shape[0]
coassociation_matrix = np.zeros((n_samples, n_samples))

for labels in base_clusterings:
    for cluster_id in np.unique(labels):
        # Treat DBSCAN noise as its own label for the ensemble
        cluster_indices = np.where(labels == cluster_id)[0]
        coassociation_matrix[np.ix_(cluster_indices, cluster_indices)] += 1

coassociation_matrix = coassociation_matrix / len(base_clusterings)

# Convert similarity to distance
distance_matrix = 1 - coassociation_matrix

start_time = time.time()

ensemble_model = AgglomerativeClustering(
    n_clusters=10,
    metric="precomputed",
    linkage="average"
)

ensemble_labels = ensemble_model.fit_predict(distance_matrix)
ensemble_runtime = time.time() - start_time

ensemble_silhouette = silhouette_score(X_scaled, ensemble_labels)
ensemble_ch = calinski_harabasz_score(X_scaled, ensemble_labels)
ensemble_db = davies_bouldin_score(X_scaled, ensemble_labels)

ensemble_ari = adjusted_rand_score(y, ensemble_labels)
ensemble_nmi = normalized_mutual_info_score(y, ensemble_labels)
ensemble_fmi = fowlkes_mallows_score(y, ensemble_labels)

print("Cluster ensemble completed.")
print("Runtime:", round(ensemble_runtime, 4), "seconds")
print("Silhouette:", round(ensemble_silhouette, 4))
print("Calinski-Harabasz:", round(ensemble_ch, 4))
print("Davies-Bouldin:", round(ensemble_db, 4))
print("ARI:", round(ensemble_ari, 4))
print("NMI:", round(ensemble_nmi, 4))
print("Fowlkes-Mallows:", round(ensemble_fmi, 4))
# -----------------------------
# Step 12: PCA visualization of clustering results
# -----------------------------

print("\nCreating PCA visualizations of clustering results...")

pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_vis = pca_vis.fit_transform(X_scaled)

def save_cluster_plot(labels, title, filename):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca_vis[:, 0],
        X_pca_vis[:, 1],
        c=labels,
        cmap="tab10",
        s=12,
        alpha=0.75
    )
    plt.colorbar(scatter, label="Cluster / Class")
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

save_cluster_plot(y, "PCA Projection with True Digit Labels", "q4_pca_true_labels.png")
save_cluster_plot(kmeans_labels, "PCA Projection with K-Means Clusters", "q4_pca_kmeans_clusters.png")
save_cluster_plot(gmm_labels, "PCA Projection with GMM Clusters", "q4_pca_gmm_clusters.png")
save_cluster_plot(dbscan_labels, "PCA Projection with DBSCAN Clusters", "q4_pca_dbscan_clusters.png")
save_cluster_plot(agg_labels, "PCA Projection with Agglomerative Clusters", "q4_pca_agglomerative_clusters.png")
save_cluster_plot(ensemble_labels, "PCA Projection with Ensemble Clusters", "q4_pca_ensemble_clusters.png")

print("PCA visualization plots saved:")
print("q4_pca_true_labels.png")
print("q4_pca_kmeans_clusters.png")
print("q4_pca_gmm_clusters.png")
print("q4_pca_dbscan_clusters.png")
print("q4_pca_agglomerative_clusters.png")
print("q4_pca_ensemble_clusters.png")

# -----------------------------
# Step 13: Final comparison plots
# -----------------------------

print("\nCreating final comparison plots for Q4...")

# ARI comparison
plt.figure(figsize=(8, 5))
plt.bar(final_results["Algorithm"], final_results["ARI"])
plt.title("ARI Comparison of Clustering Algorithms")
plt.xlabel("Algorithm")
plt.ylabel("Adjusted Rand Index")
plt.tight_layout()
plt.savefig("q4_ari_comparison.png", dpi=300)
plt.close()

# NMI comparison
plt.figure(figsize=(8, 5))
plt.bar(final_results["Algorithm"], final_results["NMI"])
plt.title("NMI Comparison of Clustering Algorithms")
plt.xlabel("Algorithm")
plt.ylabel("Normalized Mutual Information")
plt.tight_layout()
plt.savefig("q4_nmi_comparison.png", dpi=300)
plt.close()

# Stability comparison
plt.figure(figsize=(8, 5))
plt.bar(
    stability_df["Algorithm"],
    stability_df["Stability ARI Mean"],
    yerr=stability_df["Stability ARI Std"],
    capsize=5
)
plt.title("Cluster Stability Comparison")
plt.xlabel("Algorithm")
plt.ylabel("Stability ARI")
plt.tight_layout()
plt.savefig("q4_stability_comparison.png", dpi=300)
plt.close()

print("Final Q4 comparison plots saved:")
print("q4_ari_comparison.png")
print("q4_nmi_comparison.png")
print("q4_stability_comparison.png")