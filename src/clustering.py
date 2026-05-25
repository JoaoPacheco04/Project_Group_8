import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


def _as_array(X):
    return X.to_numpy() if isinstance(X, pd.DataFrame) else X


def analyze_optimal_k(X, min_k: int = 2, max_k: int = 10):
    """Return elbow and silhouette analysis for a K range."""
    X_values = _as_array(X)
    rows = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_values)
        rows.append(
            {
                "k": k,
                "inertia": kmeans.inertia_,
                "silhouette": silhouette_score(X_values, labels),
            }
        )
    results = pd.DataFrame(rows)
    best_k = int(results.loc[results["silhouette"].idxmax(), "k"])
    return results, best_k


def optimal_k_elbow(X, max_k: int = 10, output_path: str | None = None):
    """Compute SSE and silhouette and plot both metrics."""
    results, best_k = analyze_optimal_k(X, min_k=2, max_k=max_k)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results["k"], results["inertia"], marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Inertia")
    plt.xticks(results["k"])

    plt.subplot(1, 2, 2)
    plt.plot(results["k"], results["silhouette"], marker="o")
    plt.title("Silhouette Score by K")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Silhouette score")
    plt.xticks(results["k"])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=180)
    else:
        plt.show()
    plt.close()
    return results, best_k


def plot_dendrogram(X, method: str = "ward", sample_size: int = 1000, output_path: str | None = None):
    """Generate and display a dendrogram using a manageable sample."""
    X_values = _as_array(X)
    if len(X_values) > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_values), size=sample_size, replace=False)
        X_values = X_values[indices]

    linked = linkage(X_values, method=method)
    plt.figure(figsize=(12, 6))
    dendrogram(linked, truncate_mode="lastp", p=20, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Cluster/sample index")
    plt.ylabel("Distance")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=180)
    else:
        plt.show()
    plt.close()


def run_kmeans(X, n_clusters: int = 3):
    """Fit K-Means and return labels and inertia."""
    X_values = _as_array(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_values)
    return labels, kmeans.inertia_


def run_hierarchical(X, n_clusters: int = 3, linkage_method: str = "ward"):
    """Fit Agglomerative Clustering and return labels."""
    X_values = _as_array(X)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = agg.fit_predict(X_values)
    return labels
