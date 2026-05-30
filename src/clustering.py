import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


def _as_array(X):
    """
    Helper function to ensure the input is a NumPy array.
    Many scikit-learn models and SciPy functions expect arrays rather than pandas DataFrames.
    """
    return X.to_numpy() if isinstance(X, pd.DataFrame) else X


def analyze_optimal_k(X, min_k: int = 2, max_k: int = 10):
    """
    Calculates inertia (SSE) and silhouette scores for a given range of K values.
    This fulfills the requirement to analyze the optimal number of clusters.
    Silhouette scores are computed on a stratified sample for performance.
    """
    X_values = _as_array(X)
    rows = []
    
    # Use a sample for silhouette calculation to reduce computation time
    # stratified by the full data distribution
    sample_size = min(2000, max(1000, len(X_values) // 5))
    sample_indices = np.random.RandomState(42).choice(len(X_values), size=sample_size, replace=False)
    X_sample = X_values[sample_indices]
    
    # Iterate over the specified range of K to evaluate clustering performance
    for k in range(min_k, max_k + 1):
        # Initialize KMeans with a fixed random_state for reproducible scientific results
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_full = kmeans.fit_predict(X_values)
        
        # Compute silhouette on sample for speed, but use labels from full fit
        labels_sample = kmeans.predict(X_sample)
        
        rows.append(
            {
                "k": k,
                "inertia": kmeans.inertia_, # Measures how tight the clusters are (used for Elbow Method)
                "silhouette": silhouette_score(X_sample, labels_sample), # Measures cluster cohesion and separation (on sample)
            }
        )
        
    results = pd.DataFrame(rows)
    # Identify the best K based on the highest silhouette score
    best_k = int(results.loc[results["silhouette"].idxmax(), "k"])
    return results, best_k


def optimal_k_elbow(X, max_k: int = 10, output_path: str | None = None):
    """
    Generates and optionally saves visualizations for the Elbow Method and Silhouette Score.
    These graphs provide a visual justification for the chosen number of clusters in the scientific paper.
    """
    results, best_k = analyze_optimal_k(X, min_k=2, max_k=max_k)

    plt.figure(figsize=(12, 5))

    # Plot 1: Elbow Method (Inertia/SSE)
    plt.subplot(1, 2, 1)
    plt.plot(results["k"], results["inertia"], marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Inertia")
    plt.xticks(results["k"])

    # Plot 2: Silhouette Score Analysis
    plt.subplot(1, 2, 2)
    plt.plot(results["k"], results["silhouette"], marker="o")
    plt.title("Silhouette Score by K")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Silhouette score")
    plt.xticks(results["k"])

    plt.tight_layout()
    
    # Export the plot for the final report/dashboard or display it in the notebook
    if output_path:
        plt.savefig(output_path, dpi=180)
    else:
        plt.show()
    plt.close()
    return results, best_k


def plot_dendrogram(X, method: str = "ward", sample_size: int = 1000, output_path: str | None = None):
    """
    Generates a hierarchical clustering dendrogram.
    To prevent memory overloads and ensure a legible graph, it downsamples large datasets.
    """
    X_values = _as_array(X)
    
    # Downsample if the dataset is too large, ensuring a readable dendrogram
    if len(X_values) > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_values), size=sample_size, replace=False)
        X_values = X_values[indices]

    # Perform hierarchical/agglomerative clustering using the specified linkage method
    linked = linkage(X_values, method=method)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the dendrogram, truncating the tree for better visibility of the top-level branches
    dendrogram(linked, truncate_mode="lastp", p=20, leaf_rotation=90)
    
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Cluster/sample index")
    plt.ylabel("Distance")
    plt.tight_layout()
    
    # Save or show the figure
    if output_path:
        plt.savefig(output_path, dpi=180)
    else:
        plt.show()
    plt.close()


def run_kmeans(X, n_clusters: int = 3):
    """
    Fits the K-Means algorithm to the dataset using the identified optimal number of clusters.
    """
    X_values = _as_array(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_values)
    return labels, kmeans.inertia_


def run_hierarchical(X, n_clusters: int = 3, linkage_method: str = "ward"):
    """
    Fits an Agglomerative (Hierarchical) Clustering model to the dataset.
    """
    X_values = _as_array(X)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = agg.fit_predict(X_values)
    return labels