import numpy as np
from sklearn.decomposition import TruncatedSVD


def apply_pca(X, variance=0.95, max_components=300, random_state=42):
    """
    Applies Principal Component Analysis (PCA) using Truncated Singular Value Decomposition (SVD).
    
    This function explicitly fulfills the project requirement to use dimensionality 
    reduction based on SVD[cite: 65]. It employs a dynamic, two-pass approach: 
    first, it estimates the cumulative explained variance across a broad number 
    of components, and then it refits the model using the optimal, minimum number 
    of components required to retain the requested variance (e.g., 95%).

    Parameters
    ----------
    X : array-like or sparse matrix
        The feature matrix to be reduced.
    variance : float, default=0.95
        The target cumulative explained variance ratio (must be between 0 and 1).
    max_components : int, default=300
        The upper limit of components to test during the probing phase.
    random_state : int, default=42
        Ensures reproducibility of the scientific results.
    """
    if not 0 < variance <= 1:
        raise ValueError("Variance threshold must be strictly between 0 and 1.")

    # Step 1: Establish the maximum valid components constraint.
    # SVD strictly requires the number of components to be strictly less than 
    # the number of features and the number of samples.
    max_valid_components = max(2, min(X.shape[0] - 1, X.shape[1] - 1, max_components))
    
    # Step 2: The Probing Phase
    # Fit an initial SVD model to evaluate the variance explained by the maximum valid components.
    probe = TruncatedSVD(n_components=max_valid_components, random_state=random_state)
    probe.fit(X)

    # Step 3: Variance Analysis
    # Calculate the cumulative sum of the explained variance ratio.
    cumulative_variance = np.cumsum(probe.explained_variance_ratio_)
    
    # Identify the exact index where the cumulative variance meets or exceeds the target threshold.
    # np.searchsorted is highly efficient for finding this optimal k-value.
    selected_components = int(np.searchsorted(cumulative_variance, variance) + 1)
    
    # Ensure the selected components do not exceed the mathematical limits of the dataset.
    selected_components = min(selected_components, max_valid_components)

    # Step 4: Final Dimensionality Reduction
    # Instantiate and fit the final SVD model with the optimized number of components.
    svd = TruncatedSVD(n_components=selected_components, random_state=random_state)
    X_reduced = svd.fit_transform(X)
    
    # Returning both the reduced matrix and the SVD object allows downstream 
    # scripts to apply the exact same transformation to test/validation sets.
    return X_reduced, svd