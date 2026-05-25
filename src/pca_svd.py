import numpy as np
from sklearn.decomposition import TruncatedSVD


def apply_pca(X, variance=0.95, max_components=300, random_state=42):
    """Apply dimensionality reduction with TruncatedSVD.

    The function first fits a broad SVD to estimate cumulative explained
    variance, then refits using the smallest number of components that reaches
    the requested variance threshold.
    """
    if not 0 < variance <= 1:
        raise ValueError("variance must be between 0 and 1.")

    max_valid_components = max(2, min(X.shape[0] - 1, X.shape[1] - 1, max_components))
    probe = TruncatedSVD(n_components=max_valid_components, random_state=random_state)
    probe.fit(X)

    cumulative_variance = np.cumsum(probe.explained_variance_ratio_)
    selected_components = int(np.searchsorted(cumulative_variance, variance) + 1)
    selected_components = min(selected_components, max_valid_components)

    svd = TruncatedSVD(n_components=selected_components, random_state=random_state)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd
