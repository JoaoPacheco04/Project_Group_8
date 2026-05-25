# src/export_rf_tree.py
"""Utility to export a single decision tree from a trained RandomForestRegressor.
The tree is saved as a PNG using ``sklearn.tree.plot_tree`` (no external
Graphviz CLI required).  A ``.dot`` file is also written via
``sklearn.tree.export_graphviz`` for reference.

Usage example (run after the model has been trained and saved as ``rf_model.pkl``)::

    import joblib
    from src.export_rf_tree import export_random_forest_tree

    rf = joblib.load('artifacts/rf_model.pkl')
    export_random_forest_tree(rf, 'rf_tree.dot', 'rf_tree.png')
"""

import os

import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz, plot_tree


def export_random_forest_tree(model, dot_path: str, png_path: str = None, feature_names=None) -> None:
    """Export the first tree of a RandomForestRegressor.

    Parameters
    ----------
    model: RandomForestRegressor
        Trained Random Forest model.
    dot_path: str
        Destination path for the ``.dot`` file.
    png_path: str, optional
        If provided, the tree will be rendered to a PNG using
        ``sklearn.tree.plot_tree`` (no Graphviz CLI needed).
    feature_names: iterable, optional
        Names of the transformed features used by the forest.
    """
    if not hasattr(model, "estimators_"):
        raise AttributeError("The provided model does not contain 'estimators_'.")

    # Take the first tree (any tree works for illustration)
    tree = model.estimators_[0]

    # Ensure output directory exists
    dot_dir = os.path.dirname(dot_path)
    if dot_dir:
        os.makedirs(dot_dir, exist_ok=True)

    export_graphviz(
        tree,
        out_file=dot_path,
        feature_names=feature_names if feature_names is not None else getattr(model, "feature_names_in", None),
        filled=True,
        rounded=True,
        impurity=False,
        special_characters=True,
    )

    if png_path:
        png_dir = os.path.dirname(png_path)
        if png_dir:
            os.makedirs(png_dir, exist_ok=True)

        plt.figure(figsize=(28, 14))
        plot_tree(
            tree,
            feature_names=feature_names if feature_names is not None else getattr(model, "feature_names_in_", None),
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=7,
        )
        plt.title("Random Forest — Representative Tree (first estimator)", fontsize=14)
        plt.tight_layout()
        plt.savefig(png_path, dpi=180)
        plt.close()
        print(f"Random Forest tree exported to {png_path}")
    else:
        print(f"Random Forest tree exported to {dot_path}")
