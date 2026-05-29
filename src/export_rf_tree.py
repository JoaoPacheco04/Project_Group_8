# src/export_rf_tree.py
"""
Utility to export a single decision tree from a trained RandomForest model.

This script directly fulfills the project requirement for Ensemble Methods, 
which mandates that an image of the tree must be provided when using a Random Forest.
The tree is saved as a highly legible PNG for inclusion in the scientific paper 
and dashboard. A .dot file is also generated for advanced Graphviz rendering if needed.

Usage example (run after the model has been trained and saved as `rf_model.pkl`):

    import joblib
    from src.export_rf_tree import export_random_forest_tree

    rf = joblib.load('artifacts/rf_model.pkl')
    export_random_forest_tree(rf, 'rf_tree.dot', 'rf_tree.png')
"""

import os
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz, plot_tree


def export_random_forest_tree(model, dot_path: str, png_path: str = None, feature_names=None) -> None:
    """
    Exports a representative decision tree from a trained Random Forest model.

    Parameters
    ----------
    model: RandomForestRegressor or RandomForestClassifier
        The trained Random Forest ensemble model.
    dot_path: str
        Destination file path for the .dot file (Graphviz format).
    png_path: str, optional
        Destination file path for the .png image. Uses `sklearn.tree.plot_tree` 
        to ensure a standalone, easily exportable image for the final report.
    feature_names: iterable, optional
        Names of the features used during training, vital for tree interpretability.
    """
    # Verify that the model is indeed an ensemble containing multiple trees
    if not hasattr(model, "estimators_"):
        raise AttributeError("The provided model does not contain 'estimators_'.")

    # Extract the first tree (index 0) from the forest to serve as a representative visual.
    # Analyzing a single tree helps demystify the "black box" nature of ensemble methods.
    tree = model.estimators_[0]

    # Ensure the output directory for the .dot file exists
    dot_dir = os.path.dirname(dot_path)
    if dot_dir:
        os.makedirs(dot_dir, exist_ok=True)

    # Export the raw structure to a Graphviz .dot file for record-keeping or alternate rendering
    export_graphviz(
        tree,
        out_file=dot_path,
        feature_names=feature_names if feature_names is not None else getattr(model, "feature_names_in", None),
        filled=True,
        rounded=True,
        impurity=False, # Set to True if you need to analyze Gini/Entropy explicitly
        special_characters=True,
    )

    # Generate and save the PNG image required for the scientific paper
    if png_path:
        png_dir = os.path.dirname(png_path)
        if png_dir:
            os.makedirs(png_dir, exist_ok=True)

        # Set a large figure size to accommodate the breadth of the tree branches
        plt.figure(figsize=(28, 14))
        
        # Plot the tree. max_depth=3 is strictly enforced to maintain legibility 
        # in the final document, avoiding visual clutter from deep splits.
        plot_tree(
            tree,
            feature_names=feature_names if feature_names is not None else getattr(model, "feature_names_in_", None),
            filled=True,
            rounded=True,
            max_depth=3, 
            fontsize=7,
        )
        
        plt.title("Random Forest — Representative Tree (First Estimator)", fontsize=14)
        plt.tight_layout()
        
        # Save at 180 DPI for high-quality, publication-ready resolution
        plt.savefig(png_path, dpi=180)
        plt.close()
        print(f"Random Forest tree exported to {png_path}")
    else:
        print(f"Random Forest tree exported to {dot_path}")