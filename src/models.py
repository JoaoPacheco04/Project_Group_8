from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline # Note: You might need this if using scaling later
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def get_models(problem_type: str = "regression"):
    """
    Returns a comprehensive dictionary of uninstantiated or baseline models 
    required for the assignment's comparative analysis.

    Parameters
    ----------
    problem_type : str
        Either 'regression' or 'classification', defining the final goal of the dataset.
    """
    if problem_type not in {"regression", "classification"}:
        raise ValueError("Problem type must be either 'regression' or 'classification'.")

    # ==========================================
    # REGRESSION MODELS
    # ==========================================
    if problem_type == "regression":
        return {
            # Standard linear models
            "Linear": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(alpha=0.001), # Alpha analysis is conducted during hyperparameter tuning
            
            # Support Vector Machines - explicitly testing multiple kernels as required
            "SVR_linear": SVR(kernel="linear", max_iter=500),
            "SVR_rbf": SVR(kernel="rbf", max_iter=500),
            "SVR_poly": SVR(kernel="poly", degree=3, max_iter=5000, cache_size=1000, tol=1e-2),
            
            # Instance-based learning
            "KNN": KNeighborsRegressor(n_jobs=-1),
            
            # Tree-based and Ensemble methods
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "RandomForest": RandomForestRegressor(n_jobs=-1, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            
            # Neural Networks - explicitly satisfying the single layer vs multi-layer requirement
            "MLP_single": MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
            "MLP_multi": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        }

    # ==========================================
    # CLASSIFICATION MODELS
    # ==========================================
    return {
        "LogisticRegression": LogisticRegression(max_iter=500, n_jobs=-1, class_weight="balanced"),
        "NaiveBayes": GaussianNB(),
        
        # Support Vector Classifiers with multiple kernels
        "SVC_linear": SVC(kernel="linear", random_state=42, class_weight="balanced"),
        "SVC_rbf": SVC(kernel="rbf", random_state=42, class_weight="balanced"),
        "SVC_poly": SVC(kernel="poly", degree=3, random_state=42, class_weight="balanced"),
        
        "KNNClassifier": KNeighborsClassifier(n_jobs=-1),
        
        # Decision Tree configured specifically to analyze the Gini impurity metric
        # max_depth is constrained to ensure the exported tree graphic remains legible
        "DecisionTreeClassifier": DecisionTreeClassifier(
            criterion="gini",
            max_depth=5, 
            random_state=42,
        ),
        
        "RandomForestClassifier": RandomForestClassifier(n_jobs=-1, random_state=42, class_weight="balanced"),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        
        # Single and multi-layer Perceptron classifiers
        "MLPClassifier_single": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
        "MLPClassifier_multi": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }


def get_models_for_svd(problem_type: str = "regression"):
    """
    Returns models specifically optimized for the SVD-reduced feature space.
    
    This function addresses the requirement to apply Dimensionality Reduction based on SVD 
    and compare the resulting accuracy and execution time. Because SVD transforms sparse data 
    into dense principal components, distance-based models like SVMs often require adjusted 
    iteration budgets to prevent premature convergence collapse.
    """
    models = get_models(problem_type)

    if problem_type == "regression":
        # Remove polynomial SVR for SVD pipelines as it often fails to converge 
        # in a reasonable timeframe on dense, reduced datasets.
        models.pop("SVR_poly", None)
        
        # Increase the max iteration budget for linear and RBF kernels to adapt 
        # to the new, dense feature distribution created by SVD.
        models["SVR_linear"] = SVR(kernel="linear", max_iter=2000)
        models["SVR_rbf"] = SVR(kernel="rbf", max_iter=2000)

    return models