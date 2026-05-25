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
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def get_models(problem_type: str = "regression"):
    """Return the estimators required by the assignment."""
    if problem_type not in {"regression", "classification"}:
        raise ValueError("Problem type must be either 'regression' or 'classification'.")

    if problem_type == "regression":
        return {
            "Linear": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "SVR_linear": SVR(kernel="linear"),
            "SVR_rbf": SVR(kernel="rbf"),
            "SVR_poly": SVR(kernel="poly", degree=3),
            "KNN": KNeighborsRegressor(n_jobs=-1),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "RandomForest": RandomForestRegressor(n_jobs=-1, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "MLP_single": MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
            "MLP_multi": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        }

    return {
        "LogisticRegression": LogisticRegression(max_iter=500, n_jobs=-1),
        "NaiveBayes": GaussianNB(),
        "SVC_linear": SVC(kernel="linear", random_state=42),
        "SVC_rbf": SVC(kernel="rbf", random_state=42),
        "SVC_poly": SVC(kernel="poly", degree=3, random_state=42),
        "KNNClassifier": KNeighborsClassifier(n_jobs=-1),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            random_state=42,
        ),
        "RandomForestClassifier": RandomForestClassifier(n_jobs=-1, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "MLPClassifier_single": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
        "MLPClassifier_multi": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }
