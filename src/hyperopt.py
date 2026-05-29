import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _unwrap_model(model):
    """
    Utility function to extract the core estimator if it is wrapped inside a scikit-learn Pipeline.
    This ensures we can identify the model type regardless of preprocessing steps.
    """
    return model.named_steps["model"] if isinstance(model, Pipeline) else model


def _set_model_params(estimator, **params):
    """
    Dynamically sets hyperparameters. If the estimator is a Pipeline, 
    it prefixes the parameter keys with 'model__' to correctly target the final step.
    """
    if isinstance(estimator, Pipeline):
        return estimator.set_params(**{f"model__{key}": value for key, value in params.items()})
    return estimator.set_params(**params)


def _model_name(model):
    """Returns the class name of the base model for logging purposes."""
    base_model = _unwrap_model(model)
    return type(base_model).__name__


def _suggest_params(trial, base_model):
    """
    Defines the hyperparameter search space for Optuna trials based on the specific model architecture.
    This fulfills the requirement to perform Hyperparameter optimization (Optuna).
    """
    if isinstance(base_model, LinearRegression):
        return {"fit_intercept": trial.suggest_categorical("fit_intercept", [True, False])}

    # Addresses the requirement: "Ridge and lasso regression (including alpha analysis)"
    if isinstance(base_model, (Ridge, Lasso)):
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }

    if isinstance(base_model, LogisticRegression):
        return {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True), # Inverse of regularization strength
            "solver": "lbfgs",
            "penalty": "l2",
            "max_iter": 500, # Increased to ensure convergence
        }

    if isinstance(base_model, GaussianNB):
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)}

    # Addresses the requirement: "SVM including the analysis of multiple kernels"
    if isinstance(base_model, (SVR, SVC)):
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        params = {
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": kernel,
        }
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)
        return params

    if isinstance(base_model, (KNeighborsRegressor, KNeighborsClassifier)):
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 31, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2), # 1: Manhattan distance, 2: Euclidean
        }

    if isinstance(base_model, (DecisionTreeRegressor, DecisionTreeClassifier)):
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

    if isinstance(base_model, (RandomForestRegressor, RandomForestClassifier)):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    if isinstance(base_model, (GradientBoostingRegressor, GradientBoostingClassifier)):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0), # Stochastic gradient boosting
        }

    # Addresses the requirement: "Neural Networks: single layer and multi layer"
    if isinstance(base_model, (MLPRegressor, MLPClassifier)):
        layer_options = [(50,), (100, 50), (200, 100)] # Defines both single and multi-layer architectures
        layer_idx = trial.suggest_int("hidden_layer_sizes_idx", 0, len(layer_options) - 1)
        return {
            "hidden_layer_sizes": layer_options[layer_idx],
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True), # L2 penalty
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }

    return {}


def _greedy_param_grid(base_model):
    """
    Defines discrete parameter grids for the greedy search algorithm.
    This serves as an alternative to GridSearch/Optuna for faster, coordinate-wise optimization.
    """
    if isinstance(base_model, LinearRegression):
        return {"fit_intercept": [True, False]}

    if isinstance(base_model, (Ridge, Lasso)):
        return {"alpha": [0.001, 0.01, 0.1, 1, 10, 100], "fit_intercept": [True, False]}

    if isinstance(base_model, LogisticRegression):
        return {"C": [0.01, 0.1, 1, 10, 100]}

    if isinstance(base_model, GaussianNB):
        return {"var_smoothing": [1e-12, 1e-10, 1e-9, 1e-8, 1e-6]}

    if isinstance(base_model, (SVR, SVC)):
        return {"kernel": ["linear", "rbf", "poly"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"], "degree": [2, 3, 4]}

    if isinstance(base_model, (KNeighborsRegressor, KNeighborsClassifier)):
        return {"n_neighbors": list(range(3, 32, 2)), "weights": ["uniform", "distance"], "p": [1, 2]}

    if isinstance(base_model, (DecisionTreeRegressor, DecisionTreeClassifier)):
        return {"max_depth": [3, 5, 10, 20, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 5]}

    if isinstance(base_model, (RandomForestRegressor, RandomForestClassifier)):
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", None],
        }

    if isinstance(base_model, (GradientBoostingRegressor, GradientBoostingClassifier)):
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 5],
            "subsample": [0.7, 0.85, 1.0],
        }

    if isinstance(base_model, (MLPRegressor, MLPClassifier)):
        return {
            "hidden_layer_sizes": [(50,), (100, 50), (200, 100)],
            "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate_init": [1e-4, 1e-3, 1e-2],
        }

    return {}


def _mean_cv_score(estimator, X, y, cv, scoring):
    """
    Computes the mean cross-validation score robustly.
    Fulfills the project requirement: "Perform a Cross Validation".
    """
    try:
        # n_jobs=1 prevents multi-processing conflicts within nested optimization loops
        scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score=np.nan)
    except Exception:
        return -np.inf

    score = np.nanmean(scores)
    if np.isnan(score):
        return -np.inf
    return float(score)


def _mlflow_safe_params(params):
    """
    Sanitizes hyperparameters for MLflow logging, converting complex objects (like lists or None) to strings.
    """
    safe_params = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple, dict)):
            safe_params[key] = str(value)
        elif value is None:
            safe_params[key] = "None"
        else:
            safe_params[key] = value
    return safe_params


def grid_search(model, param_grid, X, y, cv=5, scoring="neg_mean_squared_error"):
    """
    Executes standard GridSearchCV. 
    This addresses the requirement: "Hyperparameters optimization (gridsearch)".
    Logs the parameters, metrics, and serialized model artifact to MLflow.
    """
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=1)
    grid.fit(X, y)
    best = grid.best_estimator_
    
    # Context manager ensures runs are properly closed. Nested=True allows organizing runs under a parent experiment.
    with mlflow.start_run(run_name=f"GridSearch_{type(model).__name__}", nested=True):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("grid_best_score", grid.best_score_)
        mlflow.sklearn.log_model(best, f"grid_{type(model).__name__}_model")
    return best


def greedy_search(model, X, y, cv=5, scoring="neg_mean_squared_error"):
    """
    Executes a custom coordinate-wise greedy search for hyperparameter tuning.
    Optimizes one parameter at a time while holding others constant.
    Logs the experiment tracking data via MLflow.
    """
    base_model = _unwrap_model(model)
    param_grid = _greedy_param_grid(base_model)
    best_params = {}
    best_score = -np.inf

    if not param_grid:
        best_estimator = clone(model)
        best_estimator.fit(X, y)
        return best_estimator, {"best_score": best_score, "best_params": best_params}

    for param_name, values in param_grid.items():
        local_best_value = None
        local_best_score = -np.inf

        for value in values:
            candidate_params = {**best_params, param_name: value}
            estimator = _set_model_params(clone(model), **candidate_params)
            score = _mean_cv_score(estimator, X, y, cv, scoring)

            if score > local_best_score:
                local_best_score = score
                local_best_value = value

        if local_best_value is not None:
            best_params[param_name] = local_best_value
            best_score = local_best_score

    best_estimator = _set_model_params(clone(model), **best_params)
    best_estimator.fit(X, y)

    # Manage model versions with MLflow
    with mlflow.start_run(run_name=f"Greedy_{_model_name(model)}", nested=True):
        mlflow.log_params(_mlflow_safe_params(best_params))
        mlflow.log_metric("greedy_best_score", best_score)
        mlflow.sklearn.log_model(best_estimator, f"greedy_{_model_name(model)}_model")

    return best_estimator, {"best_score": best_score, "best_params": best_params}


def optuna_study(model, X, y, n_trials=50, cv=5, scoring="neg_mean_squared_error"):
    """
    Executes Optuna hyperparameter optimization.
    This fulfills the requirement: "Hyperparameters optimization (optuna)".
    Uses a Bayesian optimization approach to maximize the CV score.
    """

    def objective(trial):
        base_model = _unwrap_model(model)
        params = _suggest_params(trial, base_model)
        # Store params in the trial object to retrieve later if this trial is the best
        trial.set_user_attr("estimator_params", params)
        
        estimator = _set_model_params(clone(model), **params) if params else clone(model)
        return _mean_cv_score(estimator, X, y, cv, scoring)

    # We maximize because scikit-learn's "neg_" scoring metrics return negative values 
    # where closer to 0 is better.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.user_attrs.get("estimator_params", study.best_params)
    best_estimator = _set_model_params(clone(model), **best_params)
    
    # Retrain on the full provided dataset with the optimal parameters found
    best_estimator.fit(X, y)

    # Manage model versions with MLflow
    with mlflow.start_run(run_name=f"Optuna_{_model_name(model)}", nested=True):
        mlflow.log_params(_mlflow_safe_params(best_params))
        mlflow.log_metric("optuna_best_score", study.best_value)
        mlflow.sklearn.log_model(best_estimator, f"optuna_{_model_name(model)}_model")

    return best_estimator, {"best_score": study.best_value, "best_params": best_params}