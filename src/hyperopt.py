import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


def _unwrap_model(model):
    return model.named_steps["model"] if isinstance(model, Pipeline) else model


def _set_model_params(estimator, **params):
    if isinstance(estimator, Pipeline):
        return estimator.set_params(**{f"model__{key}": value for key, value in params.items()})
    return estimator.set_params(**params)


def grid_search(model, param_grid, X, y, cv=5, scoring="neg_mean_squared_error"):
    """Run GridSearchCV and log the best estimator to MLflow."""
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=1)
    grid.fit(X, y)
    best = grid.best_estimator_
    with mlflow.start_run(run_name=f"GridSearch_{type(model).__name__}", nested=True):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("grid_best_score", grid.best_score_)
        mlflow.sklearn.log_model(best, f"grid_{type(model).__name__}_model")
    return best


def optuna_study(model, X, y, n_trials=50, cv=5, scoring="neg_mean_squared_error"):
    """Run Optuna on a plain estimator or a Pipeline with final step ``model``."""

    def objective(trial):
        base_model = _unwrap_model(model)

        if isinstance(base_model, (Ridge, Lasso)):
            estimator = _set_model_params(clone(model), alpha=trial.suggest_float("alpha", 1e-4, 10.0, log=True))
        elif isinstance(base_model, SVR):
            estimator = _set_model_params(
                clone(model),
                C=trial.suggest_float("C", 0.1, 10.0, log=True),
                gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
                kernel=trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
            )
        elif isinstance(base_model, RandomForestRegressor):
            estimator = _set_model_params(
                clone(model),
                n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
                max_depth=trial.suggest_int("max_depth", 5, 30),
            )
        elif isinstance(base_model, MLPRegressor):
            _layer_options = [(50,), (100, 50), (200, 100)]
            layer_idx = trial.suggest_int("hidden_layer_sizes_idx", 0, len(_layer_options) - 1)
            estimator = _set_model_params(
                clone(model),
                hidden_layer_sizes=_layer_options[layer_idx],
                alpha=trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            )
        else:
            estimator = clone(model)

        scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_estimator = _set_model_params(clone(model), **best_params)
    best_estimator.fit(X, y)

    with mlflow.start_run(run_name=f"Optuna_{type(model).__name__}", nested=True):
        mlflow.log_params(best_params)
        mlflow.log_metric("optuna_best_score", study.best_value)
        mlflow.sklearn.log_model(best_estimator, f"optuna_{type(model).__name__}_model")

    return best_estimator
