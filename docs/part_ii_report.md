# Part II - Machine Learning Report

## Dataset Feature To Learn

The main supervised learning target is `score` from `datasets/ratings.csv`. This is the numeric anime rating assigned by users and is suitable for a regression task because the objective is to estimate a continuous value.

A secondary classification target is derived from the same feature:

```text
high_score = 1 if score >= 7 else 0
```

This derived target allows the project to apply classification algorithms such as Logistic Regression and Naive Bayes while keeping the learning problem connected to the original dataset.

## Final Goal

The primary goal is prediction: estimate the expected anime score from available features such as user/anime identifiers and engineered numeric/categorical variables.

The secondary goal is classification: identify whether an anime-user rating belongs to the high-score class (`score >= 7`) or lower-score class (`score < 7`).

## Models Applied

The model registry is implemented in `src/models.py`.

Regression models used for score prediction:

- Linear Regression
- Ridge Regression
- Lasso Regression
- SVM/SVR with linear, RBF, and polynomial kernels. Polynomial SVR is kept in the baseline benchmark to satisfy the multiple-kernel analysis, but excluded from the SVD comparison if it becomes numerically unstable.
- K-NN Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Neural Network with one hidden layer
- Neural Network with multiple hidden layers

Classification models used for the high-score task:

- Logistic Regression
- Naive Bayes
- SVM/SVC with linear, RBF, and polynomial kernels
- K-NN Classifier
- Decision Tree Classifier using Gini impurity
- Random Forest Classifier
- Gradient Boosting Classifier
- Neural Network with one hidden layer
- Neural Network with multiple hidden layers

## Comparison Metrics

Regression models are compared using:

- RMSE
- MAE
- R2
- Fit time in seconds

Classification models are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Fit time in seconds

The main generated artifacts are:

- `artifacts/baseline_results.json`
- `artifacts/cv_results.csv`
- `artifacts/pca_results.json`
- `artifacts/classification_results.json`
- `reports/baseline_rmse.png`
- `reports/baseline_fit_times.png`
- `reports/cross_validation_rmse.png`
- `reports/classification_metrics.png`
- `reports/classification_fit_times.png`

## Ridge And Lasso Alpha Analysis

Ridge and Lasso are evaluated with multiple alpha values through GridSearchCV and a manual alpha sweep. The goal is to identify how regularization strength affects RMSE.

Generated artifacts:

- `artifacts/alpha_analysis.csv`
- `reports/ridge_alpha_analysis.png`
- `reports/lasso_alpha_analysis.png`

## SVM Kernel Analysis

SVM is evaluated with multiple kernels:

- Linear
- RBF
- Polynomial

For regression, these are implemented as `SVR_linear`, `SVR_rbf`, and `SVR_poly`. If polynomial SVR produces unstable predictions outside the natural 0-10 score range after SVD, it is excluded only from the SVD plots and interpreted as evidence that kernel choice affects both performance and numerical stability. For classification, these are implemented as `SVC_linear`, `SVC_rbf`, and `SVC_poly`.

The comparison shows whether a linear boundary is sufficient or whether nonlinear kernels better capture the relation between dataset features and score behavior.

## K-NN Optimal Number Of Neighbours

The optimal number of neighbours is selected with GridSearchCV over odd values from 3 to 19.

Generated artifact:

- `artifacts/knn_best_result.json`

The selected `n_neighbors` is the value that minimizes validation RMSE.

## Decision Tree And Gini Analysis

The classification Decision Tree uses `criterion="gini"`. The Gini impurity at the root node and the final tree depth/leaves are saved for analysis.

Generated artifacts:

- `reports/decision_tree_gini.png`
- `artifacts/decision_tree_gini_summary.json`

The Gini value measures class impurity. A value closer to zero means the node is purer; a larger value means the classes are more mixed.

## Ensemble Methods

The ensemble methods include:

- Random Forest
- Gradient Boosting

For Random Forest, the project exports one representative tree from the forest as an image.

Generated artifacts:

- `reports/rf_tree.dot`
- `reports/rf_tree.png`

## Fit-Time Analysis

Fit time is measured for each baseline regression model and each classification model. This allows the project to compare predictive performance against computational cost.

Generated artifacts:

- `reports/baseline_fit_times.png`
- `reports/classification_fit_times.png`
- `reports/pca_fit_times.png`

## Hyperparameter Optimization

Two optimization approaches are used:

- GridSearchCV for Ridge, Lasso, K-NN regressor, and K-NN classifier
- Optuna for Random Forest, SVR, and the multi-layer neural network

The implementation is in `src/hyperopt.py`.

The K-NN classifier grid compares odd `n_neighbors` values from 3 to 21 and both `uniform` and `distance` weighting. The selected classifier parameters and metrics are saved in `artifacts/knn_classifier_best_result.json`.

## MLflow Version Management

MLflow is used to track:

- Model parameters
- Model metrics
- Fit times
- Trained model artifacts
- Generated plots
- Recommendation run artifacts

The experiment name is `Projeto_Grupo_8`. Local tracking files are stored in `mlflow.db` and `mlruns/`.

## PCA / SVD Dimensionality Reduction

Dimensionality reduction is applied using `TruncatedSVD`, which is appropriate after one-hot encoding because the transformed matrix can be sparse.

The function is implemented in `src/pca_svd.py`.

Generated artifacts:

- `artifacts/pca_results.json`
- `artifacts/pca_summary.json`
- `reports/pca_rmse.png`
- `reports/pca_fit_times.png`

The comparison checks whether SVD reduces training time while preserving acceptable RMSE.

## Clustering Analysis

Clustering is applied to the SVD-reduced feature representation. The project uses:

- K-Means clustering
- Hierarchical clustering with Ward linkage

The optimal number of clusters is selected using inertia and silhouette score.

Generated artifacts:

- `artifacts/clustering_k_analysis.csv`
- `artifacts/clustering_summary.json`
- `artifacts/cluster_assignments.csv`
- `reports/clustering_elbow_silhouette.png`
- `reports/hierarchical_dendrogram.png`

## Cross Validation

Five-fold cross validation is performed for the regression models using negative mean squared error, converted back to RMSE.

Generated artifacts:

- `artifacts/cv_results.csv`
- `reports/cross_validation_rmse.png`

## Dashboard Integration

The dashboard integrates the trained best model artifacts:

- `artifacts/best_score_model.pkl`
- `artifacts/best_classification_model.pkl`
- `artifacts/preprocess_classification.pkl`
- `artifacts/training_feature_columns.json`

The Streamlit dashboard includes:

- A score prediction tab
- A high-score classification tab

The dashboard file is `dashboard.py`.

## Recommendation Challenge

Recommendation is possible because the dataset contains user-item-rating interactions. The recommendation implementation uses matrix factorization through SVD-style collaborative filtering.

Implementation:

- `src/recommender.py`
- `notebooks/06_recommendation.ipynb`
- `run_all.py`

The recommender uses `(user_id, item_id, rating)` triplets and can generate top-N recommendations for a selected user.

Generated artifacts:

- `artifacts/recommender_predictions.csv`
- `artifacts/top_recommendations_example.csv`
- `artifacts/recommender_summary.json`
- `artifacts/simple_svd_recommender.pkl`
