# Final Presentation Outline

## Slide 1 - Project Title

MyAnimeList Analytics Dashboard and Machine Learning System

## Slide 2 - Dataset And Objective

- Dataset: MyAnimeList ratings and anime metadata
- Main target: `score`
- Main objective: predict anime rating
- Secondary objective: classify ratings as high score or lower score

## Slide 3 - Data Preparation

- Missing value imputation
- Numeric scaling
- Categorical one-hot encoding
- Removal of identifier/high-cardinality columns where appropriate

## Slide 4 - Regression Models

- Linear Regression
- Ridge and Lasso
- SVM with linear, RBF, and polynomial kernels
- K-NN
- Decision Tree
- Random Forest and Gradient Boosting
- Single-layer and multi-layer neural networks

## Slide 5 - Classification Models

- Logistic Regression
- Naive Bayes
- SVM kernels
- K-NN
- Decision Tree with Gini
- Random Forest and Gradient Boosting
- Neural networks

## Slide 6 - Model Comparison

Show:

- `reports/baseline_rmse.png`
- `reports/baseline_fit_times.png`
- `reports/classification_metrics.png`
- `reports/classification_fit_times.png`

## Slide 7 - Hyperparameter Optimization

- GridSearchCV: Ridge, Lasso, K-NN
- Optuna: Random Forest, SVR, neural network
- Alpha analysis for Ridge and Lasso

Show:

- `reports/ridge_alpha_analysis.png`
- `reports/lasso_alpha_analysis.png`

## Slide 8 - Tree-Based Models

- Decision Tree Gini analysis
- Random Forest representative tree image

Show:

- `reports/decision_tree_gini.png`
- `reports/rf_tree.png`

## Slide 9 - PCA / SVD

- Dimensionality reduction with TruncatedSVD
- Comparison of RMSE and fit time before and after SVD

Show:

- `reports/pca_rmse.png`
- `reports/pca_fit_times.png`

## Slide 10 - Clustering

- K-Means clustering
- Hierarchical clustering
- Optimal cluster analysis using elbow and silhouette

Show:

- `reports/clustering_elbow_silhouette.png`
- `reports/hierarchical_dendrogram.png`

## Slide 11 - Recommendation Challenge

- Collaborative filtering using user-item-rating data
- SVD-style recommender implementation
- Top-N anime recommendations for a user

## Slide 12 - Dashboard Demo

- Exploratory analysis tabs
- Prediction tab
- Classification tab
- Best model loaded from exported artifacts

## Slide 13 - Conclusions

- Best predictive model based on RMSE
- Best classification model based on F1-score
- Trade-off between accuracy and fit time
- SVD can reduce dimensionality and improve execution time depending on model

