import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply a log1p (log(1 + x)) transformation.
    
    This is particularly useful for heavily right-skewed features (e.g., number 
    of watched episodes), normalizing the distribution to improve the performance 
    of linear models and distance-based algorithms (like KNN and SVM). 
    Inheriting from BaseEstimator and TransformerMixin ensures seamless 
    integration with scikit-learn Pipelines.
    """

    def fit(self, X, y=None):
        # No parameters to learn for a logarithmic transformation.
        return self

    def transform(self, X):
        # Applies the natural logarithm of one plus the input array, element-wise.
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        # Preserves the original feature names for downstream interpretability
        # (e.g., when analyzing feature importance in Random Forests).
        return np.asarray(input_features, dtype=object)


def build_preprocess(df: pd.DataFrame, target: str) -> ColumnTransformer:
    """
    Constructs the master preprocessing pipeline (ColumnTransformer).
    
    This function explicitly fulfills the project requirement to normalize 
    and standardize the dataset. It segregates features by data type and 
    applies tailored transformation strategies to each subset.
    """
    # Separate the independent variables from the target variable
    feature_df = df.drop(columns=[target])
    
    # Identify column types, deliberately excluding high-cardinality identifiers 
    # that offer no predictive value and could cause data leakage or memory issues.
    cols_to_exclude = ['username', 'anime_id', 'mal_id']
    
    num_cols = [c for c in feature_df.select_dtypes(include=["int64", "float64"]).columns if c not in cols_to_exclude]
    cat_cols = [c for c in feature_df.select_dtypes(include=["object", "category"]).columns if c not in cols_to_exclude]
    
    # Isolate specific columns that require logarithmic transformation prior to scaling
    log_cols = [c for c in ["num_watched_episodes"] if c in feature_df.columns]
    other_num_cols = [c for c in num_cols if c not in log_cols]

    # Pipeline 1: Skewed Numeric Features
    # Strategy: Impute missing values -> Apply log1p to handle skewness -> Standardize
    log_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", Log1pTransformer()),
        ("scaler", StandardScaler())
    ])
    
    # Pipeline 2: Standard Numeric Features
    # Strategy: Impute missing values -> Standardize (Z-score normalization)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Pipeline 3: Categorical Features
    # Strategy: Impute with the mode -> Convert to binary vectors via One-Hot Encoding
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # sparse_output=False is required if applying SVD/PCA later in the pipeline
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Assemble all sub-pipelines into a single ColumnTransformer
    preprocess = ColumnTransformer(
        [
            ("log_num", log_pipe, log_cols),
            ("num", numeric_pipe, other_num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        sparse_threshold=0.0, # Ensures dense matrix output, compatible with all models
    )
    return preprocess


def get_preprocess_feature_names(preprocess: ColumnTransformer) -> np.ndarray:
    """
    Extracts the updated feature names after all transformations have been applied.
    
    This helper function is vital for maintaining the interpretability of the models 
    (e.g., understanding which One-Hot Encoded category is driving a prediction) and 
    avoids compatibility issues across different versions of scikit-learn.
    """
    feature_names = []
    
    # Iterate through each step of the ColumnTransformer
    for name, transformer, columns in preprocess.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
            
        if transformer == "passthrough":
            feature_names.extend([f"{name}__{col}" for col in columns])
            continue
            
        if name == "cat":
            # For categorical pipelines, retrieve the dynamically generated names 
            # from the OneHotEncoder (e.g., 'cat__genre_Action').
            encoder = transformer.named_steps["encoder"]
            feature_names.extend(encoder.get_feature_names_out(columns))
        else:
            # For numeric pipelines, the names remain structurally the same, 
            # but we prefix them to indicate which pipeline they passed through.
            feature_names.extend([f"{name}__{col}" for col in columns])
            
    return np.asarray(feature_names, dtype=object)