import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocess(df: pd.DataFrame, target: str) -> ColumnTransformer:
    """Create a preprocessing pipeline.

    - Numeric columns: median imputation + StandardScaler
    - Categorical columns: most‑frequent imputation + OneHotEncoder (ignore unknown)
    """
    # Excluir a coluna alvo
    feature_df = df.drop(columns=[target])
    # Identificar tipos (filtrando colunas de alta cardinalidade/IDs que não devem ser features)
    cols_to_exclude = ['username', 'anime_id', 'mal_id']
    num_cols = [c for c in feature_df.select_dtypes(include=["int64", "float64"]).columns if c not in cols_to_exclude]
    cat_cols = [c for c in feature_df.select_dtypes(include=["object", "category"]).columns if c not in cols_to_exclude]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])
    return preprocess
