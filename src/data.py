import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


TARGET = "Revenue"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def build_preprocessor(X: pd.DataFrame):
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return preprocessor