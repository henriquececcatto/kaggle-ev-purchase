from sklearn.model_selection import train_test_split
from data import load_training_data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def prepare_data(
    path, target="Will_Buy_EV", 
    id_col="id", holdout_size=0.05, random_state=42,):

    df = load_training_data(path, target=target, id_col=id_col)

    X = df.drop(columns=[id_col, target])
    y = df[target].copy()

    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=random_state,
        stratify=y,
    )

    return X_dev, X_holdout, y_dev, y_holdout


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    return ColumnTransformer([
        ("numeric", "passthrough", numeric_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),categorical_cols)
    ])

