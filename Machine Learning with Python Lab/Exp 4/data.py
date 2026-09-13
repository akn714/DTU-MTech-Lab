import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


def load_data():
    # Load dataset ID 2 (Adult Census Income)
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    # Normalize column names by stripping whitespace
    X.columns = X.columns.str.strip()

    # The dataset uses '?' for missing values (often with whitespace)
    X = X.replace(r"^\s*\?\s*$", np.nan, regex=True)

    # Clean target values and encode binary target (<=50K -> 0, >50K -> 1)
    target_col = y.columns[0]
    y_clean = y[target_col].astype(str).str.strip().str.rstrip(".")
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y_clean), name="income")

    return X, y_encoded


if __name__ == "__main__":
    X, y = load_data()
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print("Class distribution:\n", y.value_counts())