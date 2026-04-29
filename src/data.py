from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    df = pd.read_csv(DATA_DIR / "processed" / "ibtracs_antilles_model.csv")

    features = ["SEASON", "LAT", "LON", "WMO_WIND", "WMO_PRES"]
    target = "high_risk"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
