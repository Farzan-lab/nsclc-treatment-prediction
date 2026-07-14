"""
Phase 3.2 - Preprocessing pipeline

- numeric features -> StandardScaler (fit on TRAIN only -> no leakage)
- binary features  -> passthrough

numeric vs binary is auto-detected from the TRAIN set by cardinality,
so the same builder works for the 18-clinical set and the 68-feature set.

class_weight='balanced' belongs on the estimator, not here. Wrap this
preprocessor + estimator in a single sklearn Pipeline so the scaler is
refit correctly inside every CV fold.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def split_feature_types(X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Columns with >2 distinct train values are numeric; the rest are binary."""
    numeric, binary = [], []
    for col in X_train.columns:
        if X_train[col].nunique(dropna=False) > 2:
            numeric.append(col)
        else:
            binary.append(col)
    return numeric, binary


def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric, binary = split_feature_types(X_train)
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("bin", "passthrough", binary),
        ],
        remainder="drop",
    )
    return pre, numeric, binary
