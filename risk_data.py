from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "AcademicRisk"
EXCLUDED_COLUMNS = ("StudentID", "GPA", "GradeClass")
RISK_THRESHOLD = 2.0


def load_features_and_target(data_path: Path):
    """Load the project dataset and apply the same target and feature rules used in training."""
    frame = pd.read_csv(data_path)
    frame.columns = [str(column).strip() for column in frame.columns]

    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Dataset contains duplicate column names after trimming: {duplicates}")

    if TARGET_COLUMN not in frame.columns:
        if "GPA" not in frame.columns:
            raise ValueError(
                f"Dataset must contain either {TARGET_COLUMN!r} or 'GPA' to define the target."
            )
        frame[TARGET_COLUMN] = (pd.to_numeric(frame["GPA"], errors="raise") < RISK_THRESHOLD).astype(int)

    feature_columns = [
        column
        for column in frame.columns
        if column != TARGET_COLUMN and column not in EXCLUDED_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("Dataset does not contain any model features.")

    cleaned = frame.dropna(subset=feature_columns + [TARGET_COLUMN]).reset_index(drop=True)
    X = cleaned.loc[:, feature_columns].copy()
    y_numeric = pd.to_numeric(cleaned[TARGET_COLUMN], errors="raise")

    if not set(y_numeric.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1.")
    if y_numeric.nunique() != 2:
        raise ValueError(f"{TARGET_COLUMN} must contain both classes (0 and 1).")

    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="raise")

    return X, y_numeric.astype(int)


def load_and_prepare_data(data_path: Path):
    """Return the stratified train/test split and full cleaned feature table."""
    X, y = load_features_and_target(data_path)
    split = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return split, X, y
