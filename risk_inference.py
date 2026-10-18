from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from risk_data import load_and_prepare_data


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "academic_risk_xgboost.pkl"
DATA_PATH = BASE_DIR / "Student_performance_data _.csv"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    minimum: float
    maximum: float
    is_integer: bool
    category_codes: tuple[float, ...]

    @property
    def is_categorical(self) -> bool:
        return bool(self.category_codes)


def model_feature_names(model) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
    if names is None or len(names) == 0:
        raise ValueError("The saved model does not contain feature names; safe input ordering is unknown.")
    return [str(name) for name in names]


def load_model(model_path: Path = MODEL_PATH):
    """Load a trusted, local joblib artifact. Never pass user-uploaded files here."""
    model = joblib.load(model_path)
    names = model_feature_names(model)
    classes = list(getattr(model, "classes_", []))
    if set(classes) != {0, 1}:
        raise ValueError(f"Expected a binary classifier with classes 0 and 1; found {classes!r}.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved model does not support probability predictions.")
    if len(names) != len(set(names)):
        raise ValueError("The saved model contains duplicate feature names.")
    return model


def build_feature_specs(X: pd.DataFrame, expected_features: list[str]) -> list[FeatureSpec]:
    missing = [name for name in expected_features if name not in X.columns]
    if missing:
        raise ValueError(f"Reference dataset is missing model features: {missing}")

    specs = []
    scalar_fields = {"Age", "Absences", "StudyTimeWeekly"}
    for name in expected_features:
        values = pd.to_numeric(X[name], errors="raise").dropna()
        if values.empty:
            raise ValueError(f"Reference dataset has no usable values for {name!r}.")
        minimum = float(values.min())
        maximum = float(values.max())
        integer_values = np.all(np.isclose(values.to_numpy(dtype=float) % 1, 0))
        codes = tuple(sorted(float(value) for value in values.unique()))
        categorical = name not in scalar_fields and integer_values and len(codes) <= 12
        specs.append(
            FeatureSpec(
                name=name,
                minimum=minimum,
                maximum=maximum,
                is_integer=bool(integer_values),
                category_codes=codes if categorical else (),
            )
        )
    return specs


def class_label(value: int) -> str:
    return "At risk" if int(value) == 1 else "Not at risk"


def positive_class_index(model) -> int:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model does not include the AcademicRisk=1 class.")
    return classes.index(1)


def predict_one(model, row: pd.DataFrame, expected_features: list[str]) -> dict:
    ordered = _coerce_feature_frame(row, expected_features)
    predicted = int(model.predict(ordered)[0])
    probabilities = model.predict_proba(ordered)[0]
    risk_probability = float(probabilities[positive_class_index(model)])
    return {
        "predicted_class": predicted,
        "class_label": class_label(predicted),
        "risk_probability": risk_probability,
        "not_at_risk_probability": 1.0 - risk_probability,
    }


def validate_batch(upload: pd.DataFrame, specs: list[FeatureSpec]):
    """Return a result table plus a numeric feature table for rows that pass validation."""
    frame = upload.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"CSV contains duplicate column names after trimming: {duplicates}")

    expected_features = [spec.name for spec in specs]
    missing = [name for name in expected_features if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required model columns: {', '.join(missing)}")

    parsed_rows = []
    result = frame.copy()
    statuses = []
    details = []

    for _, row in frame.iterrows():
        parsed = {}
        errors = []
        for spec in specs:
            value = row[spec.name]
            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                errors.append(f"{spec.name}: value is missing")
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"{spec.name}: expected a numeric value")
                continue
            if not np.isfinite(numeric):
                errors.append(f"{spec.name}: value must be finite")
                continue
            if spec.is_categorical and numeric not in spec.category_codes:
                codes = ", ".join(f"{code:g}" for code in spec.category_codes)
                errors.append(f"{spec.name}: unsupported code {numeric:g}; use one of {codes}")
                continue
            if numeric < spec.minimum or numeric > spec.maximum:
                errors.append(
                    f"{spec.name}: {numeric:g} is outside the observed range "
                    f"{spec.minimum:g}-{spec.maximum:g}"
                )
                continue
            if spec.is_integer and not np.isclose(numeric % 1, 0):
                errors.append(f"{spec.name}: expected a whole number")
                continue
            parsed[spec.name] = numeric

        valid = not errors
        statuses.append("Valid" if valid else "Invalid")
        details.append("; ".join(errors))
        if valid:
            parsed_rows.append(parsed)

    result["Validation status"] = statuses
    result["Validation details"] = details
    valid_features = pd.DataFrame(parsed_rows, columns=expected_features)
    return result, valid_features


def predict_batch(model, upload: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    result, valid_features = validate_batch(upload, specs)
    result["Predicted class"] = pd.NA
    result["Risk probability"] = pd.NA
    result["Not-at-risk probability"] = pd.NA

    if valid_features.empty:
        return result

    predictions = model.predict(valid_features)
    probabilities = model.predict_proba(valid_features)
    risk_probabilities = probabilities[:, positive_class_index(model)]

    valid_positions = [index for index, status in enumerate(result["Validation status"]) if status == "Valid"]
    for position, predicted, risk_probability in zip(
        valid_positions, predictions, risk_probabilities
    ):
        result.at[result.index[position], "Predicted class"] = class_label(int(predicted))
        result.at[result.index[position], "Risk probability"] = float(risk_probability)
        result.at[result.index[position], "Not-at-risk probability"] = float(1.0 - risk_probability)
    return result


def evaluate_saved_model(model, data_path: Path = DATA_PATH) -> dict:
    """Evaluate the saved artifact on the same deterministic held-out split as training."""
    (X_train, X_test, y_train, y_test), _, _ = load_and_prepare_data(data_path)
    expected = model_feature_names(model)
    if list(X_test.columns) != expected:
        raise ValueError("Dataset feature names/order do not match the saved model.")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, positive_class_index(model)]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_test, probabilities))
            if y_test.nunique() == 2
            else None
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "y_test": y_test.reset_index(drop=True),
        "predictions": pd.Series(predictions),
        "risk_probabilities": pd.Series(probabilities),
        "test_size": len(y_test),
        "train_size": len(X_train),
    }


def feature_importance(model, expected_features: list[str]) -> pd.DataFrame:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("The saved model does not expose built-in feature importances.")
    if len(importances) != len(expected_features):
        raise ValueError("Feature importance length does not match the model feature list.")
    return (
        pd.DataFrame({"Feature": expected_features, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def explain_prediction(model, row: pd.DataFrame, predicted_class: int, explainer=None) -> pd.DataFrame:
    """Return binary TreeSHAP contributions, oriented toward/away from the predicted class."""
    import shap

    expected = model_feature_names(model)
    ordered = _coerce_feature_frame(row, expected)
    explanation = (explainer or shap.TreeExplainer(model)).shap_values(
        ordered, check_additivity=False
    )
    if isinstance(explanation, list):
        values = np.asarray(explanation[positive_class_index(model)])[0]
    else:
        values = np.asarray(explanation)
        if values.ndim == 3:
            values = values[0, :, positive_class_index(model)]
        else:
            values = values[0]

    toward_prediction = values if int(predicted_class) == 1 else -values
    return (
        pd.DataFrame(
            {
                "Feature": expected,
                "SHAP contribution": values,
                "Contribution toward prediction": toward_prediction,
                "Direction": np.where(
                    toward_prediction >= 0,
                    "Toward predicted class",
                    "Away from predicted class",
                ),
            }
        )
        .assign(_magnitude=lambda frame: frame["SHAP contribution"].abs())
        .sort_values("_magnitude", ascending=False)
        .drop(columns="_magnitude")
        .reset_index(drop=True)
    )


def _coerce_feature_frame(frame: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    missing = [name for name in expected_features if name not in frame.columns]
    if missing:
        raise ValueError(f"Prediction input is missing model features: {missing}")
    ordered = frame.loc[:, expected_features].copy()
    for column in expected_features:
        ordered[column] = pd.to_numeric(ordered[column], errors="raise")
    if ordered.isna().any().any() or not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError("Prediction inputs must be complete, finite numeric values.")
    return ordered
