from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from risk_data import load_and_prepare_data

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Student_performance_data _.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "predictions": preds,
            "accuracy": accuracy_score(y_test, preds),
        }

    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    results["XGBoost (Initial)"] = {
        "model": xgb_model,
        "predictions": xgb_preds,
        "accuracy": accuracy_score(y_test, xgb_preds),
    }

    return results


def tune_xgboost(X_train, y_train, X_test, y_test):
    print("\nTuning XGBoost...")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }

    xgb_base = XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_preds = best_model.predict(X_test)
    best_acc = accuracy_score(y_test, best_preds)

    print("Best XGBoost parameters:", grid_search.best_params_)
    print(f"Tuned XGBoost Accuracy: {best_acc * 100:.2f}%")
    print(classification_report(y_test, best_preds, target_names=["Not At Risk", "At Risk"]))

    return best_model, best_preds, best_acc


def plot_accuracy_comparison(results):
    labels = []
    accuracies = []
    colors = ["#95a5a6", "#3498db", "#e74c3c"]

    for idx, (name, info) in enumerate(results.items()):
        labels.append(name)
        accuracies.append(info["accuracy"] * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accuracies, color=colors[: len(labels)], width=0.5)
    ax.set_ylim(85, 100)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = OUTPUT_DIR / "model_accuracy_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    print("\nFeature importance:")
    print(importance_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#3498db")
    ax.set_xlabel("Importance Score")
    ax.set_title("XGBoost Feature Importance for Academic Risk Prediction")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
    plt.close(fig)

    cumulative_df = importance_df.copy()
    cumulative_df["Cumulative"] = cumulative_df["Importance"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_df["Feature"], cumulative_df["Cumulative"], marker="o", linewidth=2, color="#e74c3c")
    ax.axhline(y=0.8, color="green", linestyle="--", label="80% Threshold")
    ax.set_xlabel("Features (sorted by importance)")
    ax.set_ylabel("Cumulative Importance")
    ax.set_title("Cumulative Feature Importance")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cumulative_feature_importance.png", dpi=300)
    plt.close(fig)

    return importance_df


def explain_sample_prediction(model, X_test, y_test):
    print("\nIndividual Student Prediction (First Student):")
    sample = X_test.iloc[[0]]
    actual_risk = y_test.iloc[0]
    sample_pred = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    print("Features:", sample.iloc[0].to_dict())
    print("Actual Risk:", actual_risk)
    print(f"Predicted Risk: {sample_pred}")
    print(f"Risk Probability: Not At Risk={probability[0]:.2%}, At Risk={probability[1]:.2%}")


def main():
    (X_train, X_test, y_train, y_test), X, y = load_and_prepare_data(DATA_PATH)

    print("Preprocessing Done")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Features:", X.columns.tolist())

    print("\nTraining models...\n")
    results = train_models(X_train, X_test, y_train, y_test)

    print("=" * 50)
    print(f"{'Model':<25} {'Accuracy':>10}")
    print("=" * 50)
    for name, info in results.items():
        print(f"{name:<25} {info['accuracy'] * 100:>9.2f}%")
    print("=" * 50)

    best_xgb, _, _ = tune_xgboost(X_train, y_train, X_test, y_test)

    print("\nGenerating Feature Importance Analysis...")
    plot_feature_importance(best_xgb, X.columns)

    explain_sample_prediction(best_xgb, X_test, y_test)

    joblib.dump(best_xgb, BASE_DIR / "academic_risk_xgboost.pkl")
    print("\nBest XGBoost model saved as academic_risk_xgboost.pkl")
    print("\nFeature Analysis Complete!")

    plot_accuracy_comparison(results)
    print(f"\nCharts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
