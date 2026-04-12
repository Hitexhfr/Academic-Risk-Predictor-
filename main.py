import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv("Student_performance_data _.csv")
df['AcademicRisk'] = (df['GPA'] < 2.0).astype(int)

# ── STAGE 3 — PREPROCESSING ───────────────────────────────
df = df.drop(columns=['StudentID', 'GPA', 'GradeClass'])

X = df.drop(columns=['AcademicRisk'])
y = df['AcademicRisk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Preprocessing Done")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", X.columns.tolist())

# ── STAGE 4 — TRAIN 3 MODELS & COMPARE ────────────────────
print("\n🔄 Training models...\n")

# Model 1 — Logistic Regression (baseline)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Model 2 — Random Forest (existing approach)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# Model 3 — XGBoost (our proposed approach)
xgb = XGBClassifier(n_estimators=100, random_state=42,
                    eval_metric='logloss', verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

# ── COMPARISON TABLE ──────────────────────────────────────
print("=" * 50)
print(f"{'Model':<25} {'Accuracy':>10}")
print("=" * 50)
print(f"{'Logistic Regression':<25} {lr_acc*100:>9.2f}%")
print(f"{'Random Forest':<25} {rf_acc*100:>9.2f}%")
print(f"{'XGBoost (Proposed)':<25} {xgb_acc*100:>9.2f}%")
print("=" * 50)

# ── DETAILED REPORT FOR XGBOOST ───────────────────────────
print("\n📊 XGBoost Detailed Report:")
print(classification_report(y_test, xgb_pred,
      target_names=['Not At Risk', 'At Risk']))

# ── COMPARISON BAR CHART ──────────────────────────────────
models = ['Logistic\nRegression', 'Random\nForest', 'XGBoost\n(Proposed)']
accuracies = [lr_acc * 100, rf_acc * 100, xgb_acc * 100]
colors = ['#95a5a6', '#3498db', '#e74c3c']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors, width=0.5)
plt.ylim(85, 100)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1,
             f'{acc:.2f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# ── SAVE XGBOOST MODEL ────────────────────────────────────
joblib.dump(xgb, 'academic_risk_xgboost.pkl')
print("\n✅ XGBoost model saved as academic_risk_xgboost.pkl")
from sklearn.model_selection import GridSearchCV

print("\n🔄 Tuning XGBoost...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

xgb_tuned = XGBClassifier(random_state=42,
                           eval_metric='logloss',
                           verbosity=0)

grid_search = GridSearchCV(xgb_tuned, param_grid,
                           cv=5, scoring='accuracy',
                           n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print("\n✅ Best Parameters:", grid_search.best_params_)

best_xgb = grid_search.best_estimator_
best_pred = best_xgb.predict(X_test)
best_acc = accuracy_score(y_test, best_pred)

print(f"\n🏆 Tuned XGBoost Accuracy: {best_acc*100:.2f}%")
print(f"vs Logistic Regression:    93.95%")
print(f"vs Random Forest:          92.69%")

print("\n📊 Tuned XGBoost Report:")
print(classification_report(y_test, best_pred,
      target_names=['Not At Risk', 'At Risk']))

# Save tuned model
joblib.dump(best_xgb, 'academic_risk_xgboost.pkl')
print("\n✅ Tuned XGBoost saved!")

print("\n🔄 Generating Feature Importance Analysis...")

# ── FEATURE IMPORTANCE FROM XGBOOST (Built-in, no SHAP issues) ─
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance:")
print(importance_df.to_string(index=False))

# ── CHART 1: FEATURE IMPORTANCE BAR PLOT ──────────────────
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3498db')
plt.xlabel('Importance Score')
plt.title('XGBoost Feature Importance for Academic Risk Prediction')
plt.tight_layout()
plt.show()

# ── CHART 2: CUMULATIVE IMPORTANCE ────────────────────────
importance_df['Cumulative'] = importance_df['Importance'].cumsum()
plt.figure(figsize=(10, 6))
plt.plot(importance_df['Feature'], importance_df['Cumulative'], marker='o', linewidth=2, color='#e74c3c')
plt.axhline(y=0.8, color='green', linestyle='--', label='80% Threshold')
plt.xlabel('Features (sorted by importance)')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# ── INDIVIDUAL STUDENT PREDICTION ─────────────────────────
print("\n📋 Individual Student Prediction (First Student):")
student_features = X_test.iloc[0].to_dict()
print("Features:", student_features)
print("Actual Risk:", y_test.iloc[0])
student_pred = xgb.predict(X_test.iloc[[0]])[0]
student_prob = xgb.predict_proba(X_test.iloc[[0]])[0]
print(f"Predicted Risk: {student_pred}")
print(f"Risk Probability: Not At Risk={student_prob[0]:.2%}, At Risk={student_prob[1]:.2%}")

print("\n✅ Feature Analysis Complete!")
