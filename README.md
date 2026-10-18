# Academic Insight

Academic Insight is a local Streamlit interface for the project's saved academic-risk classifier. It supports individual and CSV batch predictions, local SHAP explanations, and evaluation of the saved model on the project's reproducible held-out split.

## Run locally on Windows

Open PowerShell in this project folder and activate the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Streamlit prints a local URL, normally `http://localhost:8501`. If PowerShell blocks virtual-environment activation, invoke it directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

The app expects `academic_risk_xgboost.pkl` and `Student_performance_data _.csv` beside `app.py`. The current saved model is a binary XGBoost classifier with the exact feature order shown in the app. The model artifact is loaded, not retrained, when the app starts.

## Batch CSV format

Provide one row per student with the exact model feature headers:

```text
Age,Gender,Ethnicity,ParentalEducation,StudyTimeWeekly,Absences,Tutoring,ParentalSupport,Extracurricular,Sports,Music,Volunteering
```

Encoded categorical values must use codes present in the project dataset. The app shows those raw codes because the repository does not include a codebook mapping them to category names. Missing, non-numeric, non-finite, out-of-observed-range, and unsupported categorical values are reported row by row; valid rows can still be scored.

## Model, target, and evaluation

The training pipeline defines `AcademicRisk = 1` when `GPA < 2.0`, and `AcademicRisk = 0` otherwise. `StudentID`, `GPA`, and `GradeClass` are excluded from model inputs. The app reuses the same shared preprocessing and a stratified 80/20 split with `random_state=42` to evaluate the saved artifact. Reported evaluation numbers are therefore held-out metrics for that split, not training metrics or independent external validation.

Local explanations use TreeSHAP contributions from the saved XGBoost model. Global built-in feature importance and per-student SHAP values are different measures; neither establishes causation.

## Responsible use and privacy

This is a student project using the included dataset, not a validated system for university decisions. Predictions are uncertain estimates of the dataset's GPA-based label and must not be treated as diagnoses, disciplinary evidence, or definitive judgments. Consider dataset coverage, potential bias, and the effects of false negatives and false positives; advisors should review relevant context with students before acting.

The app runs locally and does not intentionally write uploaded student data to disk or log row contents. Uploads remain in the local Streamlit process/session while it is running. Do not expose the app publicly or upload identifiable student data unless you have appropriate authorization and safeguards. The `.pkl` model is loaded with `joblib`; only use a trusted local artifact because pickle-based files can execute code when loaded.

## Existing project pipeline

Train/tune and regenerate the model and charts manually when needed:

```powershell
.\.venv\Scripts\python.exe main.py
```

This training command can take several minutes. It is separate from normal app startup.
