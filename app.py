import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from risk_data import load_features_and_target
from risk_inference import (
    DATA_PATH,
    MODEL_PATH,
    build_feature_specs,
    evaluate_saved_model,
    explain_prediction,
    feature_importance,
    load_model,
    model_feature_names,
    predict_batch,
    predict_one,
)


st.set_page_config(page_title="Academic Insight", layout="wide", initial_sidebar_state="expanded")

NAVY = "#20313f"
SLATE = "#657481"
TEAL = "#26766f"
PAPER = "#f7f6f1"
BORDER = "#dedfd9"
RISK = "#a75e55"
PROJECT_DIR = Path(__file__).resolve().parent

st.markdown(
    f"""
    <style>
    :root {{ --paper:{PAPER}; --navy:{NAVY}; --slate:{SLATE}; --teal:{TEAL}; --border:{BORDER}; }}
    .stApp {{ background:var(--paper); color:var(--navy); }}
    [data-testid="stSidebar"] {{ background:#f0efe9; border-right:1px solid var(--border); }}
    [data-testid="stSidebar"] h1 {{ font-size:1.35rem; letter-spacing:-.03em; }}
    h1,h2,h3 {{ color:var(--navy); letter-spacing:-.025em; }}
    h1 {{ font-size:clamp(1.8rem,4vw,2.55rem); }}
    .eyebrow {{ color:var(--teal); text-transform:uppercase; letter-spacing:.12em;
                font-size:.72rem; font-weight:700; margin-bottom:.4rem; }}
    .dek {{ color:var(--slate); font-size:1.02rem; max-width:760px; line-height:1.65; }}
    .section-note {{ color:var(--slate); font-size:.88rem; line-height:1.55; }}
    div[data-testid="stMetric"] {{ background:#fbfaf6; border:1px solid var(--border);
                                  border-radius:3px; padding:.9rem 1rem; }}
    div[data-testid="stMetricValue"] {{ color:var(--navy); font-size:1.65rem; }}
    .stButton > button,.stDownloadButton > button {{ border-radius:3px; border:1px solid var(--teal);
      color:white; background:var(--teal); font-weight:600; }}
    .stButton > button:hover,.stDownloadButton > button:hover {{ border-color:var(--navy);
      background:var(--navy); color:white; }}
    hr {{ border-color:var(--border); }}
    @media(max-width:700px) {{ section.main > div {{ padding-left:1rem; padding-right:1rem; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the saved model…")
def cached_model(path: str, modified_ns: int):
    del modified_ns
    return load_model(Path(path))


@st.cache_data(show_spinner=False)
def cached_source_data(path: str, modified_ns: int):
    del modified_ns
    return load_features_and_target(Path(path))


@st.cache_data(show_spinner="Calculating held-out evaluation…")
def cached_evaluation(path: str, modified_ns: int, data_path: str, data_modified_ns: int):
    del modified_ns, data_modified_ns
    return evaluate_saved_model(load_model(Path(path)), Path(data_path))


@st.cache_resource(show_spinner=False)
def cached_explainer(_model, modified_ns: int):
    import shap

    del modified_ns
    return shap.TreeExplainer(_model)


def artifact_mtime(path: Path) -> int:
    return path.stat().st_mtime_ns


def page_heading(eyebrow: str, title: str, description: str):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="dek">{description}</p>', unsafe_allow_html=True)


def load_project_context():
    missing = [path.name for path in (MODEL_PATH, DATA_PATH) if not path.is_file()]
    if missing:
        st.error(
            "Required project file(s) are missing: "
            + ", ".join(f"`{name}`" for name in missing)
            + ". Place the original files beside `app.py`, then reload."
        )
        st.stop()
    try:
        model = cached_model(str(MODEL_PATH), artifact_mtime(MODEL_PATH))
        X, y = cached_source_data(str(DATA_PATH), artifact_mtime(DATA_PATH))
        features = model_feature_names(model)
        if list(X.columns) != features:
            st.error("Dataset feature names/order do not match the saved model. Refusing to predict.")
            st.stop()
        specs = build_feature_specs(X, features)
        return model, X, y, features, specs
    except Exception as exc:
        st.error(f"Could not load the project model or dataset: {exc}")
        st.stop()


def get_evaluation():
    return cached_evaluation(
        str(MODEL_PATH), artifact_mtime(MODEL_PATH), str(DATA_PATH), artifact_mtime(DATA_PATH)
    )


def overview_page(X, y):
    page_heading(
        "ACADEMIC ANALYTICS",
        "Academic Insight",
        "A local decision-support tool for reviewing a GPA-based academic-risk estimate. "
        "It surfaces model evidence for an advisor to consider—not a definitive judgment about a student.",
    )
    st.markdown("---")
    st.subheader("Saved model · held-out evaluation")
    try:
        evaluation = get_evaluation()
        st.caption(
            f"Saved XGBoost model evaluated on a stratified 80/20 holdout "
            f"({evaluation['test_size']:,} test rows; random seed 42). "
            "These are held-out results, not training metrics or external validation."
        )
        columns = st.columns(4)
        for column, label, key in zip(
            columns,
            ["Accuracy", "At-risk precision", "At-risk recall", "At-risk F1"],
            ["accuracy", "precision", "recall", "f1"],
        ):
            column.metric(label, f"{evaluation[key]:.1%}")
    except Exception as exc:
        st.warning(f"Held-out metrics are unavailable: {exc}")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("What the label represents")
        st.markdown(
            "In this dataset, **At risk (1)** means the recorded GPA is below **2.0**; "
            "**Not at risk (0)** means GPA is 2.0 or higher. GPA, grade class, and student ID "
            "are excluded from model inputs. The estimate predicts this dataset label, not a "
            "future outcome or a student's potential."
        )
        st.markdown(
            '<p class="section-note">The reference dataset has no codebook for its integer-encoded '
            'categories. The app shows the original numeric codes rather than guessing their meaning.</p>',
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Reference labels")
        counts = y.value_counts().reindex([0, 1], fill_value=0)
        chart_data = pd.DataFrame(
            {"Dataset label": ["Not at risk (0)", "At risk (1)"], "Rows": counts.to_numpy()}
        )
        fig = px.bar(
            chart_data,
            x="Dataset label",
            y="Rows",
            color="Dataset label",
            color_discrete_map={"Not at risk (0)": "#9ca8a9", "At risk (1)": TEAL},
        )
        fig.update_layout(
            showlegend=False, height=300, margin=dict(l=8, r=8, t=12, b=8),
            paper_bgcolor=PAPER, plot_bgcolor=PAPER, font=dict(color=NAVY), xaxis_title=None,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Observed dataset labels only · {len(y):,} usable rows.")

    st.markdown("---")
    st.subheader("Choose a task")
    cards = st.columns(3, gap="medium")
    for column, title, detail in zip(
        cards,
        ["Student assessment", "Batch assessment", "Model evaluation"],
        ["Enter one student's recorded features.", "Validate and score a CSV locally.", "Review test metrics and existing analyses."],
    ):
        with column:
            st.markdown(f"**{title}**")
            st.markdown(f'<p class="section-note">{detail}</p>', unsafe_allow_html=True)
            if st.button(f"Open {title.lower()}", key=f"goto_{title}"):
                st.session_state["page"] = title
                st.rerun()


def feature_input(spec, X):
    if spec.is_categorical:
        options = list(spec.category_codes)
        counts = X[spec.name].value_counts()
        default = float(counts.index[0])
        return float(
            st.selectbox(
                f"{spec.name} — encoded value",
                options=options,
                index=options.index(default) if default in options else 0,
                format_func=lambda code: f"{code:g}",
                key=f"student_{spec.name}",
                help="Choose an original numeric code. No category codebook is available.",
            )
        )
    median = float(X[spec.name].median())
    if spec.is_integer:
        return int(
            st.number_input(
                spec.name,
                min_value=int(spec.minimum),
                max_value=int(spec.maximum),
                value=int(round(median)),
                step=1,
                key=f"student_{spec.name}",
                help=f"Observed dataset range: {spec.minimum:g}–{spec.maximum:g}.",
            )
        )
    return float(
        st.number_input(
            spec.name,
            min_value=float(spec.minimum),
            max_value=float(spec.maximum),
            value=min(max(median, spec.minimum), spec.maximum),
            step=0.5,
            format="%.3f",
            key=f"student_{spec.name}",
            help=f"Observed dataset range: {spec.minimum:.3f}–{spec.maximum:.3f}.",
        )
    )


def explanation_chart(explanation):
    shown = explanation.head(10).sort_values("Contribution toward prediction")
    custom = [
        [float(value), str(direction)]
        for value, direction in zip(shown["SHAP contribution"], shown["Direction"])
    ]
    fig = go.Figure(
        go.Bar(
            x=shown["Contribution toward prediction"],
            y=shown["Feature"],
            orientation="h",
            marker_color=[
                TEAL if direction == "Toward predicted class" else RISK
                for direction in shown["Direction"]
            ],
            customdata=custom,
            hovertemplate=(
                "%{y}<br>Contribution toward result: %{x:.3f}"
                "<br>Positive-class SHAP: %{customdata[0]:.3f}"
                "<br>%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color=BORDER)
    fig.update_layout(
        height=max(330, 32 * len(shown)),
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(color=NAVY),
        xaxis_title="SHAP contribution toward / away from predicted class (log-odds)",
        yaxis_title=None,
        showlegend=False,
    )
    return fig


def individual_page(model, X, features, specs):
    page_heading(
        "STUDENT ASSESSMENT",
        "Individual student",
        "Enter values as recorded in the dataset. Encoded category fields use original numeric "
        "codes; no undocumented code labels are inferred.",
    )
    st.info(
        "The estimate is a model output for review, not a diagnosis or final judgment. "
        "Use relevant student context and human review."
    )
    with st.form("student_assessment"):
        columns = st.columns(3, gap="large")
        values = {}
        for index, spec in enumerate(specs):
            with columns[index % len(columns)]:
                values[spec.name] = feature_input(spec, X)
        st.caption(
            "Initial field values are derived from reference-data medians or common codes so they "
            "remain within observed ranges. Review every field against a real student record; "
            "no estimate is shown until you confirm the values."
        )
        verified = st.checkbox(
            "I verified every field against an actual student record.",
            key="student_record_verified",
        )
        submitted = st.form_submit_button("Assess student", type="primary")
    if not submitted:
        return
    if not verified:
        st.warning("Confirm that all values belong to an actual record before requesting an estimate.")
        return

    row = pd.DataFrame([[values[name] for name in features]], columns=features)
    try:
        result = predict_one(model, row, features)
        explanation = explain_prediction(
            model, row, result["predicted_class"], cached_explainer(model, artifact_mtime(MODEL_PATH))
        )
        left, right = st.columns([0.8, 1.2], gap="large")
        with left:
            st.markdown("---")
            st.subheader("Assessment result")
            st.markdown(f"### {result['class_label']}")
            st.metric(
                "Estimated probability of the dataset's at-risk label",
                f"{result['risk_probability']:.1%}",
            )
            st.caption(
                f"Not-at-risk probability: {result['not_at_risk_probability']:.1%}. "
                "Probabilities are model estimates, not calibrated certainty."
            )
        with right:
            st.plotly_chart(explanation_chart(explanation), width="stretch")
            strongest = explanation.iloc[0]
            st.markdown(
                f"**Plain-language reading:** Among the factors shown, **{strongest['Feature']}** "
                f"contributed most strongly {strongest['Direction'].lower()} for this prediction. "
                "This describes model behavior, not a cause of the student's outcome."
            )
            st.caption(
                "Directions are relative to the predicted class. SHAP values are on the model's "
                "log-odds scale, not percentage points."
            )
    except Exception as exc:
        st.error(f"Could not generate this assessment or its explanation: {exc}")


def batch_page(model, features, specs):
    page_heading(
        "BATCH ASSESSMENT",
        "Review a student file",
        "Upload a CSV with one row per student. Required values are validated before scoring; "
        "invalid rows remain visible and are not predicted.",
    )
    st.markdown("**Required headers:** " + ", ".join(f"`{name}`" for name in features))
    st.caption(
        "Numeric ranges and category codes are checked against the included dataset. "
        "Extra columns are retained in the download but are not used by the model."
    )
    uploaded = st.file_uploader("Choose a student CSV", type=["csv"], key="batch_upload")
    if uploaded is None:
        return
    try:
        uploaded_bytes = uploaded.getvalue()
        token = hashlib.sha256(uploaded_bytes).hexdigest()
        upload_df = pd.read_csv(uploaded)
        if upload_df.empty:
            st.error("The uploaded CSV contains no student rows.")
            return
        upload_df.columns = [str(column).strip() for column in upload_df.columns]
        if upload_df.columns.duplicated().any():
            st.error("The CSV contains duplicate headers after trimming whitespace.")
            return
        missing = [name for name in features if name not in upload_df.columns]
        if missing:
            st.error(f"Required model columns are missing: {', '.join(missing)}")
            st.caption("No predictions were made. Correct the headers and upload again.")
            return
        extras = [name for name in upload_df.columns if name not in features]
        if extras:
            st.warning("Extra columns are retained but ignored by the model: " + ", ".join(extras))
        if st.session_state.get("batch_upload_token") != token:
            st.session_state.pop("batch_result", None)
            st.session_state["batch_upload_token"] = token

        st.subheader("Preview")
        st.dataframe(upload_df.head(8), width="stretch", hide_index=True)
        if st.button("Validate and assess rows", type="primary", key="run_batch"):
            try:
                st.session_state["batch_result"] = predict_batch(model, upload_df, specs)
            except Exception as exc:
                st.error(f"Could not process this CSV: {exc}")
                return
        result = st.session_state.get("batch_result")
        if result is None:
            return
        valid = result[result["Validation status"] == "Valid"]
        invalid_count = len(result) - len(valid)
        summary = st.columns(3)
        summary[0].metric("Rows received", f"{len(result):,}")
        summary[1].metric("Rows assessed", f"{len(valid):,}")
        summary[2].metric("Rows needing correction", f"{invalid_count:,}")
        if not valid.empty:
            at_risk = int((valid["Predicted class"] == "At risk").sum())
            st.markdown(
                f"Among {len(valid):,} valid rows, the model classified {at_risk:,} as "
                f"**At risk** and {len(valid) - at_risk:,} as **Not at risk**."
            )
        st.dataframe(result, width="stretch", hide_index=True)
        st.download_button(
            "Download assessment CSV",
            data=result.to_csv(index=False).encode("utf-8-sig"),
            file_name="academic_risk_assessments.csv",
            mime="text/csv",
            type="primary",
        )
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")


def confusion_chart(matrix):
    fig = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted not at risk", "Predicted at risk"],
        y=["Actual not at risk", "Actual at risk"],
        color_continuous_scale=["#f3f1ea", TEAL],
        aspect="auto",
    )
    fig.update_layout(
        height=340,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor=PAPER,
        font=dict(color=NAVY),
        coloraxis_showscale=False,
    )
    fig.update_traces(textfont_color=NAVY)
    return fig


def evaluation_page(model, features):
    page_heading(
        "MODEL EVALUATION",
        "Held-out evaluation",
        "Results below are computed from the saved model and the project's fixed, stratified "
        "test split. They are not evidence of independent or real-world validation.",
    )
    try:
        evaluation = get_evaluation()
    except Exception as exc:
        st.warning(f"Evaluation metrics are unavailable: {exc}")
        evaluation = None
    if evaluation:
        st.caption(
            f"{evaluation['test_size']:,} test rows from the deterministic 80/20 stratified split "
            "(random seed 42) · positive class: AcademicRisk = 1."
        )
        columns = st.columns(5)
        metrics = [
            ("Accuracy", "accuracy"),
            ("At-risk precision", "precision"),
            ("At-risk recall", "recall"),
            ("At-risk F1", "f1"),
            ("ROC–AUC", "roc_auc"),
        ]
        for column, (label, key) in zip(columns, metrics):
            value = evaluation[key]
            column.metric(label, f"{value:.1%}" if value is not None else "Unavailable")
        left, right = st.columns([0.8, 1.2], gap="large")
        with left:
            st.subheader("Confusion matrix")
            st.plotly_chart(confusion_chart(evaluation["confusion_matrix"]), width="stretch")
        with right:
            st.subheader("Why recall matters")
            st.markdown(
                "At-risk recall measures how many students with the dataset's at-risk label are "
                "identified. Low recall means more labeled at-risk students are missed, so recall "
                "matters when the purpose is supportive follow-up. Precision matters too: false "
                "alerts can waste advisor time or cause concern. Neither metric replaces review "
                "of an individual student's circumstances."
            )

    st.markdown("---")
    st.subheader("Global built-in feature importance")
    st.markdown(
        '<p class="section-note">These values come from XGBoost’s built-in '
        'feature_importances_ attribute. They summarize model behavior across trees; they are '
        'not local SHAP contributions and do not establish causes.</p>',
        unsafe_allow_html=True,
    )
    try:
        importance = feature_importance(model, features).sort_values("Importance")
        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color_discrete_sequence=[TEAL],
        )
        fig.update_layout(
            height=400,
            margin=dict(l=8, r=8, t=12, b=8),
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            font=dict(color=NAVY),
            xaxis_title="Built-in importance",
            yaxis_title=None,
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.info(f"Built-in feature-importance values are unavailable: {exc}")

    comparison = PROJECT_DIR / "outputs" / "model_accuracy_comparison.png"
    if comparison.is_file():
        st.markdown("---")
        st.subheader("Saved initial-model comparison")
        st.caption(
            "Archived comparison of initial, untuned models from the training script. It is "
            "displayed as an existing artifact, not recomputed or compared with the tuned model."
        )
        st.image(str(comparison), width="stretch")

    images = [
        ("EDA visualizations", PROJECT_DIR / "EDA_Visualizations.png"),
        ("Saved global SHAP plot", PROJECT_DIR / "shap_global_bar.png"),
        ("Saved SHAP waterfall", PROJECT_DIR / "shap_waterfall.png"),
    ]
    images = [(label, path) for label, path in images if path.is_file()]
    if images:
        st.markdown("---")
        st.subheader("Existing project visualizations")
        for tab, (label, path) in zip(st.tabs([label for label, _ in images]), images):
            with tab:
                st.image(str(path), caption=label, width="stretch")


def about_page():
    page_heading(
        "ABOUT & LIMITATIONS",
        "Use as a prompt for review, not a decision",
        "Academic Insight exposes the behavior of one saved model and one project dataset. "
        "Its estimates are not validated for institutional decision-making.",
    )
    st.subheader("Dataset and target")
    st.markdown(
        "The app reads `Student_performance_data _.csv`. The binary target is **AcademicRisk = 1 "
        "when GPA < 2.0**, and 0 otherwise. StudentID, GPA, and GradeClass are excluded from "
        "inputs. The CSV has integer-encoded categories but no codebook; the app displays original "
        "numeric codes and does not guess their labels."
    )
    st.subheader("Model and evaluation")
    st.markdown(
        "Predictions use the saved `academic_risk_xgboost.pkl` classifier and its stored feature "
        "order. Training and parameter search do not run on app startup. Metrics are calculated "
        "on the fixed stratified 80/20 split (random seed 42), an internal holdout from the same "
        "dataset—not external validation or proof of performance for another cohort."
    )
    st.subheader("Interpretation and limitations")
    st.markdown(
        "TreeSHAP explains how this fitted model responds to a row; it does not show causal "
        "effects. Built-in global feature importance is a separate summary. The dataset may not "
        "represent all student populations, its GPA threshold may miss broader needs, and model "
        "errors can affect students differently. A probability is an estimate, not a diagnosis, "
        "guaranteed outcome, or definitive judgment of ability."
    )
    st.subheader("Privacy and responsible use")
    st.markdown(
        "Uploaded rows are processed in the local Streamlit process and are not written to disk by "
        "the app. Do not expose the server publicly or use identifiable student data without "
        "authorization and appropriate safeguards. Advisors should review context and discuss "
        "supportive next steps rather than acting on a score alone. Only load a trusted local "
        "joblib artifact; pickle-based files can execute code when deserialized."
    )


def main():
    model, X, y, features, specs = load_project_context()
    pages = [
        "Overview",
        "Student assessment",
        "Batch assessment",
        "Model evaluation",
        "About & limitations",
    ]
    with st.sidebar:
        st.markdown('<div class="eyebrow">ACADEMIC ANALYTICS</div>', unsafe_allow_html=True)
        st.title("Academic Insight")
        st.caption("Local academic-risk review")
        current = st.session_state.get("page", "Overview")
        if current not in pages:
            current = "Overview"
        page = st.radio("Workspace", pages, index=pages.index(current))
        st.session_state["page"] = page
        st.markdown("---")
        st.caption("Saved XGBoost model · loaded locally")
        st.caption(f"{len(features)} model features")

    if page == "Overview":
        overview_page(X, y)
    elif page == "Student assessment":
        individual_page(model, X, features, specs)
    elif page == "Batch assessment":
        batch_page(model, features, specs)
    elif page == "Model evaluation":
        evaluation_page(model, features)
    else:
        about_page()


if __name__ == "__main__":
    main()