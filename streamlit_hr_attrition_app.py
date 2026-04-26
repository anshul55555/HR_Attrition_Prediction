import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .success-box {
            padding: 1.3rem;
            border-radius: 18px;
            background: #eafaf1;
            border: 1px solid #c6f6d5;
            color: #14532d;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .danger-box {
            padding: 1.3rem;
            border-radius: 18px;
            background: #fff1f2;
            border: 1px solid #fecdd3;
            color: #881337;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .info-box {
            padding: 1rem;
            border-radius: 14px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Constants
# -----------------------------
MODEL_FILE_NAME = "hr_attrition_model.pkl"


# -----------------------------
# Streamlit compatibility helpers
# -----------------------------
def show_dataframe(df: pd.DataFrame, **kwargs) -> None:
    """Display dataframe using the new width API, with fallback for older Streamlit."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        # Fallback only for older Streamlit versions that do not support width="stretch".
        st.dataframe(df, use_container_width=True, **kwargs)


def full_width_button(label: str, **kwargs) -> bool:
    """Create a full-width button using the new width API."""
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, use_container_width=True, **kwargs)


def full_width_download_button(**kwargs) -> bool:
    """Create a full-width download button using the new width API."""
    try:
        return st.download_button(width="stretch", **kwargs)
    except TypeError:
        return st.download_button(use_container_width=True, **kwargs)


# -----------------------------
# Model loading helpers
# -----------------------------
def find_model_file() -> Path | None:
    """Find the model file in the same folder as this app."""
    app_dir = Path(__file__).resolve().parent
    candidate = app_dir / MODEL_FILE_NAME
    return candidate if candidate.exists() else None


@st.cache_resource(show_spinner=False)
def load_artifact_from_path(path: str):
    """Load the saved pickle/joblib artifact from disk."""
    return joblib.load(path)


def load_uploaded_artifact(uploaded_file):
    """Load model from an uploaded PKL file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        return joblib.load(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def get_model_classes(model) -> list[Any] | None:
    """Extract model classes from direct estimators or sklearn pipelines."""
    if hasattr(model, "classes_"):
        return list(model.classes_)

    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return list(step.classes_)

    if hasattr(model, "steps"):
        for _, step in reversed(model.steps):
            if hasattr(step, "classes_"):
                return list(step.classes_)

    return None


def infer_feature_columns(model) -> list[str]:
    """Try to infer feature names if they were not saved separately."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    if hasattr(model, "steps"):
        for _, step in model.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return []


def validate_artifact(artifact):
    """Return normalized model metadata from either a dict artifact or direct model."""
    if isinstance(artifact, dict):
        model = artifact.get("model")
        feature_columns = artifact.get("feature_columns") or []
        feature_schema = artifact.get("feature_schema") or {}
        numeric_features = artifact.get("numeric_features") or []
        categorical_features = artifact.get("categorical_features") or []
        prediction_labels = artifact.get("prediction_labels") or {0: "No", 1: "Yes"}
        model_name = artifact.get("model_name") or type(model).__name__
        metrics = artifact.get("best_metrics") or artifact.get("metrics") or {}
    else:
        model = artifact
        feature_columns = []
        feature_schema = {}
        numeric_features = []
        categorical_features = []
        prediction_labels = {0: "No", 1: "Yes"}
        model_name = type(model).__name__
        metrics = {}

    if model is None:
        raise ValueError("The pickle file does not contain a valid model.")

    if not feature_columns:
        feature_columns = infer_feature_columns(model)

    if not feature_columns:
        raise ValueError(
            "Feature columns were not found in the pickle file. "
            "Please save the model with `feature_columns`, or update the app manually."
        )

    return {
        "model": model,
        "feature_columns": list(feature_columns),
        "feature_schema": feature_schema,
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "prediction_labels": prediction_labels,
        "model_name": model_name,
        "metrics": metrics,
    }


# -----------------------------
# Data cleaning and display helpers
# -----------------------------
def is_missing(value: Any) -> bool:
    """Safely detect missing scalar values."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except Exception:
        pass
    return False


def stringify_cell(value: Any) -> str:
    """Convert complex or mixed values into safe strings for Streamlit/PyArrow display."""
    if is_missing(value):
        return ""
    if isinstance(value, dict):
        return ", ".join(f"{key}: {val}" for key, val in value.items())
    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value))
    return str(value)


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed object columns to strings so Streamlit/PyArrow can display safely."""
    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == "object":
            safe_df[col] = safe_df[col].map(stringify_cell)
    return safe_df


def coerce_dataframe_for_model(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Coerce CSV input values to match the saved feature schema when possible."""
    clean_df = df.copy()

    for feature, info in schema.items():
        if feature not in clean_df.columns:
            continue

        feature_type = info.get("type")
        if feature_type == "numeric":
            clean_df[feature] = pd.to_numeric(clean_df[feature], errors="coerce")
            default_value = info.get("default")
            if default_value is None:
                default_value = clean_df[feature].median()
            clean_df[feature] = clean_df[feature].fillna(default_value)
        elif feature_type == "categorical":
            default_value = info.get("default")
            if default_value is None:
                options = info.get("options") or []
                default_value = options[0] if options else ""
            clean_df[feature] = clean_df[feature].fillna(default_value).astype(str)

    return clean_df


def get_positive_probability(model, row_df, positive_class=1):
    """Safely get probability of the attrition class."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(row_df)[0]
    classes = get_model_classes(model) or list(range(len(probabilities)))

    if positive_class in classes:
        positive_index = classes.index(positive_class)
    elif "Yes" in classes:
        positive_index = classes.index("Yes")
    elif "yes" in classes:
        positive_index = classes.index("yes")
    else:
        positive_index = min(1, len(probabilities) - 1)

    return float(probabilities[positive_index])


def format_label(prediction, prediction_labels):
    """Map prediction output to readable label."""
    try:
        key = int(prediction)
        return prediction_labels.get(key, str(prediction))
    except Exception:
        return prediction_labels.get(prediction, str(prediction))


def make_input_widget(feature, schema):
    """Create a Streamlit widget based on schema metadata."""
    feature_info = schema.get(feature, {})
    feature_type = feature_info.get("type")
    options = feature_info.get("options") or []

    if feature_type == "categorical" or options:
        options = [str(option) for option in options]
        if not options:
            options = [""]
        default = str(feature_info.get("default", options[0]))
        index = options.index(default) if default in options else 0
        return st.selectbox(feature, options=options, index=index)

    min_value = feature_info.get("min", 0)
    max_value = feature_info.get("max", 100)
    default = feature_info.get("default", min_value)

    try:
        min_float = float(min_value)
        max_float = float(max_value)
        default_float = float(default)
    except (TypeError, ValueError):
        return st.text_input(feature, value=str(default) if default is not None else "")

    # Avoid invalid number_input ranges.
    if min_float > max_float:
        min_float, max_float = max_float, min_float
    default_float = min(max(default_float, min_float), max_float)

    is_int_like = all(float(v).is_integer() for v in [min_float, max_float, default_float])
    if is_int_like:
        return st.number_input(
            feature,
            min_value=int(min_float),
            max_value=int(max_float),
            value=int(default_float),
            step=1,
        )

    return st.number_input(
        feature,
        min_value=float(min_float),
        max_value=float(max_float),
        value=float(default_float),
        step=0.5,
    )


def predict_dataframe(model, df, prediction_labels):
    """Predict labels and probabilities for a dataframe."""
    predictions = model.predict(df)
    result_df = df.copy()
    result_df["Predicted Attrition"] = [
        format_label(prediction, prediction_labels) for prediction in predictions
    ]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df)
        classes = get_model_classes(model) or list(range(probabilities.shape[1]))

        if 1 in classes:
            positive_index = classes.index(1)
        elif "Yes" in classes:
            positive_index = classes.index("Yes")
        elif "yes" in classes:
            positive_index = classes.index("yes")
        else:
            positive_index = min(1, probabilities.shape[1] - 1)

        result_df["Attrition Probability (%)"] = np.round(
            probabilities[:, positive_index] * 100, 2
        )

    return result_df


# -----------------------------
# Header
# -----------------------------
st.title("👥 HR Employee Attrition Prediction")
st.caption("Predict whether an employee is likely to leave the company using the trained ML model.")

with st.sidebar:
    st.header("📦 Model Loader")
    st.write("Place `hr_attrition_model.pkl` in the same folder as this app, or upload it below.")

    uploaded_model = st.file_uploader("Upload PKL model file", type=["pkl"])

    st.divider()
    st.markdown("### Run command")
    st.code("streamlit run streamlit_hr_attrition_app_corrected.py", language="bash")


# -----------------------------
# Load model
# -----------------------------
try:
    if uploaded_model is not None:
        raw_artifact = load_uploaded_artifact(uploaded_model)
        model_source = "Uploaded PKL file"
    else:
        model_path = find_model_file()
        if model_path is None:
            st.warning(
                "Model file not found. Please keep `hr_attrition_model.pkl` in the same folder "
                "as this Streamlit file, or upload it from the sidebar."
            )
            st.stop()

        raw_artifact = load_artifact_from_path(str(model_path))
        model_source = model_path.name

    meta = validate_artifact(raw_artifact)

except Exception as e:
    st.error("Unable to load the model file.")
    st.write("Error details:")
    st.code(str(e))

    st.info(
        "If you see an error like `MT19937 is not a known BitGenerator module`, "
        "it is usually caused by a NumPy / scikit-learn version mismatch. "
        "Create a fresh environment and reinstall the requirements."
    )

    st.code(
        """
pip install --upgrade pip
pip install streamlit pandas numpy scipy scikit-learn joblib pyarrow
streamlit run streamlit_hr_attrition_app_corrected.py
        """.strip(),
        language="bash",
    )
    st.stop()


model = meta["model"]
feature_columns = meta["feature_columns"]
feature_schema = meta["feature_schema"]
prediction_labels = meta["prediction_labels"]
model_name = meta["model_name"]
metrics = meta["metrics"]


# -----------------------------
# Model Overview
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", model_name)

with col2:
    acc = metrics.get("accuracy")
    st.metric("Accuracy", f"{acc * 100:.2f}%" if acc is not None else "N/A")

with col3:
    roc_auc = metrics.get("roc_auc")
    st.metric("ROC-AUC", f"{roc_auc:.3f}" if roc_auc is not None else "N/A")

with col4:
    st.metric("Model Source", model_source)


tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction", "ℹ️ Model Details"])


# -----------------------------
# Single Prediction
# -----------------------------
with tab1:
    st.subheader("Enter Employee Details")

    input_values = {}

    personal_features = [
        "Age",
        "Gender",
        "MaritalStatus",
        "Education",
        "EducationField",
        "DistanceFromHome",
    ]

    job_features = [
        "Department",
        "JobRole",
        "JobLevel",
        "BusinessTravel",
        "OverTime",
        "MonthlyIncome",
        "DailyRate",
        "HourlyRate",
        "MonthlyRate",
        "PercentSalaryHike",
        "PerformanceRating",
        "StockOptionLevel",
    ]

    satisfaction_features = [
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]

    experience_features = [
        "NumCompaniesWorked",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]

    sections = [
        ("Personal Information", personal_features),
        ("Job Information", job_features),
        ("Satisfaction Ratings", satisfaction_features),
        ("Experience Details", experience_features),
    ]

    for section_name, section_features in sections:
        with st.expander(section_name, expanded=(section_name == "Personal Information")):
            cols = st.columns(3)
            for i, feature in enumerate(section_features):
                if feature in feature_columns:
                    with cols[i % 3]:
                        input_values[feature] = make_input_widget(feature, feature_schema)

    remaining_features = [feature for feature in feature_columns if feature not in input_values]
    if remaining_features:
        with st.expander("Other Features"):
            cols = st.columns(3)
            for i, feature in enumerate(remaining_features):
                with cols[i % 3]:
                    input_values[feature] = make_input_widget(feature, feature_schema)

    input_df = pd.DataFrame(
        [[input_values[col] for col in feature_columns]],
        columns=feature_columns,
    )
    input_df = coerce_dataframe_for_model(input_df, feature_schema)

    st.markdown("### Preview Input")
    show_dataframe(make_arrow_safe(input_df))

    predict_btn = full_width_button("Predict Attrition", type="primary")

    if predict_btn:
        try:
            prediction = model.predict(input_df)[0]
            label = format_label(prediction, prediction_labels)
            probability = get_positive_probability(model, input_df)

            if label.lower() == "yes":
                st.markdown(
                    "<div class='danger-box'>⚠️ Prediction: Employee is likely to leave.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='success-box'>✅ Prediction: Employee is not likely to leave.</div>",
                    unsafe_allow_html=True,
                )

            if probability is not None:
                st.progress(min(max(probability, 0.0), 1.0))
                st.metric("Attrition Probability", f"{probability * 100:.2f}%")

            st.markdown(
                """
                <div class='info-box'>
                Note: This prediction is based on historical HR data and should be used as a decision-support tool, not as the only decision factor.
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error("Prediction failed. Please check that the input values match the model training columns.")
            st.code(str(e))


# -----------------------------
# Batch Prediction
# -----------------------------
with tab2:
    st.subheader("Upload CSV for Batch Prediction")

    st.write("Your CSV must contain these columns:")
    st.code(", ".join(feature_columns))

    sample_row = {
        feature: feature_schema.get(feature, {}).get("default", "")
        for feature in feature_columns
    }
    sample_df = pd.DataFrame([sample_row])
    sample_csv = sample_df.to_csv(index=False).encode("utf-8")
    full_width_download_button(
        label="Download Sample CSV Template",
        data=sample_csv,
        file_name="hr_attrition_sample_template.csv",
        mime="text/csv",
    )

    csv_file = st.file_uploader("Upload employee CSV", type=["csv"], key="batch_csv")

    if csv_file is not None:
        try:
            batch_df = pd.read_csv(csv_file)

            missing_cols = [col for col in feature_columns if col not in batch_df.columns]
            if missing_cols:
                st.error("Missing required columns:")
                st.code(", ".join(missing_cols))
            else:
                clean_df = batch_df[feature_columns].copy()
                clean_df = coerce_dataframe_for_model(clean_df, feature_schema)
                predictions_df = predict_dataframe(model, clean_df, prediction_labels)

                st.success("Batch prediction completed.")
                show_dataframe(make_arrow_safe(predictions_df))

                csv_output = predictions_df.to_csv(index=False).encode("utf-8")
                full_width_download_button(
                    label="Download Prediction CSV",
                    data=csv_output,
                    file_name="hr_attrition_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error("Could not process the uploaded CSV.")
            st.code(str(e))


# -----------------------------
# Model Details
# -----------------------------
with tab3:
    st.subheader("Model Details")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Required Features")
        features_df = pd.DataFrame({"Feature": feature_columns})
        show_dataframe(make_arrow_safe(features_df), hide_index=True)

    with col_b:
        st.markdown("#### Best Metrics")
        if metrics:
            metrics_df = pd.DataFrame([metrics]).T.reset_index()
            metrics_df.columns = ["Metric", "Value"]
            show_dataframe(make_arrow_safe(metrics_df), hide_index=True)
        else:
            st.write("No metrics found in the pickle file.")

    st.markdown("#### Feature Schema")
    if feature_schema:
        schema_df = pd.DataFrame(feature_schema).T.reset_index().rename(columns={"index": "Feature"})
        show_dataframe(make_arrow_safe(schema_df), hide_index=True)
    else:
        st.write("No feature schema found.")
