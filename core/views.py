import base64
import csv
import os
from datetime import date
from functools import wraps
from urllib.parse import urlencode
from uuid import uuid4
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from django.contrib import auth
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import Group
from django.core.cache import cache
from django.core.paginator import Paginator
from django.core.exceptions import ValidationError
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from .ml_engine import (
    load_dataset,
    detect_target_column,
)
from .early_warning_training import (
    MODEL_ARTIFACT_FILENAME,
    PREPROCESSING_ARTIFACT_FILENAME,
    MODEL_NAMES,
    SELECTED_MODEL_NAME,
    _prepare_features,
    analyze_early_warning_thresholds,
    explain_learner,
    get_risk_band,
    persist_selected_model,
    predict_learner,
    train_early_warning_models,
)
from .oulad_features import (
    ASSESSMENT_CUTOFF_DAY,
    EARLY_WARNING_DATASET_FILENAME,
    generate_early_warning_dataset,
)
from .recommendation_engine import generate_recommendations
from .models import Intervention, StudentProfile
from .batch_prediction import (
    DOWNLOAD_COLUMNS,
    predict_batch,
    summarise_risk_levels,
)


ROLE_GROUPS = ("ADMIN", "EDUCATOR", "STUDENT")


def ensure_default_roles():
    for role in ROLE_GROUPS:
        Group.objects.get_or_create(name=role)


def get_user_role(user):
    if not user or not getattr(user, "is_authenticated", False):
        return "GUEST"
    group_names = {group.name.upper() for group in user.groups.all()}
    if "ADMIN" in group_names:
        return "ADMIN"
    if "EDUCATOR" in group_names:
        return "EDUCATOR"
    if "STUDENT" in group_names:
        return "STUDENT"
    return "UNASSIGNED"


def access_denied(request, message="You don't have permission to view this page.", required_roles=None):
    return render(
        request,
        "core/access_denied.html",
        {
            "message": message,
            "required_roles": required_roles or ("ADMIN", "EDUCATOR", "STUDENT"),
        },
        status=403,
    )


def role_required(*allowed_roles):
    def decorator(view_func):
        @wraps(view_func)
        @login_required(login_url="login")
        def wrapped(request, *args, **kwargs):
            ensure_default_roles()
            if not allowed_roles:
                return view_func(request, *args, **kwargs)
            if get_user_role(request.user) not in allowed_roles:
                return access_denied(
                    request,
                    "You don't have permission to view this page.",
                    list(allowed_roles),
                )
            return view_func(request, *args, **kwargs)

        return wrapped

    return decorator


def get_student_profile_for_user(user):
    try:
        return user.student_profile
    except StudentProfile.DoesNotExist:
        return None


def student_record_access_allowed(request, learner_id=None):
    if get_user_role(request.user) != "STUDENT":
        return True
    profile = get_student_profile_for_user(request.user)
    if profile is None:
        return False
    if learner_id is None:
        return True
    return str(learner_id) == str(profile.student_id)


def login_view(request):
    ensure_default_roles()
    if request.user.is_authenticated and request.method != "POST":
        return redirect("upload_dataset")

    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        redirect_target = request.POST.get("next") or request.GET.get("next") or reverse("upload_dataset")
        if form.is_valid():
            user = form.get_user()
            auth.login(request, user)
            return redirect(redirect_target)
        return render(
            request,
            "core/login.html",
            {"form": form, "next_url": redirect_target},
        )

    return render(
        request,
        "core/login.html",
        {"form": AuthenticationForm(), "next_url": request.GET.get("next") or reverse("upload_dataset")},
    )


def logout_view(request):
    auth.logout(request)
    return redirect("login")


def settings_view(request):
    if not request.user.is_authenticated:
        return redirect("login")
    return render(
        request,
        "core/settings.html",
        {
            "user_role": get_user_role(request.user),
            "username": request.user.username,
        },
    )


def help_view(request):
    if not request.user.is_authenticated:
        return redirect("login")
    return render(
        request,
        "core/help.html",
        {"user_role": get_user_role(request.user)},
    )


def access_denied_view(request):
    return access_denied(request, "You don't have permission to view this page.")


def health_check(request):
    return HttpResponse("OK", content_type="text/plain; charset=utf-8")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_data_directory():

    data_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ),
        "data"
    )

    os.makedirs(
        data_dir,
        exist_ok=True
    )

    return data_dir


def get_dataset_path():

    return os.path.join(
        get_data_directory(),
        "uploaded_dataset.csv"
    )


def get_processed_dataset_path():

    return os.path.join(
        get_data_directory(),
        "processed_dataset.csv"
    )


def has_uploaded_dataset(request):

    return bool(
        request.session.get("generic_dataset_uploaded")
        and os.path.exists(get_dataset_path())
    )


def get_current_dataset_context(request):

    if not has_uploaded_dataset(request):
        return {}

    dataset_path = get_dataset_path()
    df = load_dataset(dataset_path)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(value)
            for column, value in df.isnull().sum().items()
        },
        "target": detect_target_column(df),
        "dataset_name": request.session.get(
            "generic_dataset_name",
            "uploaded_dataset.csv",
        ),
        "dataset_available": True,
    }


def get_early_warning_dataset_path():

    return os.path.join(
        get_data_directory(),
        EARLY_WARNING_DATASET_FILENAME
    )


def get_early_warning_artifact_directory():

    artifact_directory = os.path.join(
        get_data_directory(),
        "early_warning_artifacts",
    )
    os.makedirs(artifact_directory, exist_ok=True)
    return artifact_directory


def generate_roc_curve_image(y_true, probabilities, model_name):

    try:

        if probabilities is None:
            return None

        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)

        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")

        unique_classes = sorted(pd.Series(y_true).unique().tolist())

        if len(unique_classes) == 2:

            positive_class = 1 if 1 in unique_classes else unique_classes[-1]
            positive_index = 1 if probabilities.shape[1] > 1 else 0

            if probabilities.shape[1] > 1:
                score_values = probabilities[:, positive_index]
            else:
                score_values = probabilities[:, 0]

            fpr, tpr, _ = roc_curve(y_true, score_values, pos_label=positive_class)
            roc_auc = roc_auc_score(y_true, score_values)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

        else:

            for class_label in unique_classes:
                if class_label >= probabilities.shape[1]:
                    continue

                y_true_binary = (y_true == class_label).astype(int)
                class_probabilities = probabilities[:, class_label]
                fpr, tpr, _ = roc_curve(
                    y_true_binary,
                    class_probabilities,
                )
                class_auc = roc_auc_score(
                    y_true_binary,
                    class_probabilities,
                )
                plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {class_auc:.3f})")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    except Exception:

        return None


def _ensure_early_warning_assets():
    """Create the Day-60 dataset and ensure the persisted model artifacts exist."""
    dataset_path = get_early_warning_dataset_path()
    if not os.path.exists(dataset_path):
        generate_early_warning_dataset(
            get_data_directory(),
            dataset_path,
            assessment_cutoff_day=ASSESSMENT_CUTOFF_DAY,
        )
    artifact_directory = get_early_warning_artifact_directory()
    model_path = os.path.join(artifact_directory, MODEL_ARTIFACT_FILENAME)
    preprocessing_path = os.path.join(
        artifact_directory,
        PREPROCESSING_ARTIFACT_FILENAME,
    )
    if not os.path.exists(model_path) or not os.path.exists(preprocessing_path):
        persist_selected_model(dataset_path, artifact_directory)
    return dataset_path, artifact_directory


def _build_roc_curve_data(dataset_path):
    """Return model ROC data for charting as actual evaluated measurements."""
    table = pd.read_csv(dataset_path)
    y = table["dropout"].astype("int8")
    feature_columns = [
        column
        for column in table.columns
        if column not in ("id_student", "dropout") and column != "final_result"
    ]
    X, _ = _prepare_features(table, feature_columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    model_specs = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }
    colors = {
        "Logistic Regression": "#2563eb",
        "Decision Tree": "#f59e0b",
        "Random Forest": "#34d399",
    }
    curves = []
    for model_name, model in model_specs.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        curves.append(
            {
                "name": model_name,
                "auc": round(roc_auc_score(y_test, probabilities) * 100, 2),
                "color": colors.get(model_name, "#2563eb"),
                "points": [
                    {
                        "x": round(float(x), 4),
                        "y": round(float(y), 4),
                    }
                    for x, y in zip(fpr, tpr)
                ],
            }
        )
    return curves


def _current_dashboard_risk_summary():
    try:
        dataset_path, artifact_directory = _ensure_early_warning_assets()
        table = pd.read_csv(dataset_path)
        results = predict_batch(table, artifact_directory)
        return summarise_risk_levels(results)
    except Exception:
        return {"total_learners": 0, "high_risk": 0, "medium_risk": 0, "low_risk": 0}


def _dashboard_attention_learners():
    try:
        _, results = _dashboard_prediction_results()
        ranked = sorted(
            results,
            key=lambda item: (RISK_ORDER.get(item["risk_level"], -1), item["dropout_probability"]),
            reverse=True,
        )
        return ranked[:5]
    except Exception:
        return []


def _module_summary(results):
    grouped = {}
    for row in results:
        module = row.get("code_module", "Unknown")
        grouped.setdefault(
            module,
            {"module": module, "learners": 0, "High": 0, "Medium": 0, "Low": 0},
        )
        grouped[module]["learners"] += 1
        risk_level = row.get("risk_level", "Low")
        if risk_level in grouped[module]:
            grouped[module][risk_level] += 1
    return sorted(
        grouped.values(),
        key=lambda item: (-item["learners"], item["module"]),
    )


def _model_performance_context():
    dataset_path, artifact_directory = _ensure_early_warning_assets()
    model_training = train_early_warning_models(dataset_path)
    diagnostics = analyze_early_warning_thresholds(dataset_path)
    comparison = [
        {
            "label": "Accuracy",
            "series": [
                {"name": item["name"], "value": item["accuracy"], "color": color}
                for item, color in zip(model_training["results"], ("#2563eb", "#f59e0b", "#34d399"))
            ],
        },
        {
            "label": "Precision",
            "series": [
                {"name": item["name"], "value": item["precision"], "color": color}
                for item, color in zip(model_training["results"], ("#2563eb", "#f59e0b", "#34d399"))
            ],
        },
        {
            "label": "Recall",
            "series": [
                {"name": item["name"], "value": item["recall"], "color": color}
                for item, color in zip(model_training["results"], ("#2563eb", "#f59e0b", "#34d399"))
            ],
        },
        {
            "label": "F1",
            "series": [
                {"name": item["name"], "value": item["f1"], "color": color}
                for item, color in zip(model_training["results"], ("#2563eb", "#f59e0b", "#34d399"))
            ],
        },
        {
            "label": "ROC-AUC",
            "series": [
                {"name": item["name"], "value": item["roc_auc"], "color": color}
                for item, color in zip(model_training["results"], ("#2563eb", "#f59e0b", "#34d399"))
            ],
        },
    ]
    return {
        "results": model_training["results"],
        "diagnostics": diagnostics,
        "comparison_chart": comparison,
        "roc_curve_data": _build_roc_curve_data(dataset_path),
        "selected_model": SELECTED_MODEL_NAME,
        "selected_model_explanation": (
            "Logistic Regression is the current project-selected model because it achieved the highest measured accuracy and ROC-AUC among the evaluated models in the current experiment. "
            "These results are based on the current held-out evaluation dataset and may differ when the model is applied to data from another institution or learner population."
        ),
        "model_limits": [
            "Day 60 is an experimental early-warning cutoff.",
            "Current OULAD data does not contain learner-level studentVle activity.",
            "Current predictions are based on the available early-warning features.",
            "Risk thresholds are initial project-defined thresholds.",
            "Feature contributions are model indicators, not causal explanations.",
            "Historical dataset performance does not guarantee the same performance on new institutions.",
            "Recommendations are support suggestions, not guaranteed interventions.",
        ],
    }


def _reports_context():
    dataset_path, artifact_directory = _ensure_early_warning_assets()
    results = predict_batch(pd.read_csv(dataset_path), artifact_directory)
    summary = summarise_risk_levels(results)
    intervention_summary = _intervention_overview()
    module_summary = _module_summary(results)
    distribution = [
        {
            "label": label,
            "count": summary[f"{label.lower()}_risk"],
            "percentage": round((summary[f"{label.lower()}_risk"] / summary["total_learners"] * 100) if summary["total_learners"] else 0, 2),
        }
        for label in ("High", "Medium", "Low")
    ]
    return {
        "summary": summary,
        "distribution": distribution,
        "intervention_summary": intervention_summary,
        "module_summary": module_summary,
        "report_title": "Current Early-Warning System Overview",
    }


# =========================================================
# 1. UPLOAD + DATASET ANALYSIS
# =========================================================

@role_required("ADMIN", "EDUCATOR", "STUDENT")
def upload_dataset(request):

    user_role = get_user_role(request.user)
    if user_role == "EDUCATOR":
        if request.method == "POST":
            return access_denied(request)
        return redirect("student_risk_dashboard")
    if user_role == "STUDENT":
        if request.method == "POST":
            return access_denied(request)
        return redirect("individual_prediction")

    context = get_current_dataset_context(request)
    context["intervention_overview"] = _intervention_overview()
    context["risk_summary"] = _current_dashboard_risk_summary()
    context["risk_distribution"] = _distribution(context["risk_summary"])
    context["attention_learners"] = _dashboard_attention_learners()
    context["attention_summary"] = {
        "high_risk_learners": context["risk_summary"].get("high_risk", 0),
        "follow_up_due_soon": max(
            0,
            context["intervention_overview"].get("pending", 0)
            + context["intervention_overview"].get("in_progress", 0),
        ),
        "urgent_interventions": context["intervention_overview"].get("overdue", 0),
    }

    if request.method == "POST":

        uploaded_file = request.FILES.get(
            "dataset"
        )

        # -------------------------------------------------
        # Check file
        # -------------------------------------------------

        if not uploaded_file:

            context["error"] = (
                "Please select a CSV file."
            )

            return render(
                request,
                "core/upload.html",
                context
            )

        if not uploaded_file.name.lower().endswith(
            ".csv"
        ):

            context["error"] = (
                "Please upload a CSV file."
            )

            return render(
                request,
                "core/upload.html",
                context
            )

        try:

            # -------------------------------------------------
            # Save uploaded dataset
            # -------------------------------------------------

            dataset_path = get_dataset_path()

            with open(
                dataset_path,
                "wb+"
            ) as destination:

                for chunk in uploaded_file.chunks():

                    destination.write(
                        chunk
                    )

            # -------------------------------------------------
            # Load dataset
            # -------------------------------------------------

            df = load_dataset(
                dataset_path
            )

            # -------------------------------------------------
            # Detect target column
            # -------------------------------------------------

            target = detect_target_column(
                df
            )

            # -------------------------------------------------
            # Dataset information
            # -------------------------------------------------

            context["rows"] = int(
                df.shape[0]
            )

            context["columns"] = int(
                df.shape[1]
            )

            context["column_names"] = list(
                df.columns
            )

            # -------------------------------------------------
            # Data types
            # -------------------------------------------------

            context["data_types"] = {

                column: str(dtype)

                for column, dtype
                in df.dtypes.items()

            }

            # -------------------------------------------------
            # Missing values
            # -------------------------------------------------

            context["missing_values"] = {

                column: int(value)

                for column, value
                in df.isnull().sum().items()

            }

            # -------------------------------------------------
            # Target
            # -------------------------------------------------

            context["target"] = target
            context["dataset_name"] = uploaded_file.name

            context["success"] = True
            request.session["generic_dataset_uploaded"] = True
            request.session["generic_dataset_name"] = uploaded_file.name
            request.session.modified = True

            processed_path = get_processed_dataset_path()
            if os.path.exists(processed_path):
                os.remove(processed_path)

        except Exception as e:

            context["error"] = (
                f"Could not analyse dataset: {str(e)}"
            )

    return render(
        request,
        "core/upload.html",
        context
    )


@role_required("ADMIN")
def remove_dataset(request):

    if request.method == "POST":
        request.session.pop("generic_dataset_uploaded", None)
        request.session.pop("generic_dataset_name", None)
        request.session.modified = True

        for path in (
            get_dataset_path(),
            get_processed_dataset_path(),
        ):
            if os.path.exists(path):
                os.remove(path)

    return redirect("upload_dataset")


# =========================================================
# 2. DATA PREPROCESSING
# =========================================================

@role_required("ADMIN")
def data_preprocessing(request):

    context = {}

    dataset_path = get_dataset_path()

    # -----------------------------------------------------
    # Check uploaded dataset
    # -----------------------------------------------------

    if not has_uploaded_dataset(request) or not os.path.exists(dataset_path):

        context["error"] = (
            "No Dataset Uploaded. Please upload a CSV dataset "
            "from the Dashboard before starting preprocessing."
        )

        return render(
            request,
            "core/preprocessing.html",
            context
        )

    try:

        # -------------------------------------------------
        # Load dataset
        # -------------------------------------------------

        df = pd.read_csv(
            dataset_path
        )

        # -------------------------------------------------
        # Detect target
        # -------------------------------------------------

        target = detect_target_column(
            df
        )

        # -------------------------------------------------
        # Original information
        # -------------------------------------------------

        original_rows = len(df)

        original_columns = len(
            df.columns
        )

        # -------------------------------------------------
        # Missing values
        # -------------------------------------------------

        missing_before = int(
            df.isnull().sum().sum()
        )

        for column in df.columns:

            if df[column].isnull().sum() > 0:

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    df[column] = (
                        df[column].fillna(
                            df[column].median()
                        )
                    )

                else:

                    mode = (
                        df[column].mode()
                    )

                    if not mode.empty:

                        df[column] = (
                            df[column].fillna(
                                mode.iloc[0]
                            )
                        )

                    else:

                        df[column] = (
                            df[column].fillna(
                                "Unknown"
                            )
                        )

        missing_after = int(
            df.isnull().sum().sum()
        )

        # -------------------------------------------------
        # Remove duplicates
        # -------------------------------------------------

        duplicates_before = int(
            df.duplicated().sum()
        )

        df = df.drop_duplicates()

        duplicates_removed = (
            duplicates_before
        )

        # -------------------------------------------------
        # Encode categorical features
        # -------------------------------------------------

        categorical_columns = []

        for column in df.columns:

            if column == target:

                continue

            if df[column].dtype == "object":

                categorical_columns.append(
                    column
                )

                df[column] = pd.factorize(
                    df[column].astype(str)
                )[0]

        # -------------------------------------------------
        # Encode target column
        # -------------------------------------------------

        target_mapping = {}

        if target in df.columns:

            if df[target].dtype == "object":

                target_values = (
                    df[target].astype(str)
                )

                encoded_target, unique_values = (
                    pd.factorize(
                        target_values
                    )
                )

                df[target] = (
                    encoded_target
                )

                target_mapping = {

                    int(index): str(value)

                    for index, value
                    in enumerate(
                        unique_values
                    )

                }

        # -------------------------------------------------
        # Save processed dataset
        # -------------------------------------------------

        processed_path = (
            get_processed_dataset_path()
        )

        df.to_csv(
            processed_path,
            index=False
        )

        # -------------------------------------------------
        # Frontend information
        # -------------------------------------------------

        context["success"] = True

        context["original_rows"] = (
            original_rows
        )

        context["original_columns"] = (
            original_columns
        )

        context["processed_rows"] = int(
            df.shape[0]
        )

        context["processed_columns"] = int(
            df.shape[1]
        )

        context["target"] = target

        context["missing_before"] = (
            missing_before
        )

        context["missing_after"] = (
            missing_after
        )

        context["duplicates_removed"] = (
            duplicates_removed
        )

        context["categorical_columns"] = (
            categorical_columns
        )

        context["target_mapping"] = (
            target_mapping
        )

        context["processed_columns_list"] = (
            list(df.columns)
        )

    except Exception as e:

        context["error"] = (
            f"Could not preprocess dataset: {str(e)}"
        )

    return render(
        request,
        "core/preprocessing.html",
        context
    )


# =========================================================
# 3. MODEL TRAINING
# =========================================================

@role_required("ADMIN")
def model_training(request):

    context = {}

    processed_path = (
        get_processed_dataset_path()
    )

    # -----------------------------------------------------
    # Check processed dataset
    # -----------------------------------------------------

    if (
        not has_uploaded_dataset(request)
        or not os.path.exists(processed_path)
    ):

        context["error"] = (
            "No Dataset Uploaded. Please upload a CSV dataset "
            "from the Dashboard before training a model."
        )

        return render(
            request,
            "core/model_training.html",
            context
        )

    try:

        # -------------------------------------------------
        # Load processed dataset
        # -------------------------------------------------

        df = pd.read_csv(
            processed_path
        )

        # -------------------------------------------------
        # Detect target
        # -------------------------------------------------

        target = detect_target_column(
            df
        )

        if target not in df.columns:

            context["error"] = (
                "Target column could not be found."
            )

            return render(
                request,
                "core/model_training.html",
                context
            )

        # -------------------------------------------------
        # Separate features and target
        # -------------------------------------------------

        X = df.drop(
            columns=[target]
        ).copy()

        y = df[target].copy()

        # =================================================
        # IMPORTANT FIX
        # =================================================
        # Convert every text/categorical feature
        # into numbers before sending it to ML models.
        #
        # This prevents errors such as:
        #
        # could not convert string to float: 'BBB'
        #
        # =================================================

        for column in X.columns:

            if not pd.api.types.is_numeric_dtype(
                X[column]
            ):

                X[column] = pd.factorize(
                    X[column].astype(str)
                )[0]

        # -------------------------------------------------
        # Convert target to numeric
        # -------------------------------------------------

        if not pd.api.types.is_numeric_dtype(
            y
        ):

            target_encoder = LabelEncoder()

            y = target_encoder.fit_transform(
                y.astype(str)
            )

        # -------------------------------------------------
        # Replace remaining missing values
        # -------------------------------------------------

        X = X.fillna(
            0
        )

        y = pd.Series(
            y
        ).fillna(
            0
        )

        # -------------------------------------------------
        # Make sure target is integer
        # -------------------------------------------------

        y = y.astype(
            int
        )

        # -------------------------------------------------
        # Train / Test split
        # -------------------------------------------------

        unique_classes = (
            len(
                pd.Series(y).unique()
            )
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(

                X,

                y,

                test_size=0.20,

                random_state=42,

                stratify=(
                    y
                    if unique_classes > 1
                    else None
                )
            )
        )

        # =================================================
        # MACHINE LEARNING MODELS
        # =================================================

        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                ),

        }

        results = []

        # =================================================
        # TRAIN MODELS
        # =================================================

        for model_name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            roc_auc = None
            roc_curve_image = None

            try:

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(X_test)

                    if len(pd.Series(y_test).unique()) == 2:
                        if probabilities.shape[1] > 1:
                            positive_class = 1 if 1 in pd.Series(y_test).unique() else sorted(pd.Series(y_test).unique())[-1]
                            roc_auc = roc_auc_score(
                                y_test,
                                probabilities[:, 1] if positive_class == 1 else probabilities[:, 0],
                            )
                        else:
                            roc_auc = roc_auc_score(
                                y_test,
                                probabilities[:, 0],
                            )
                    else:
                        roc_auc = roc_auc_score(
                            y_test,
                            probabilities,
                            multi_class="ovr",
                            average="macro",
                        )

                    roc_curve_image = generate_roc_curve_image(
                        y_test,
                        probabilities,
                        model_name,
                    )

            except Exception:

                roc_auc = None
                roc_curve_image = None

            results.append({

                "name": model_name,

                "accuracy": round(
                    accuracy * 100,
                    2
                ),

                "precision": round(
                    precision * 100,
                    2
                ),

                "recall": round(
                    recall * 100,
                    2
                ),

                "f1": round(
                    f1 * 100,
                    2
                ),

                "roc_auc": round(
                    roc_auc * 100,
                    2
                ) if roc_auc is not None else None,

                "roc_curve_image": roc_curve_image,

            })

        # =================================================
        # SELECT BEST MODEL
        # =================================================

        best_model = max(
            results,
            key=lambda x: x["accuracy"]
        )

        # =================================================
        # SEND RESULTS TO FRONTEND
        # =================================================

        context["success"] = True

        context["target"] = target

        context["total_records"] = (
            len(df)
        )

        context["training_records"] = (
            len(X_train)
        )

        context["testing_records"] = (
            len(X_test)
        )

        # Keep the names expected by the existing training template.
        context["training_rows"] = (
            context["training_records"]
        )

        context["testing_rows"] = (
            context["testing_records"]
        )

        context["feature_count"] = (
            X.shape[1]
        )

        context["results"] = (
            results
        )

        context["best_model"] = (
            best_model["name"]
        )

        context["best_accuracy"] = (
            best_model["accuracy"]
        )

        context["best_precision"] = (
            best_model["precision"]
        )

        context["best_recall"] = (
            best_model["recall"]
        )

        context["best_f1"] = (
            best_model["f1"]
        )

    except Exception as e:

        context["error"] = (
            f"Could not train models: {str(e)}"
        )

    return render(
        request,
        "core/model_training.html",
        context
    )


@role_required("ADMIN")
def early_warning_training(request):

    context = {
        "cutoff_day": ASSESSMENT_CUTOFF_DAY,
    }

    try:
        dataset_path = get_early_warning_dataset_path()
        generate_early_warning_dataset(
            get_data_directory(),
            dataset_path,
            assessment_cutoff_day=ASSESSMENT_CUTOFF_DAY,
        )
        context.update(train_early_warning_models(dataset_path))
        context["diagnostics"] = analyze_early_warning_thresholds(dataset_path)
        context.update(
            persist_selected_model(
                dataset_path,
                get_early_warning_artifact_directory(),
            )
        )
        context["success"] = True
    except Exception as error:
        context["error"] = f"Could not train early-warning models: {error}"

    return render(
        request,
        "core/early_warning_training.html",
        context,
    )


@role_required("ADMIN", "EDUCATOR")
def model_performance(request):
    """Display the measured performance of the existing early-warning model set."""
    context = {"cutoff_day": ASSESSMENT_CUTOFF_DAY}
    try:
        context.update(_model_performance_context())
        context["success"] = True
    except Exception as error:
        context["error"] = f"Could not load model performance: {error}"
    return render(request, "core/model_performance.html", context)


@role_required("ADMIN", "EDUCATOR")
def reports(request):
    """Summarise the current early-warning and intervention system state."""
    context = {}
    try:
        context.update(_reports_context())
        context["success"] = True
    except Exception as error:
        context["error"] = f"Could not load reports: {error}"
    return render(request, "core/reports.html", context)


@role_required("ADMIN", "EDUCATOR")
def download_report(request):
    """Download a concise CSV summary of aggregate report data."""
    try:
        context = _reports_context()
    except Exception:
        return redirect("reports")

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="early_warning_report.csv"'
    writer = csv.writer(response)
    writer.writerow(["section", "metric", "value", "risk_level", "module"])

    for label in ("High", "Medium", "Low"):
        summary = context["summary"]
        count_value = summary[f"{label.lower()}_risk"]
        percentage = 0 if not summary["total_learners"] else round((count_value / summary["total_learners"]) * 100, 2)
        writer.writerow(["risk_distribution", "count", count_value, label, ""])
        writer.writerow(["risk_distribution", "percentage", percentage, label, ""])

    for metric, value in {
        "learners_analysed": context["summary"]["total_learners"],
        "interventions": context["intervention_summary"]["total_interventions"],
        "pending_interventions": context["intervention_summary"]["pending"],
        "in_progress_interventions": context["intervention_summary"]["in_progress"],
        "completed_interventions": context["intervention_summary"]["completed"],
        "overdue_interventions": context["intervention_summary"]["overdue"],
    }.items():
        writer.writerow(["overview", metric, value, "", ""])

    for item in context["module_summary"]:
        writer.writerow([
            "module_summary",
            "learners",
            item["learners"],
            "",
            item["module"],
        ])
        writer.writerow([
            "module_summary",
            "high_risk",
            item["High"],
            "High",
            item["module"],
        ])
        writer.writerow([
            "module_summary",
            "medium_risk",
            item["Medium"],
            "Medium",
            item["module"],
        ])
        writer.writerow([
            "module_summary",
            "low_risk",
            item["Low"],
            "Low",
            item["module"],
        ])

    return response


@role_required("ADMIN", "EDUCATOR", "STUDENT")
def individual_prediction(request):

    context = {
        "cutoff_day": ASSESSMENT_CUTOFF_DAY,
        "intervention_statuses": Intervention.Status.choices,
        "intervention_saved": request.GET.get("intervention_saved") == "1",
        "intervention_updated": request.GET.get("intervention_updated") == "1",
        "update_error": request.GET.get("update_error") == "1",
    }
    user_role = get_user_role(request.user)
    profile = None
    if user_role == "STUDENT":
        profile = get_student_profile_for_user(request.user)
        if profile is None:
            return access_denied(request, "You can only access your own learner profile.")
        requested_student_id = request.GET.get("id_student") or request.POST.get("id_student") or str(profile.student_id)
        if str(requested_student_id) != str(profile.student_id):
            return access_denied(request, "You can only access your own learner profile.")
    dataset_path = get_early_warning_dataset_path()

    try:
        if not os.path.exists(dataset_path):
            generate_early_warning_dataset(
                get_data_directory(),
                dataset_path,
                assessment_cutoff_day=ASSESSMENT_CUTOFF_DAY,
            )
        table = pd.read_csv(dataset_path)
        options = table[
            ["id_student", "code_module", "code_presentation"]
        ].drop_duplicates().sort_values(
            ["id_student", "code_module", "code_presentation"]
        )
        if user_role == "STUDENT":
            options = options[
                options["id_student"].astype(str) == str(profile.student_id)
            ]
        context["learner_options"] = options.to_dict("records")

        artifact_directory = get_early_warning_artifact_directory()
        model_path = os.path.join(artifact_directory, MODEL_ARTIFACT_FILENAME)
        preprocessing_path = os.path.join(
            artifact_directory,
            PREPROCESSING_ARTIFACT_FILENAME,
        )
        if not os.path.exists(model_path) or not os.path.exists(preprocessing_path):
            persist_selected_model(dataset_path, artifact_directory)

        selection_source = request.POST if request.method == "POST" else request.GET
        student_requested_id = request.GET.get("id_student") or request.POST.get("id_student") or (str(profile.student_id) if user_role == "STUDENT" else "")
        has_selection = request.method == "POST" or any(
            selection_source.get(field, "")
            for field in ("id_student", "code_module", "code_presentation")
        ) or student_requested_id
        if user_role == "STUDENT" and not any(
            selection_source.get(field, "")
            for field in ("code_module", "code_presentation")
        ):
            own_options = options[
                options["id_student"].astype(str) == str(profile.student_id)
            ]
            if not own_options.empty:
                own_option = own_options.iloc[0]
                learner_key = (
                    str(profile.student_id),
                    own_option["code_module"],
                    own_option["code_presentation"],
                )
                has_selection = True
            else:
                learner_key = (str(profile.student_id), "", "")
        else:
            learner_key = (
                student_requested_id or selection_source.get("id_student", ""),
                selection_source.get("code_module", ""),
                selection_source.get("code_presentation", ""),
            )
        if has_selection:
            context["prediction"] = predict_learner(
                table,
                learner_key,
                artifact_directory,
            )
            context["current_risk_percentage"] = (
                context["prediction"]["dropout_probability"] * 100
            )
            context["explanation"] = explain_learner(
                table,
                learner_key,
                artifact_directory,
            )
            context["recommendations"] = _learner_recommendations(
                context["prediction"], context["explanation"]
            )
            context["interventions"] = _learner_intervention_history(
                learner_key,
                context["prediction"]["dropout_probability"],
            )
    except (ValueError, FileNotFoundError) as error:
        context["error"] = str(error)

    return render(
        request,
        "core/individual_prediction.html",
        context,
    )


def _learner_recommendations(prediction, explanation):
    """Apply the existing recommendation rules to a persisted prediction."""
    feature_names = (
        "registration_offset_days",
        "early_assessment_count",
        "early_assessment_score_mean",
        "early_assessment_score_std",
        "early_assessment_scheduled_count",
        "early_assessment_submission_rate",
    )
    return generate_recommendations(
        {
            "student_id": prediction["row"]["id_student"],
            "module": prediction["row"]["code_module"],
            "presentation": prediction["row"]["code_presentation"],
            "dropout_probability": prediction["dropout_probability"],
            "risk_level": prediction["risk_band"],
            "early_warning_features": {
                name: prediction["row"].get(name) for name in feature_names
            },
            "higher_risk_contributions": explanation["higher"],
            "lower_risk_contributions": explanation["lower"],
        }
    )


@role_required("ADMIN", "EDUCATOR", "STUDENT")
def recommendations(request):
    """Show support suggestions for one existing early-warning learner."""
    context = {"cutoff_day": ASSESSMENT_CUTOFF_DAY}
    user_role = get_user_role(request.user)
    profile = None
    if user_role == "STUDENT":
        profile = get_student_profile_for_user(request.user)
        if profile is None:
            return access_denied(request, "You can only view your own recommendations.")
        requested_student_id = request.GET.get("id_student") or str(profile.student_id)
        if str(requested_student_id) != str(profile.student_id):
            return access_denied(request, "You can only view your own recommendations.")
    dataset_path = get_early_warning_dataset_path()

    try:
        if not os.path.exists(dataset_path):
            generate_early_warning_dataset(
                get_data_directory(),
                dataset_path,
                assessment_cutoff_day=ASSESSMENT_CUTOFF_DAY,
            )
        table = pd.read_csv(dataset_path)
        options = table[
            ["id_student", "code_module", "code_presentation"]
        ].drop_duplicates().sort_values(
            ["id_student", "code_module", "code_presentation"]
        )
        if user_role == "STUDENT":
            options = options[
                options["id_student"].astype(str) == str(profile.student_id)
            ]
        context["learner_options"] = options.to_dict("records")

        artifact_directory = get_early_warning_artifact_directory()
        model_path = os.path.join(artifact_directory, MODEL_ARTIFACT_FILENAME)
        preprocessing_path = os.path.join(
            artifact_directory,
            PREPROCESSING_ARTIFACT_FILENAME,
        )
        if not os.path.exists(model_path) or not os.path.exists(preprocessing_path):
            persist_selected_model(dataset_path, artifact_directory)

        learner_key = (
            request.GET.get("id_student") or (str(profile.student_id) if user_role == "STUDENT" else ""),
            request.GET.get("code_module", ""),
            request.GET.get("code_presentation", ""),
        )
        if any(learner_key):
            if not all(learner_key):
                raise ValueError("Select a learner/course/presentation combination")
            prediction = predict_learner(table, learner_key, artifact_directory)
            explanation = explain_learner(table, learner_key, artifact_directory)
            context.update(
                {
                    "prediction": prediction,
                    "risk_percentage": prediction["dropout_probability"] * 100,
                    "explanation": explanation,
                    "recommendations": _learner_recommendations(
                        prediction, explanation
                    ),
                }
            )
    except (ValueError, FileNotFoundError) as error:
        context["error"] = str(error)

    return render(request, "core/recommendations.html", context)


DASHBOARD_CACHE_TIMEOUT = 60 * 60
RISK_ORDER = {"High": 2, "Medium": 1, "Low": 0}


def _dashboard_prediction_results():
    """Return cached predictions for the repository's Day-60 learner table."""
    dataset_path = get_early_warning_dataset_path()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            "The Day-60 early-warning dataset is unavailable. Visit "
            "/early-warning-training/ to prepare it."
        )
    artifact_directory = get_early_warning_artifact_directory()
    model_path = os.path.join(artifact_directory, MODEL_ARTIFACT_FILENAME)
    preprocessing_path = os.path.join(
        artifact_directory, PREPROCESSING_ARTIFACT_FILENAME
    )
    if not os.path.exists(model_path) or not os.path.exists(preprocessing_path):
        raise FileNotFoundError(
            "Early-warning model artifacts are missing. Visit "
            "/early-warning-training/ to create them."
        )

    cache_key = "student-risk-dashboard:{}:{}".format(
        os.path.getmtime(dataset_path), os.path.getmtime(model_path)
    )
    results = cache.get(cache_key)
    table = pd.read_csv(dataset_path)
    if results is None:
        results = predict_batch(table, artifact_directory)
        cache.set(cache_key, results, DASHBOARD_CACHE_TIMEOUT)
    return table, results


def _filter_and_paginate_results(request, results):
    """Filter, sort, and paginate precomputed result records only."""
    search = request.GET.get("search", "").strip()
    selected_risk = request.GET.get("risk", "All")
    selected_module = request.GET.get("module", "All")
    selected_sort = request.GET.get("sort", "highest")
    modules = sorted({str(item["code_module"]) for item in results})

    # Work on copies so filtering and display-only fields never alter cached
    # inference results used by subsequent dashboard or download requests.
    filtered = [
        {
            **item,
            "dropout_percentage": item["dropout_probability"] * 100,
        }
        for item in results
    ]
    if search:
        filtered = [
            item for item in filtered if search.lower() in str(item["id_student"]).lower()
        ]
    if selected_risk in RISK_ORDER:
        filtered = [item for item in filtered if item["risk_level"] == selected_risk]
    else:
        selected_risk = "All"
    if selected_module in modules:
        filtered = [item for item in filtered if item["code_module"] == selected_module]
    else:
        selected_module = "All"

    if selected_sort == "lowest":
        filtered.sort(
            key=lambda item: (item["dropout_probability"], str(item["id_student"]))
        )
    elif selected_sort == "student":
        filtered.sort(key=lambda item: str(item["id_student"]))
    elif selected_sort == "module":
        filtered.sort(key=lambda item: (item["code_module"], str(item["id_student"])))
    else:
        selected_sort = "highest"
        filtered.sort(
            key=lambda item: (
                RISK_ORDER[item["risk_level"]],
                item["dropout_probability"],
            ),
            reverse=True,
        )

    paginator = Paginator(filtered, 25)
    page_obj = paginator.get_page(request.GET.get("page", 1))
    current_page = page_obj.number
    page_numbers = range(
        max(1, current_page - 2), min(paginator.num_pages, current_page + 2) + 1
    )
    query_string = urlencode(
        {
            "search": search,
            "risk": selected_risk,
            "module": selected_module,
            "sort": selected_sort,
        }
    )
    return {
        "results": filtered,
        "summary": summarise_risk_levels(filtered),
        "page_obj": page_obj,
        "page_numbers": page_numbers,
        "modules": modules,
        "search": search,
        "selected_risk": selected_risk,
        "selected_module": selected_module,
        "selected_sort": selected_sort,
        "query_string": query_string,
    }


def _add_dashboard_indicators(page_results, table, artifact_directory):
    """Attach existing model-native explanations to only the visible rows."""
    for result in page_results:
        learner_key = (
            result["id_student"],
            result["code_module"],
            result["code_presentation"],
        )
        try:
            explanation = explain_learner(table, learner_key, artifact_directory)
            result["key_risk_indicators"] = ", ".join(
                item["feature"] for item in explanation["higher"][:2]
            ) or "No higher-risk indicators"
        except ValueError:
            result["key_risk_indicators"] = "Indicators unavailable"
    return page_results


def _distribution(summary):
    total = summary["total_learners"]
    return [
        {
            "label": label,
            "count": summary[f"{label.lower()}_risk"],
            "percent": (summary[f"{label.lower()}_risk"] / total * 100) if total else 0,
        }
        for label in ("High", "Medium", "Low")
    ]


@role_required("ADMIN", "EDUCATOR")
def student_risk_dashboard(request):
    """Display cached, persisted-model predictions for all Day-60 learners."""
    context = {"intervention_overview": _intervention_overview()}
    try:
        table, results = _dashboard_prediction_results()
        context.update(_filter_and_paginate_results(request, results))
        context["all_summary"] = summarise_risk_levels(results)
        context["distribution"] = _distribution(context["summary"])
        context["page_obj"].object_list = _add_dashboard_indicators(
            list(context["page_obj"].object_list),
            table,
            get_early_warning_artifact_directory(),
        )
    except (ValueError, FileNotFoundError) as error:
        context["error"] = str(error)
    return render(request, "core/student_risk_dashboard.html", context)


@role_required("ADMIN", "EDUCATOR")
def batch_prediction(request):
    """Accept compatible Day-60 CSV data and cache its batch predictions."""
    context = {}
    if request.method == "POST":
        uploaded_file = request.FILES.get("dataset")
        if not uploaded_file or not uploaded_file.name.lower().endswith(".csv"):
            context["error"] = "Please select a CSV file containing Day-60 early-warning features."
        else:
            try:
                uploaded_table = pd.read_csv(uploaded_file)
                results = predict_batch(
                    uploaded_table, get_early_warning_artifact_directory()
                )
                cache_key = f"batch-prediction:{uuid4()}"
                cache.set(cache_key, results, DASHBOARD_CACHE_TIMEOUT)
                request.session["batch_prediction_cache_key"] = cache_key
                request.session.modified = True
                context["success"] = True
            except ValueError as error:
                context["error"] = str(error)
            except (FileNotFoundError, pd.errors.ParserError):
                context["error"] = (
                    "The CSV could not be used for early-warning prediction. "
                    "Check that it contains the required Day-60 columns and valid values."
                )

    cache_key = request.session.get("batch_prediction_cache_key")
    results = cache.get(cache_key) if cache_key else None
    if results:
        context.update(_filter_and_paginate_results(request, results))
        context["distribution"] = _distribution(context["summary"])
        context["has_results"] = True
    return render(request, "core/batch_prediction.html", context)


@role_required("ADMIN", "EDUCATOR")
def download_batch_predictions(request):
    """Download the current user's cached batch results without outcome data."""
    results = cache.get(request.session.get("batch_prediction_cache_key"))
    if not results:
        return redirect("batch_prediction")

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="early_warning_predictions.csv"'
    writer = csv.DictWriter(response, fieldnames=DOWNLOAD_COLUMNS)
    writer.writeheader()
    for result in results:
        writer.writerow({column: result[column] for column in DOWNLOAD_COLUMNS})
    return response


def _intervention_overview():
    """Return database-derived intervention counts without changing records."""
    interventions = Intervention.objects.all()
    today = timezone.localdate()
    overdue = interventions.filter(
        status__in=[Intervention.Status.PENDING, Intervention.Status.IN_PROGRESS],
        follow_up_date__lt=today,
    ).count()
    return {
        "total_interventions": interventions.count(),
        "pending": interventions.filter(status=Intervention.Status.PENDING).count(),
        "in_progress": interventions.filter(status=Intervention.Status.IN_PROGRESS).count(),
        "completed": interventions.filter(status=Intervention.Status.COMPLETED).count(),
        "overdue": overdue,
    }


def _learner_intervention_history(learner_key, current_probability):
    """Attach neutral current-vs-recorded probability information to history."""
    history = Intervention.objects.filter(
        student_id=str(learner_key[0]),
        code_module=learner_key[1],
        code_presentation=learner_key[2],
    )
    for intervention in history:
        intervention.current_risk_probability = current_probability
        intervention.risk_probability_at_intervention_percentage = (
            intervention.risk_probability_at_intervention * 100
        )
        intervention.current_risk_percentage = current_probability * 100
        intervention.risk_change_percentage_points = (
            current_probability - intervention.risk_probability_at_intervention
        ) * 100
    return history


def _current_learner_prediction(learner_key):
    """Use the existing persisted-model learner prediction for an action form."""
    dataset_path = get_early_warning_dataset_path()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("The Day-60 early-warning dataset is unavailable.")
    artifact_directory = get_early_warning_artifact_directory()
    if not all(
        os.path.exists(os.path.join(artifact_directory, filename))
        for filename in (MODEL_ARTIFACT_FILENAME, PREPROCESSING_ARTIFACT_FILENAME)
    ):
        persist_selected_model(dataset_path, artifact_directory)
    table = pd.read_csv(dataset_path)
    return predict_learner(table, learner_key, artifact_directory)


@role_required("ADMIN", "EDUCATOR")
def record_intervention(request):
    """Create a support action with learner identity and risk captured server-side."""
    source = request.POST if request.method == "POST" else request.GET
    learner_key = (
        source.get("id_student", ""),
        source.get("code_module", ""),
        source.get("code_presentation", ""),
    )
    context = {
        "learner_key": learner_key,
        "intervention_types": Intervention.InterventionType.choices,
        "priorities": Intervention.Priority.choices,
        "form_values": request.POST if request.method == "POST" else {},
    }
    try:
        if not all(learner_key):
            raise ValueError("Select a learner before recording an intervention.")
        prediction = _current_learner_prediction(learner_key)
        context["prediction"] = prediction
        context["risk_percentage"] = prediction["dropout_probability"] * 100
        if request.method == "POST":
            follow_up_value = request.POST.get("follow_up_date", "")
            follow_up_date = None
            if follow_up_value:
                follow_up_date = date.fromisoformat(follow_up_value)
            intervention = Intervention(
                student_id=str(learner_key[0]),
                code_module=learner_key[1],
                code_presentation=learner_key[2],
                intervention_type=request.POST.get("intervention_type", ""),
                priority=request.POST.get("priority", ""),
                notes=request.POST.get("notes", "").strip(),
                follow_up_date=follow_up_date,
                risk_probability_at_intervention=prediction["dropout_probability"],
            )
            intervention.full_clean()
            intervention.save()
            return redirect(
                reverse("individual_prediction")
                + "?id_student={}&code_module={}&code_presentation={}&intervention_saved=1".format(
                    learner_key[0], learner_key[1], learner_key[2]
                )
            )
    except (ValueError, ValidationError, FileNotFoundError):
        context["error"] = "We couldn't save the intervention. Please try again."
    return render(request, "core/record_intervention.html", context)


@role_required("ADMIN", "EDUCATOR")
def update_intervention(request, intervention_id):
    """Update only the mutable follow-up fields of an intervention record."""
    intervention = Intervention.objects.filter(pk=intervention_id).first()
    if not intervention:
        return redirect("interventions")
    if request.method == "POST":
        try:
            follow_up_value = request.POST.get("follow_up_date", "")
            intervention.status = request.POST.get("status", intervention.status)
            intervention.notes = request.POST.get("notes", "").strip()
            intervention.follow_up_date = (
                date.fromisoformat(follow_up_value) if follow_up_value else None
            )
            intervention.full_clean()
            intervention.save(update_fields=["status", "notes", "follow_up_date"])
            update_result = "intervention_updated=1"
        except (ValueError, ValidationError):
            update_result = "update_error=1"
    else:
        update_result = ""
    return redirect(
        reverse("individual_prediction")
        + "?id_student={}&code_module={}&code_presentation={}&{}#intervention-{}".format(
            intervention.student_id,
            intervention.code_module,
            intervention.code_presentation,
            update_result,
            intervention.pk,
        )
    )


@role_required("ADMIN", "EDUCATOR")
def interventions(request):
    """Provide a filterable, paginated intervention management register."""
    queryset = Intervention.objects.all()
    search = request.GET.get("search", "").strip()
    selected_status = request.GET.get("status", "All")
    selected_priority = request.GET.get("priority", "All")
    selected_type = request.GET.get("type", "All")
    sort = request.GET.get("sort", "follow_up")
    if search:
        queryset = queryset.filter(student_id__icontains=search)
    if selected_status in Intervention.Status.values:
        queryset = queryset.filter(status=selected_status)
    else:
        selected_status = "All"
    if selected_priority in Intervention.Priority.values:
        queryset = queryset.filter(priority=selected_priority)
    else:
        selected_priority = "All"
    if selected_type in Intervention.InterventionType.values:
        queryset = queryset.filter(intervention_type=selected_type)
    else:
        selected_type = "All"
    if sort != "follow_up":
        sort = "created"
    queryset = (
        queryset.order_by("follow_up_date", "-created_at")
        if sort == "follow_up"
        else queryset.order_by("-created_at")
    )
    page_obj = Paginator(queryset, 25).get_page(request.GET.get("page", 1))
    page_numbers = range(
        max(1, page_obj.number - 2),
        min(page_obj.paginator.num_pages, page_obj.number + 2) + 1,
    )
    for intervention in page_obj.object_list:
        intervention.risk_band = get_risk_band(
            intervention.risk_probability_at_intervention
        )
        intervention.risk_probability_percentage = (
            intervention.risk_probability_at_intervention * 100
        )
    query_string = urlencode({
        "search": search, "status": selected_status, "priority": selected_priority,
        "type": selected_type, "sort": sort,
    })
    return render(request, "core/interventions.html", {
        "summary": _intervention_overview(), "page_obj": page_obj,
        "page_numbers": page_numbers,
        "search": search, "selected_status": selected_status,
        "selected_priority": selected_priority, "selected_type": selected_type,
        "sort": sort, "query_string": query_string,
        "statuses": Intervention.Status.choices,
        "priorities": Intervention.Priority.choices,
        "intervention_types": Intervention.InterventionType.choices,
    })
