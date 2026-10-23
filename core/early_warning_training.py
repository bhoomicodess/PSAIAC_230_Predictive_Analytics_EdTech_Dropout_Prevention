"""Separate training and persistence workflow for early-warning predictions."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .oulad_features import (
    ASSESSMENT_CUTOFF_DAY,
    LEARNER_KEYS,
    REQUIRED_EARLY_FEATURES,
    validate_early_warning_table,
)


TARGET_COLUMN = "dropout"
PREDICTIVE_IDENTIFIER_COLUMNS = ["id_student"]
MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
]
SELECTED_MODEL_NAME = "Logistic Regression"
LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70
MODEL_ARTIFACT_FILENAME = "early_warning_model.joblib"
PREPROCESSING_ARTIFACT_FILENAME = "early_warning_preprocessing.joblib"


def _feature_columns(table):
    feature_columns = [
        column
        for column in table.columns
        if column not in PREDICTIVE_IDENTIFIER_COLUMNS
        and column != TARGET_COLUMN
        and column != "final_result"
    ]
    if not set(REQUIRED_EARLY_FEATURES).issubset(feature_columns):
        raise ValueError("Required early-warning features are not model inputs")
    return feature_columns


def _prepare_features(table, feature_columns, encoded_feature_columns=None):
    features = pd.get_dummies(table[feature_columns], dtype=float)
    if encoded_feature_columns is not None:
        features = features.reindex(columns=encoded_feature_columns, fill_value=0.0)
    medians = features.median(numeric_only=True)
    features = features.fillna(medians).fillna(0.0)
    return features, medians


def get_risk_band(probability):
    """Return the project-defined initial risk band for a dropout probability."""
    if probability < LOW_THRESHOLD:
        return "Low"
    if probability < HIGH_THRESHOLD:
        return "Medium"
    return "High"


def _read_and_prepare_dataset(dataset_path):
    table = pd.read_csv(Path(dataset_path))
    validate_early_warning_table(table)
    feature_columns = _feature_columns(table)
    X, medians = _prepare_features(table, feature_columns)
    return table, feature_columns, X, medians


def _model():
    return LogisticRegression(max_iter=1000)


def _readable_feature_name(feature_name):
    labels = {
        "registration_offset_days": "Registration timing",
        "early_assessment_count": "Early assessment count",
        "early_assessment_score_mean": "Early assessment mean score",
        "early_assessment_score_std": "Early assessment score variation",
        "early_assessment_scheduled_count": "Scheduled assessment count",
        "early_assessment_submission_rate": "Early assessment submission rate",
    }
    if feature_name in labels:
        return labels[feature_name]
    if feature_name.startswith("code_module_"):
        return f"Module: {feature_name.removeprefix('code_module_')}"
    if feature_name.startswith("code_presentation_"):
        return f"Presentation: {feature_name.removeprefix('code_presentation_')}"
    return feature_name


def explain_learner(table, learner_key, artifact_directory):
    """Return model-native feature contributions for one learner."""
    model, preprocessing = load_selected_model(artifact_directory)
    if not hasattr(model, "coef_"):
        raise ValueError("Feature contributions require Logistic Regression coefficients")
    matches = table[
        (table["id_student"].astype(str) == str(learner_key[0]))
        & (table["code_module"] == learner_key[1])
        & (table["code_presentation"] == learner_key[2])
    ]
    if len(matches) != 1:
        raise ValueError("The selected learner/course/presentation was not found")
    row = matches.iloc[[0]]
    transformed, _ = _prepare_features(
        row,
        preprocessing["feature_columns"],
        preprocessing["encoded_feature_columns"],
    )
    transformed = transformed.fillna(preprocessing["medians"]).fillna(0.0)
    values = transformed.iloc[0]
    coefficients = model.coef_[0]
    contributions = []
    for feature_name, value, coefficient in zip(
        transformed.columns,
        values,
        coefficients,
    ):
        contribution = float(value * coefficient)
        if contribution == 0:
            continue
        contributions.append(
            {
                "feature": _readable_feature_name(feature_name),
                "value": value,
                "contribution": contribution,
            }
        )
    higher = sorted(
        [item for item in contributions if item["contribution"] > 0],
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )[:5]
    lower = sorted(
        [item for item in contributions if item["contribution"] < 0],
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )[:5]
    return {"higher": higher, "lower": lower}


def analyze_early_warning_thresholds(dataset_path):
    """Evaluate project-defined thresholds on the held-out test split."""
    table, feature_columns, X, _ = _read_and_prepare_dataset(dataset_path)
    y = table[TARGET_COLUMN].astype("int8")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    model = _model()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    threshold_rows = []
    for threshold in (LOW_THRESHOLD, 0.50, HIGH_THRESHOLD):
        predictions = (probabilities >= threshold).astype("int8")
        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()
        threshold_rows.append(
            {
                "threshold": threshold,
                "predicted_positive_count": int(predictions.sum()),
                "precision": round(
                    precision_score(y_test, predictions, zero_division=0) * 100,
                    2,
                ),
                "recall": round(
                    recall_score(y_test, predictions, zero_division=0) * 100,
                    2,
                ),
                "f1": round(
                    f1_score(y_test, predictions, zero_division=0) * 100,
                    2,
                ),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            }
        )
    predictions = (probabilities >= 0.50).astype("int8")
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "test_size": len(X_test),
        "precision": round(precision_score(y_test, predictions, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, predictions, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, predictions, zero_division=0) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, probabilities) * 100, 2),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "threshold_analysis": threshold_rows,
        "feature_columns": feature_columns,
    }


def persist_selected_model(dataset_path, artifact_directory):
    """Train the configured model once and save model and preprocessing artifacts."""
    dataset_path = Path(dataset_path)
    artifact_directory = Path(artifact_directory)
    table = pd.read_csv(dataset_path)
    validate_early_warning_table(table)
    feature_columns = _feature_columns(table)
    X, medians = _prepare_features(table, feature_columns)
    model_classes = {
        "Logistic Regression": _model(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }
    if SELECTED_MODEL_NAME not in model_classes:
        raise ValueError(f"Unsupported selected model: {SELECTED_MODEL_NAME}")
    model = model_classes[SELECTED_MODEL_NAME]
    model.fit(X, table[TARGET_COLUMN].astype("int8"))
    artifact_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_directory / MODEL_ARTIFACT_FILENAME)
    joblib.dump(
        {
            "feature_columns": feature_columns,
            "encoded_feature_columns": list(X.columns),
            "medians": medians,
            "target_column": TARGET_COLUMN,
            "cutoff_day": ASSESSMENT_CUTOFF_DAY,
            "selected_model": SELECTED_MODEL_NAME,
        },
        artifact_directory / PREPROCESSING_ARTIFACT_FILENAME,
    )
    return {
        "model_path": artifact_directory / MODEL_ARTIFACT_FILENAME,
        "preprocessing_path": artifact_directory / PREPROCESSING_ARTIFACT_FILENAME,
        "selected_model": SELECTED_MODEL_NAME,
    }


def load_selected_model(artifact_directory):
    """Load the persisted model and its preprocessing metadata."""
    artifact_directory = Path(artifact_directory)
    model_path = artifact_directory / MODEL_ARTIFACT_FILENAME
    preprocessing_path = artifact_directory / PREPROCESSING_ARTIFACT_FILENAME
    if not model_path.exists() or not preprocessing_path.exists():
        raise FileNotFoundError(
            "Early-warning model artifacts are missing. Visit "
            "/early-warning-training/ to create them."
        )
    return joblib.load(model_path), joblib.load(preprocessing_path)


def predict_learner(table, learner_key, artifact_directory):
    """Predict one existing learner-course row using persisted artifacts."""
    model, preprocessing = load_selected_model(artifact_directory)
    matches = table[
        (table["id_student"].astype(str) == str(learner_key[0]))
        & (table["code_module"] == learner_key[1])
        & (table["code_presentation"] == learner_key[2])
    ]
    if len(matches) != 1:
        raise ValueError("The selected learner/course/presentation was not found")
    row = matches.iloc[[0]]
    features, _ = _prepare_features(
        row,
        preprocessing["feature_columns"],
        preprocessing["encoded_feature_columns"],
    )
    features = features.fillna(preprocessing["medians"]).fillna(0.0)
    prediction = int(model.predict(features)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0, 1])
    return {
        "row": row.iloc[0].to_dict(),
        "predicted_class": prediction,
        "dropout_probability": probability,
        "risk_band": get_risk_band(probability) if probability is not None else "Unavailable",
        "selected_model": preprocessing["selected_model"],
    }


def train_early_warning_models(dataset_path):
    """Train and compare the three existing model families on early features."""
    dataset_path = Path(dataset_path)
    table = pd.read_csv(dataset_path)
    validate_early_warning_table(table)

    y = table[TARGET_COLUMN].astype("int8")
    feature_columns = _feature_columns(table)

    print("Early-warning features used:", feature_columns)
    print("Total samples:", len(table))
    print("Dropout count:", int((y == 1).sum()))
    print("Non-dropout count:", int((y == 0).sum()))
    print("Dropout percentage:", round(float(y.mean() * 100), 2))

    X, _ = _prepare_features(table, feature_columns)

    class_counts = y.value_counts()
    stratify = y if len(class_counts) == 2 and class_counts.min() >= 2 else None
    if stratify is None:
        raise ValueError("Stratified splitting requires at least two samples per class")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify,
    )
    print("Feature columns after encoding:", list(X.columns))
    print("Train/test sizes:", len(X_train), len(X_test))

    models = {
        "Logistic Regression": _model(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "name": model_name,
                "accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
                "precision": round(
                    precision_score(
                        y_test, predictions, average="weighted", zero_division=0
                    )
                    * 100,
                    2,
                ),
                "recall": round(
                    recall_score(
                        y_test, predictions, average="weighted", zero_division=0
                    )
                    * 100,
                    2,
                ),
                "f1": round(
                    f1_score(
                        y_test, predictions, average="weighted", zero_division=0
                    )
                    * 100,
                    2,
                ),
                "roc_auc": round(roc_auc_score(y_test, probabilities) * 100, 2),
            }
        )

    return {
        "results": results,
        "feature_columns": feature_columns,
        "encoded_feature_columns": list(X.columns),
        "total_samples": len(table),
        "dropout_count": int((y == 1).sum()),
        "non_dropout_count": int((y == 0).sum()),
        "dropout_percentage": round(float(y.mean() * 100), 2),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
    }
