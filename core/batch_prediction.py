"""Validated batch inference for the persisted OULAD early-warning model."""

from pathlib import Path

import pandas as pd

from .early_warning_training import (
    _prepare_features,
    get_risk_band,
    load_selected_model,
)
from .oulad_features import REQUIRED_EARLY_FEATURES


IDENTIFIER_COLUMNS = ["id_student", "code_module", "code_presentation"]
REQUIRED_BATCH_COLUMNS = IDENTIFIER_COLUMNS + REQUIRED_EARLY_FEATURES
DOWNLOAD_COLUMNS = IDENTIFIER_COLUMNS + ["dropout_probability", "risk_level"]


def validate_batch_input(table):
    """Validate a Day-60 table before it reaches the persisted model."""
    if not isinstance(table, pd.DataFrame):
        raise ValueError("Upload a CSV file containing learner records.")

    missing = [column for column in REQUIRED_BATCH_COLUMNS if column not in table]
    if missing:
        raise ValueError(
            "The uploaded CSV is missing required columns: " + ", ".join(missing)
        )

    invalid_numeric = []
    for column in REQUIRED_EARLY_FEATURES:
        values = table[column].dropna()
        if not values.empty and pd.to_numeric(values, errors="coerce").isna().any():
            invalid_numeric.append(column)
    if invalid_numeric:
        raise ValueError(
            "The following early-warning columns must contain numeric values: "
            + ", ".join(invalid_numeric)
        )

    if table.empty:
        raise ValueError("The uploaded CSV does not contain any learner records.")
    return table


def predict_batch(table, artifact_directory):
    """Predict multiple valid learner records with existing persisted artifacts.

    ``dropout`` and ``final_result`` are deliberately ignored: neither is a
    predictive input or present in the returned result.
    """
    table = validate_batch_input(table).copy()
    model, preprocessing = load_selected_model(Path(artifact_directory))

    # The stored feature list is the authoritative model contract. It excludes
    # learner IDs and outcome columns, even if they are present in the upload.
    features, _ = _prepare_features(
        table,
        preprocessing["feature_columns"],
        preprocessing["encoded_feature_columns"],
    )
    features = features.fillna(preprocessing["medians"]).fillna(0.0)
    if not hasattr(model, "predict_proba"):
        raise ValueError("The persisted model cannot produce dropout probabilities.")

    probabilities = model.predict_proba(features)[:, 1]
    results = table[IDENTIFIER_COLUMNS].copy()
    results["dropout_probability"] = probabilities.astype(float)
    results["risk_level"] = [get_risk_band(value) for value in probabilities]
    return results[DOWNLOAD_COLUMNS].to_dict("records")


def summarise_risk_levels(results):
    """Return actual learner and risk counts for a batch result collection."""
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for result in results:
        risk_level = result.get("risk_level")
        if risk_level in counts:
            counts[risk_level] += 1
    return {
        "total_learners": len(results),
        "high_risk": counts["High"],
        "medium_risk": counts["Medium"],
        "low_risk": counts["Low"],
    }
