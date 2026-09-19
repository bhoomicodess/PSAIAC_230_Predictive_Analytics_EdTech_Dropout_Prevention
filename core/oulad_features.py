"""Leakage-aware preparation helpers for the repository's OULAD data.

This module is intentionally separate from the existing upload and training
views. The early-warning workflow calls it with an explicit assessment cutoff.
"""

from pathlib import Path

import pandas as pd


DROPOUT_MAPPING = {
    "Withdrawn": 1,
    "Pass": 0,
    "Fail": 0,
    "Distinction": 0,
}

LEARNER_KEYS = [
    "id_student",
    "code_module",
    "code_presentation",
]

ASSESSMENT_CUTOFF_DAY = 60
EARLY_WARNING_DATASET_FILENAME = "early_warning_dataset.csv"
REQUIRED_EARLY_FEATURES = [
    "registration_offset_days",
    "early_assessment_count",
    "early_assessment_score_mean",
    "early_assessment_score_std",
    "early_assessment_scheduled_count",
    "early_assessment_submission_rate",
]


def derive_dropout_target(
    student_info: pd.DataFrame,
    outcome_column: str = "final_result",
) -> pd.Series:
    """Map the observed OULAD outcomes to a binary dropout label."""
    if outcome_column not in student_info.columns:
        raise ValueError(f"Missing required outcome column: {outcome_column}")

    observed_values = set(student_info[outcome_column].dropna().unique())
    unknown_values = observed_values.difference(DROPOUT_MAPPING)
    if unknown_values:
        raise ValueError(
            "Unsupported final_result values: "
            + ", ".join(sorted(map(str, unknown_values)))
        )

    target = student_info[outcome_column].map(DROPOUT_MAPPING)
    if target.isna().any():
        raise ValueError("final_result contains missing values")

    return target.astype("int8").rename("dropout")


def build_early_warning_table(
    data_directory,
    assessment_cutoff_day: int = ASSESSMENT_CUTOFF_DAY,
) -> pd.DataFrame:
    """Build a learner-level table using data observed by a fixed cutoff.

    Assessment dates are relative to the course start in OULAD. Records after
    the cutoff are excluded so eventual performance cannot enter the features.
    The returned table contains ``dropout`` but never contains ``final_result``.
    """
    if not isinstance(assessment_cutoff_day, int):
        raise TypeError("assessment_cutoff_day must be an integer")

    data_directory = Path(data_directory)
    student_info = pd.read_csv(data_directory / "studentInfo.csv")
    registration = pd.read_csv(data_directory / "studentRegistration.csv")
    student_assessment = pd.read_csv(data_directory / "studentAssessment.csv")
    assessments = pd.read_csv(data_directory / "assessments.csv")

    missing_keys = [
        column
        for column in LEARNER_KEYS
        if column not in student_info.columns
        or column not in registration.columns
    ]
    if missing_keys:
        raise ValueError(
            "Missing learner key columns: " + ", ".join(missing_keys)
        )

    base = student_info[LEARNER_KEYS].copy()
    base = base.merge(
        registration[LEARNER_KEYS + ["date_registration"]],
        on=LEARNER_KEYS,
        how="left",
        validate="one_to_one",
    )
    base["registration_offset_days"] = pd.to_numeric(
        base.pop("date_registration"),
        errors="coerce",
    )
    base["dropout"] = derive_dropout_target(student_info).to_numpy()

    assessment_dates = pd.to_numeric(assessments["date"], errors="coerce")
    assessment_meta = assessments.loc[
        assessment_dates.notna() & (assessment_dates <= assessment_cutoff_day),
        ["id_assessment", "code_module", "code_presentation", "weight"],
    ].copy()

    submitted = student_assessment.merge(
        assessment_meta,
        on="id_assessment",
        how="inner",
        validate="many_to_one",
    )
    submitted["score"] = pd.to_numeric(submitted["score"], errors="coerce")
    submitted = submitted.dropna(subset=["score"])

    assessment_features = submitted.groupby(
        ["id_student", "code_module", "code_presentation"],
        as_index=False,
    ).agg(
        early_assessment_count=("id_assessment", "nunique"),
        early_assessment_score_mean=("score", "mean"),
        early_assessment_score_std=("score", "std"),
    )

    scheduled_counts = assessment_meta.groupby(
        ["code_module", "code_presentation"],
    )["id_assessment"].nunique()
    assessment_features["early_assessment_scheduled_count"] = (
        assessment_features.set_index(
            ["code_module", "code_presentation"]
        ).index.map(scheduled_counts).to_numpy()
    )
    assessment_features["early_assessment_submission_rate"] = (
        assessment_features["early_assessment_count"]
        / assessment_features["early_assessment_scheduled_count"]
    )

    result = base.merge(
        assessment_features,
        on=LEARNER_KEYS,
        how="left",
        validate="one_to_one",
    )
    count_columns = [
        "early_assessment_count",
        "early_assessment_scheduled_count",
    ]
    result[count_columns] = result[count_columns].fillna(0).astype("int64")
    result["early_assessment_submission_rate"] = (
        result["early_assessment_submission_rate"].fillna(0.0)
    )
    result["early_assessment_score_std"] = (
        result["early_assessment_score_std"].fillna(0.0)
    )

    return result


def validate_early_warning_table(
    table: pd.DataFrame,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Validate the generated table before it can be used for training."""
    required_columns = LEARNER_KEYS + REQUIRED_EARLY_FEATURES + ["dropout"]
    missing_columns = [
        column for column in required_columns if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(
            "Early-warning dataset is missing columns: "
            + ", ".join(missing_columns)
        )

    if "final_result" in table.columns:
        raise ValueError("final_result must not be present in early-warning features")

    if expected_rows is not None and len(table) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, found {len(table)}"
        )

    if table.duplicated(LEARNER_KEYS).any():
        raise ValueError("Duplicate learner/module/presentation records found")

    target_values = set(table["dropout"].dropna().unique())
    if target_values.difference({0, 1}) or table["dropout"].isna().any():
        raise ValueError("dropout must contain only binary values 0 and 1")

    non_numeric = [
        column
        for column in REQUIRED_EARLY_FEATURES
        if not pd.api.types.is_numeric_dtype(table[column])
    ]
    if non_numeric:
        raise ValueError(
            "Expected numeric early-warning features: "
            + ", ".join(non_numeric)
        )

    return table


def generate_early_warning_dataset(
    data_directory,
    output_path,
    assessment_cutoff_day: int = ASSESSMENT_CUTOFF_DAY,
) -> pd.DataFrame:
    """Build, validate, and persist the separate early-warning dataset."""
    data_directory = Path(data_directory)
    student_info = pd.read_csv(data_directory / "studentInfo.csv")
    table = build_early_warning_table(data_directory, assessment_cutoff_day)
    validate_early_warning_table(table, expected_rows=len(student_info))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table
