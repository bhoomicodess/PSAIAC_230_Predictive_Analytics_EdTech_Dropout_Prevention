
"""Data quality and temporal leakage audit for the OULAD dropout project.

This module audits source data without changing the existing ML pipeline.
"""

from pathlib import Path

import pandas as pd
import json


ASSESSMENT_CUTOFF_DAY = 60

LEARNER_KEYS = [
    "id_student",
    "code_module",
    "code_presentation",
]

REQUIRED_FILES = [
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "assessments.csv",
]


def validate_source_files(data_directory):
    """Check that all required OULAD source files exist."""
    data_directory = Path(data_directory)

    missing_files = [
        filename
        for filename in REQUIRED_FILES
        if not (data_directory / filename).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required source files: " + ", ".join(missing_files)
        )

    return {
        "status": "passed",
        "files_checked": len(REQUIRED_FILES),
        "message": "All required source files are available.",
    }


def check_missing_values(data_directory):
    """Report missing values in each required source file."""
    data_directory = Path(data_directory)
    report = {}

    for filename in REQUIRED_FILES:
        file_path = data_directory / filename
        dataframe = pd.read_csv(file_path)

        missing_by_column = dataframe.isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]

        report[filename] = {
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "missing_cells": int(dataframe.isna().sum().sum()),
            "missing_by_column": {
                column: int(count)
                for column, count in missing_by_column.items()
            },
        }

    return report

def check_duplicate_records(data_directory):
    """Identify duplicate records using each file's relevant keys."""
    data_directory = Path(data_directory)

    key_columns = {
        "studentInfo.csv": [
            "id_student", "code_module", "code_presentation"
        ],
        "studentRegistration.csv": [
            "id_student", "code_module", "code_presentation"
        ],
        "studentAssessment.csv": [
            "id_assessment", "id_student"
        ],
        "assessments.csv": ["id_assessment"],
    }

    report = {}

    for filename, keys in key_columns.items():
        dataframe = pd.read_csv(data_directory / filename)

        missing_keys = [
            column for column in keys
            if column not in dataframe.columns
        ]
        if missing_keys:
            report[filename] = {
                "status": "error",
                "message": "Missing key columns: " + ", ".join(missing_keys),
            }
            continue

        duplicate_count = int(dataframe.duplicated(subset=keys).sum())

        report[filename] = {
            "rows": int(len(dataframe)),
            "duplicate_records": duplicate_count,
            "status": "passed" if duplicate_count == 0 else "warning",
        }

    return report


def check_dropout_class_balance(data_directory):
    """Summarize dropout and non-dropout counts in the audit dataset."""
    dataset_path = Path(data_directory) / "early_warning_dataset.csv"

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Audit dataset not found: {dataset_path}"
        )

    dataframe = pd.read_csv(dataset_path)

    if "dropout" not in dataframe.columns:
        raise ValueError("The dataset must contain a 'dropout' column.")

    labels = dataframe["dropout"]

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("The dropout column must contain only 0 and 1.")

    counts = labels.value_counts().to_dict()
    total = len(labels)

    return {
        "total_students": int(total),
        "dropout": int(counts.get(1, 0)),
        "non_dropout": int(counts.get(0, 0)),
        "dropout_percentage": (
            round(float(counts.get(1, 0)) / total * 100, 2)
            if total else 0.0
        ),
        "non_dropout_percentage": (
            round(float(counts.get(0, 0)) / total * 100, 2)
            if total else 0.0
        ),
    }

def check_day60_submission_leakage(data_directory):
    """Find submissions after Day 60 for assessments scheduled by Day 60.

    These records are potential temporal leakage, not automatically confirmed
    leakage. The audit reports them without changing the source data.
    """
    data_directory = Path(data_directory)

    assessments = pd.read_csv(data_directory / "assessments.csv")
    submissions = pd.read_csv(data_directory / "studentAssessment.csv")

    required_assessment_columns = {
        "id_assessment", "code_module", "code_presentation", "date"
    }
    required_submission_columns = {
        "id_assessment", "id_student", "date_submitted"
    }

    missing_assessment_columns = (
        required_assessment_columns - set(assessments.columns)
    )
    missing_submission_columns = (
        required_submission_columns - set(submissions.columns)
    )

    if missing_assessment_columns or missing_submission_columns:
        missing = sorted(
            missing_assessment_columns | missing_submission_columns
        )
        raise ValueError(
            "Missing columns required for leakage audit: "
            + ", ".join(missing)
        )

    assessments["date"] = pd.to_numeric(
        assessments["date"], errors="coerce"
    )
    submissions["date_submitted"] = pd.to_numeric(
        submissions["date_submitted"], errors="coerce"
    )

    early_assessments = assessments.loc[
        assessments["date"].notna()
        & (assessments["date"] <= ASSESSMENT_CUTOFF_DAY),
        ["id_assessment", "code_module", "code_presentation", "date"],
    ].drop_duplicates(subset=["id_assessment"])

    merged = submissions.merge(
        early_assessments,
        on="id_assessment",
        how="inner",
        validate="many_to_one",
    )

    late_mask = (
        merged["date_submitted"].notna()
        & (merged["date_submitted"] > ASSESSMENT_CUTOFF_DAY)
    )
    late_submissions = merged.loc[late_mask].copy()

    return {
        "cutoff_day": ASSESSMENT_CUTOFF_DAY,
        "early_assessments_checked": int(len(early_assessments)),
        "matching_submissions_checked": int(len(merged)),
        "potential_late_submissions": int(len(late_submissions)),
        "unique_students_affected": int(
            late_submissions["id_student"].nunique()
        ),
        "status": (
            "warning" if not late_submissions.empty else "passed"
        ),
        "details": late_submissions[
            [
                "id_student",
                "id_assessment",
                "code_module",
                "code_presentation",
                "date",
                "date_submitted",
            ]
        ].to_dict(orient="records"),
        "note": (
            "These are potential cutoff violations for assessments scheduled "
            "by Day 60. Review the records and dataset construction before "
            "concluding that model leakage occurred."
        ),
    }




def run_data_quality_audit(data_directory, output_path=None):
    """Run all data-quality checks and optionally save a JSON report."""
    data_directory = Path(data_directory)

    file_check = validate_source_files(data_directory)

    report = {
        "audit_name": "EdTech Dropout Prevention Data Quality Audit",
        "cutoff_day": ASSESSMENT_CUTOFF_DAY,
        "source_files": file_check,
        "missing_values": check_missing_values(data_directory),
        "duplicate_records": check_duplicate_records(data_directory),
        "dropout_class_balance": check_dropout_class_balance(data_directory),
        "day60_submission_leakage": check_day60_submission_leakage(
            data_directory
        ),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, default=str)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the EdTech dropout data quality audit."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the OULAD CSV files.",
    )
    parser.add_argument(
        "--output",
        default="data/early_warning_audit_report.json",
        help="Path where the JSON audit report will be saved.",
    )
    args = parser.parse_args()

    result = run_data_quality_audit(args.data_dir, args.output)

    print("\nDATA QUALITY AUDIT")
    print("==================")
    print("Source files:", result["source_files"]["status"])
    print("Dataset rows:", result["dropout_class_balance"]["total_students"])
    print(
        "Potential late submissions:",
        result["day60_submission_leakage"]["potential_late_submissions"],
    )
    print(
        "Unique students affected:",
        result["day60_submission_leakage"]["unique_students_affected"],
    )
    print("Report saved to:", args.output)
