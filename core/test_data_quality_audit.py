
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from core.data_quality_audit import (
    check_day60_submission_leakage,
    check_dropout_class_balance,
    check_duplicate_records,
    check_missing_values,
    validate_source_files,
)


class DataQualityAuditTests(unittest.TestCase):

    def create_source_files(self, directory):
        root = Path(directory)

        pd.DataFrame({
            "id_student": [1, 2],
            "code_module": ["AAA", "AAA"],
            "code_presentation": ["2013J", "2013J"],
            "final_result": ["Withdrawn", "Pass"],
        }).to_csv(root / "studentInfo.csv", index=False)

        pd.DataFrame({
            "id_student": [1, 2],
            "code_module": ["AAA", "AAA"],
            "code_presentation": ["2013J", "2013J"],
            "date_registration": [-10, -5],
        }).to_csv(root / "studentRegistration.csv", index=False)

        pd.DataFrame({
            "id_assessment": [10, 10, 20],
            "id_student": [1, 2, 1],
            "date_submitted": [30, 75, 80],
            "is_banked": [0, 0, 0],
            "score": [80, 70, 50],
        }).to_csv(root / "studentAssessment.csv", index=False)

        pd.DataFrame({
            "id_assessment": [10, 20],
            "code_module": ["AAA", "AAA"],
            "code_presentation": ["2013J", "2013J"],
            "date": [30, 90],
            "weight": [50, 50],
        }).to_csv(root / "assessments.csv", index=False)

        pd.DataFrame({
            "id_student": [1, 2],
            "code_module": ["AAA", "AAA"],
            "code_presentation": ["2013J", "2013J"],
            "dropout": [1, 0],
        }).to_csv(root / "early_warning_dataset.csv", index=False)

    def test_required_source_files_exist(self):
        with TemporaryDirectory() as directory:
            self.create_source_files(directory)
            result = validate_source_files(directory)
            self.assertEqual(result["status"], "passed")

    def test_day60_audit_detects_late_submission(self):
        with TemporaryDirectory() as directory:
            self.create_source_files(directory)
            result = check_day60_submission_leakage(directory)

            self.assertEqual(result["potential_late_submissions"], 1)
            self.assertEqual(result["unique_students_affected"], 1)

    def test_class_balance_counts_both_classes(self):
        with TemporaryDirectory() as directory:
            self.create_source_files(directory)
            result = check_dropout_class_balance(directory)

            self.assertEqual(result["total_students"], 2)
            self.assertEqual(result["dropout"], 1)
            self.assertEqual(result["non_dropout"], 1)

    def test_duplicate_audit_reports_assessment_keys(self):
        with TemporaryDirectory() as directory:
            self.create_source_files(directory)
            result = check_duplicate_records(directory)

            self.assertEqual(
                result["studentAssessment.csv"]["duplicate_records"], 0
            )

    def test_missing_value_audit_reports_source_rows(self):
        with TemporaryDirectory() as directory:
            self.create_source_files(directory)
            result = check_missing_values(directory)

            self.assertEqual(
                result["studentInfo.csv"]["missing_cells"], 0
            )


if __name__ == "__main__":
    unittest.main()
