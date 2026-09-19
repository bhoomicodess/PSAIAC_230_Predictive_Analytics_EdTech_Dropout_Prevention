from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from django.test import TestCase, override_settings
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from .early_warning_training import (
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    PREDICTIVE_IDENTIFIER_COLUMNS,
    analyze_early_warning_thresholds,
    explain_learner,
    get_risk_band,
    persist_selected_model,
    predict_learner,
)
from .oulad_features import (
    LEARNER_KEYS,
    build_early_warning_table,
    derive_dropout_target,
    generate_early_warning_dataset,
    validate_early_warning_table,
)


class EarlyWarningFeatureTests(TestCase):

    def test_dropout_mapping(self):
        outcomes = pd.DataFrame(
            {"final_result": ["Withdrawn", "Pass", "Fail", "Distinction"]}
        )
        self.assertEqual(
            derive_dropout_target(outcomes).tolist(),
            [1, 0, 0, 0],
        )

    def test_cutoff_excludes_future_assessments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "code_module": ["AAA"],
                    "code_presentation": ["2013J"],
                    "id_student": [1],
                    "final_result": ["Pass"],
                }
            ).to_csv(root / "studentInfo.csv", index=False)
            pd.DataFrame(
                {
                    "code_module": ["AAA"],
                    "code_presentation": ["2013J"],
                    "id_student": [1],
                    "date_registration": [-10],
                    "date_unregistration": ["?"],
                }
            ).to_csv(root / "studentRegistration.csv", index=False)
            pd.DataFrame(
                {
                    "id_assessment": [10, 20],
                    "id_student": [1, 1],
                    "date_submitted": [15, 80],
                    "is_banked": [0, 0],
                    "score": [80, 20],
                }
            ).to_csv(root / "studentAssessment.csv", index=False)
            pd.DataFrame(
                {
                    "code_module": ["AAA", "AAA"],
                    "code_presentation": ["2013J", "2013J"],
                    "id_assessment": [10, 20],
                    "assessment_type": ["TMA", "TMA"],
                    "date": [30, 90],
                    "weight": [10.0, 10.0],
                }
            ).to_csv(root / "assessments.csv", index=False)

            table = build_early_warning_table(root, assessment_cutoff_day=60)

        self.assertEqual(int(table.loc[0, "early_assessment_count"]), 1)
        self.assertEqual(float(table.loc[0, "early_assessment_score_mean"]), 80.0)

    def test_generated_dataset_structure_and_feature_safety(self):
        data_directory = Path(__file__).resolve().parents[1] / "data"
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "early_warning_dataset.csv"
            table = generate_early_warning_dataset(data_directory, output_path)
            validate_early_warning_table(table, expected_rows=32593)
            self.assertTrue(output_path.exists())

        self.assertNotIn("final_result", table.columns)
        self.assertIn("dropout", table.columns)
        self.assertEqual(set(table["dropout"].unique()), {0, 1})
        self.assertEqual(table.duplicated(LEARNER_KEYS).sum(), 0)
        self.assertTrue(
            all(
                pd.api.types.is_numeric_dtype(table[column])
                for column in table.columns
                if column.startswith("early_")
            )
        )
        feature_columns = [
            column
            for column in table.columns
            if column not in PREDICTIVE_IDENTIFIER_COLUMNS
            and column != "dropout"
        ]
        self.assertNotIn("id_student", feature_columns)
        self.assertNotIn("final_result", feature_columns)


class IndividualPredictionTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.data_directory = Path(__file__).resolve().parents[1] / "data"
        cls.dataset_path = cls.data_directory / "early_warning_dataset.csv"
        cls.artifact_directory = cls.data_directory / "test_early_warning_artifacts"
        cls.table = pd.read_csv(cls.dataset_path)
        persist_selected_model(cls.dataset_path, cls.artifact_directory)

    @classmethod
    def tearDownClass(cls):
        for path in cls.artifact_directory.glob("*"):
            path.unlink()
        cls.artifact_directory.rmdir()
        super().tearDownClass()

    def test_prediction_route_returns_http_200(self):
        with override_settings(ALLOWED_HOSTS=["localhost"]):
            response = self.client.get(
                "/individual-prediction/",
                HTTP_HOST="localhost",
            )
        self.assertEqual(response.status_code, 200)

    def test_valid_learner_prediction_contains_probability_and_risk_band(self):
        row = self.table.iloc[0]
        prediction = predict_learner(
            self.table,
            (row["id_student"], row["code_module"], row["code_presentation"]),
            self.artifact_directory,
        )
        self.assertIn(prediction["predicted_class"], [0, 1])
        self.assertGreaterEqual(prediction["dropout_probability"], 0)
        self.assertLessEqual(prediction["dropout_probability"], 1)
        self.assertIn(prediction["risk_band"], ["Low", "Medium", "High"])

    def test_invalid_learner_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_learner(
                self.table,
                ("does-not-exist", "AAA", "2013J"),
                self.artifact_directory,
            )

    def test_risk_thresholds_are_consistent(self):
        self.assertEqual(get_risk_band(LOW_THRESHOLD - 0.01), "Low")
        self.assertEqual(get_risk_band(LOW_THRESHOLD), "Medium")
        self.assertEqual(get_risk_band(HIGH_THRESHOLD), "High")

    def test_explanation_contains_only_model_features(self):
        row = self.table.iloc[0]
        explanation = explain_learner(
            self.table,
            (row["id_student"], row["code_module"], row["code_presentation"]),
            self.artifact_directory,
        )
        for category in ("higher", "lower"):
            self.assertLessEqual(len(explanation[category]), 5)
            for item in explanation[category]:
                self.assertNotIn(
                    item["feature"],
                    {"final_result", "dropout", "date_unregistration"},
                )

    def test_threshold_analysis_and_confusion_counts_are_generated(self):
        diagnostics = analyze_early_warning_thresholds(self.dataset_path)
        self.assertEqual(diagnostics["test_size"], 6519)
        self.assertEqual(
            [item["threshold"] for item in diagnostics["threshold_analysis"]],
            [LOW_THRESHOLD, 0.50, HIGH_THRESHOLD],
        )
        for item in diagnostics["threshold_analysis"]:
            self.assertGreaterEqual(item["precision"], 0)
            self.assertLessEqual(item["precision"], 100)
            self.assertGreaterEqual(item["recall"], 0)
            self.assertLessEqual(item["recall"], 100)
            self.assertGreaterEqual(item["f1"], 0)
            self.assertLessEqual(item["f1"], 100)
        self.assertGreaterEqual(diagnostics["true_positives"], 0)
        self.assertGreaterEqual(diagnostics["false_positives"], 0)
        self.assertGreaterEqual(diagnostics["true_negatives"], 0)
        self.assertGreaterEqual(diagnostics["false_negatives"], 0)


@override_settings(ALLOWED_HOSTS=["localhost"])
class NavigationTests(TestCase):

    navigation_routes = {
        "upload_dataset": "/",
        "data_preprocessing": "/preprocessing/",
        "model_training": "/model-training/",
        "early_warning_training": "/early-warning-training/",
        "individual_prediction": "/individual-prediction/",
    }

    def test_major_pages_render_all_navigation_links(self):
        pages = [
            reverse("upload_dataset"),
            reverse("data_preprocessing"),
            reverse("model_training"),
            reverse("early_warning_training"),
            reverse("individual_prediction"),
        ]
        for page in pages:
            with self.subTest(page=page):
                response = self.client.get(page, HTTP_HOST="localhost")
                self.assertEqual(response.status_code, 200)
                for route_name, route in self.navigation_routes.items():
                    self.assertContains(
                        response,
                        f'href="{route}"',
                        msg_prefix=f"{route_name} link missing from {page}",
                    )

    def test_navigation_target_routes_return_http_200(self):
        for route_name in self.navigation_routes:
            with self.subTest(route_name=route_name):
                response = self.client.get(
                    reverse(route_name),
                    HTTP_HOST="localhost",
                )
                self.assertEqual(response.status_code, 200)

    def test_shared_sidebar_is_single_and_has_no_upload_item(self):
        response = self.client.get(
            reverse("model_training"),
            HTTP_HOST="localhost",
        )
        self.assertEqual(
            response.content.decode().count('aria-label="Primary navigation"'),
            1,
        )
        sidebar_html = response.content.decode().split(
            '<aside class="sidebar app-sidebar"',
            1,
        )[1].split("</aside>", 1)[0]
        self.assertNotIn("Upload Dataset", sidebar_html)
        self.assertIn("Train Model", sidebar_html)


class GenericDatasetStateTests(TestCase):

    def test_preprocessing_requires_a_current_upload(self):
        response = self.client.get(
            reverse("data_preprocessing"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No Dataset Uploaded")
        self.assertNotContains(response, "Original Records")
        self.assertNotContains(response, "final_result")
        self.assertNotContains(response, "32593")

    def test_model_training_requires_a_current_upload(self):
        response = self.client.get(
            reverse("model_training"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No Dataset Uploaded")
        self.assertNotContains(response, "Model Training Completed")
        self.assertNotContains(response, "32593")

    def test_dashboard_keeps_upload_form_without_dataset_state(self):
        response = self.client.get(
            reverse("upload_dataset"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload Your Dataset")

    def test_upload_marks_dataset_available_for_preprocessing(self):
        with TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "uploaded_dataset.csv"
            processed_path = Path(directory) / "processed_dataset.csv"
            dataset = SimpleUploadedFile(
                "test.csv",
                b"feature,final_result\n1,Pass\n2,Fail\n",
                content_type="text/csv",
            )
            with patch(
                "core.views.get_dataset_path",
                return_value=str(dataset_path),
            ), patch(
                "core.views.get_processed_dataset_path",
                return_value=str(processed_path),
            ):
                upload_response = self.client.post(
                    reverse("upload_dataset"),
                    {"dataset": dataset},
                )
                self.assertEqual(upload_response.status_code, 200)
                self.assertTrue(
                    self.client.session.get("generic_dataset_uploaded")
                )

                preprocessing_response = self.client.get(
                    reverse("data_preprocessing"),
                )
                self.assertEqual(preprocessing_response.status_code, 200)
                self.assertContains(
                    preprocessing_response,
                    "Original Records",
                )

    def test_dashboard_preserves_current_dataset_after_navigation(self):
        with TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "uploaded_dataset.csv"
            processed_path = Path(directory) / "processed_dataset.csv"
            dataset = SimpleUploadedFile(
                "dataset_a.csv",
                b"feature,final_result\n1,Pass\n2,Fail\n",
                content_type="text/csv",
            )
            with patch(
                "core.views.get_dataset_path",
                return_value=str(dataset_path),
            ), patch(
                "core.views.get_processed_dataset_path",
                return_value=str(processed_path),
            ):
                self.client.post(
                    reverse("upload_dataset"),
                    {"dataset": dataset},
                )
                self.client.get(reverse("data_preprocessing"))
                dashboard_response = self.client.get(
                    reverse("upload_dataset")
                )

                self.assertContains(
                    dashboard_response,
                    "dataset_a.csv",
                )
                self.assertContains(
                    dashboard_response,
                    "Dataset analysed successfully.",
                )

    def test_replacing_dataset_updates_current_dataset(self):
        with TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "uploaded_dataset.csv"
            processed_path = Path(directory) / "processed_dataset.csv"
            with patch(
                "core.views.get_dataset_path",
                return_value=str(dataset_path),
            ), patch(
                "core.views.get_processed_dataset_path",
                return_value=str(processed_path),
            ):
                for filename, value in (
                    ("dataset_a.csv", b"feature,final_result\n1,Pass\n"),
                    ("dataset_b.csv", b"feature,final_result\n1,Fail\n2,Pass\n"),
                ):
                    dataset = SimpleUploadedFile(
                        filename,
                        value,
                        content_type="text/csv",
                    )
                    self.client.post(
                        reverse("upload_dataset"),
                        {"dataset": dataset},
                    )

                response = self.client.get(reverse("upload_dataset"))
                self.assertContains(response, "dataset_b.csv")
                self.assertNotContains(response, "dataset_a.csv")
                self.assertContains(response, "Number of Records")

    def test_remove_dataset_clears_generic_state(self):
        with TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "uploaded_dataset.csv"
            processed_path = Path(directory) / "processed_dataset.csv"
            dataset_path.write_text(
                "feature,final_result\n1,Pass\n",
                encoding="utf-8",
            )
            with patch(
                "core.views.get_dataset_path",
                return_value=str(dataset_path),
            ), patch(
                "core.views.get_processed_dataset_path",
                return_value=str(processed_path),
            ):
                session = self.client.session
                session["generic_dataset_uploaded"] = True
                session["generic_dataset_name"] = "dataset_a.csv"
                session.save()

                response = self.client.post(
                    reverse("remove_dataset")
                )
                self.assertRedirects(
                    response,
                    reverse("upload_dataset"),
                )
                dashboard_response = self.client.get(
                    reverse("upload_dataset")
                )
                preprocessing_response = self.client.get(
                    reverse("data_preprocessing")
                )

                self.assertContains(
                    dashboard_response,
                    "No Dataset Analysed Yet",
                )
                self.assertContains(
                    preprocessing_response,
                    "No Dataset Uploaded",
                )
                self.assertFalse(dataset_path.exists())
    analyze_early_warning_thresholds,
    explain_learner,
