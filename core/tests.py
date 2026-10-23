from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import timedelta
from unittest.mock import patch

import pandas as pd

from django.test import TestCase, override_settings
from django.urls import reverse
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.exceptions import ValidationError
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.utils.crypto import get_random_string
from django.utils import timezone

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
from .recommendation_engine import REQUIRED_FIELDS, generate_recommendations
from .batch_prediction import (
    DOWNLOAD_COLUMNS,
    REQUIRED_BATCH_COLUMNS,
    predict_batch,
    summarise_risk_levels,
    validate_batch_input,
)
from .models import Intervention, StudentProfile


TEST_PASSWORD = get_random_string(32)


def get_or_create_group(name):
    return Group.objects.get_or_create(name=name)[0]


def make_role_user(username, role, password=TEST_PASSWORD):
    user = get_user_model().objects.create_user(username=username, password=password)
    user.groups.add(get_or_create_group(role))
    return user


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


class RecommendationEngineTests(TestCase):

    def context(self, risk_level="Medium", **features):
        return {
            "student_id": 101,
            "module": "AAA",
            "presentation": "2013J",
            "dropout_probability": 0.5,
            "risk_level": risk_level,
            "early_warning_features": {
                "early_assessment_scheduled_count": 2,
                "early_assessment_count": 2,
                "early_assessment_submission_rate": 1.0,
                "early_assessment_score_mean": 75.0,
                **features,
            },
            "higher_risk_contributions": [],
            "lower_risk_contributions": [],
        }

    def test_low_risk_learner_receives_lightweight_guidance(self):
        recommendations = generate_recommendations(
            self.context("Low", early_assessment_submission_rate=0.0)
        )
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["priority"], "Low")
        self.assertIn("Continue current learning progress", recommendations[0]["message"])

    def test_medium_risk_without_specific_indicator_is_monitored(self):
        recommendations = generate_recommendations(self.context("Medium"))
        self.assertEqual(recommendations[0]["category"], "Progress Monitoring")
        self.assertEqual(recommendations[0]["priority"], "Medium")

    def test_high_risk_with_multiple_indicators_receives_high_priority_support(self):
        recommendations = generate_recommendations(
            self.context(
                "High",
                early_assessment_submission_rate=0.25,
                early_assessment_score_mean=42.0,
            )
        )
        self.assertGreaterEqual(len(recommendations), 3)
        self.assertTrue(any(item["priority"] == "High" for item in recommendations))
        self.assertTrue(
            any(item["category"] == "Learner Follow-up" for item in recommendations)
        )

    def test_weak_early_assessment_performance_triggers_learning_support(self):
        recommendations = generate_recommendations(
            self.context(early_assessment_score_mean=49.0)
        )
        self.assertTrue(
            any(item["category"] == "Learning Support" for item in recommendations)
        )

    def test_low_submission_rate_triggers_assessment_support(self):
        recommendations = generate_recommendations(
            self.context(early_assessment_submission_rate=0.5)
        )
        self.assertTrue(
            any(item["category"] == "Assessment Support" for item in recommendations)
        )

    def test_multiple_matching_rules_do_not_produce_duplicates(self):
        recommendations = generate_recommendations(
            self.context(
                "High",
                early_assessment_submission_rate=0.25,
                early_assessment_score_mean=42.0,
            )
        )
        keys = [(item["category"], item["title"]) for item in recommendations]
        self.assertEqual(len(keys), len(set(keys)))

    def test_recommendations_have_expected_structured_fields(self):
        recommendations = generate_recommendations(self.context())
        self.assertTrue(recommendations)
        self.assertTrue(REQUIRED_FIELDS.issubset(recommendations[0]))

    def test_invalid_or_missing_prediction_context_is_safe(self):
        self.assertEqual(generate_recommendations(None), [])
        self.assertEqual(generate_recommendations({"risk_level": "Medium"}), [])
        self.assertEqual(
            generate_recommendations(
                {"risk_level": "Low", "dropout_probability": 0.1}
            ),
            [],
        )
        self.assertEqual(
            generate_recommendations(
                {"risk_level": "Unknown", "dropout_probability": 0.5}
            ),
            [],
        )


class BatchPredictionTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.data_directory = Path(__file__).resolve().parents[1] / "data"
        cls.dataset_path = cls.data_directory / "early_warning_dataset.csv"
        cls.artifact_directory = cls.data_directory / "test_batch_artifacts"
        cls.table = pd.read_csv(cls.dataset_path)
        persist_selected_model(cls.dataset_path, cls.artifact_directory)

    @classmethod
    def tearDownClass(cls):
        for path in cls.artifact_directory.glob("*"):
            path.unlink()
        cls.artifact_directory.rmdir()
        super().tearDownClass()

    def setUp(self):
        self.user = make_role_user("batch_admin", "ADMIN")
        self.client.force_login(self.user)
        cache.clear()
        self.sample = self.table.loc[:2, REQUIRED_BATCH_COLUMNS].copy()
        self.dashboard_results = [
            {
                "id_student": 1000 + index,
                "code_module": "AAA" if index < 20 else "BBB",
                "code_presentation": "2013J",
                "dropout_probability": 0.95 - index / 100,
                "risk_level": "High" if index < 10 else ("Medium" if index < 20 else "Low"),
            }
            for index in range(30)
        ]

    def test_valid_batch_prediction_has_expected_structure_and_identifiers(self):
        results = predict_batch(self.sample, self.artifact_directory)
        self.assertEqual(len(results), len(self.sample))
        self.assertEqual(set(results[0]), set(DOWNLOAD_COLUMNS))
        self.assertEqual(results[0]["id_student"], self.sample.iloc[0]["id_student"])
        self.assertEqual(results[0]["code_module"], self.sample.iloc[0]["code_module"])
        self.assertIn(results[0]["risk_level"], {"Low", "Medium", "High"})
        self.assertGreaterEqual(results[0]["dropout_probability"], 0)
        self.assertLessEqual(results[0]["dropout_probability"], 1)

    def test_missing_required_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "early_assessment_score_mean"):
            validate_batch_input(self.sample.drop(columns=["early_assessment_score_mean"]))

    def test_outcome_columns_are_not_prediction_inputs_or_outputs(self):
        clean_results = predict_batch(self.sample, self.artifact_directory)
        outcome_table = self.sample.assign(dropout=1, final_result="Withdrawn")
        outcome_results = predict_batch(outcome_table, self.artifact_directory)
        self.assertEqual(clean_results, outcome_results)
        self.assertNotIn("dropout", outcome_results[0])
        self.assertNotIn("final_result", outcome_results[0])

    def test_batch_risk_summary_uses_result_records(self):
        summary = summarise_risk_levels(
            [
                {"risk_level": "High"},
                {"risk_level": "Medium"},
                {"risk_level": "Low"},
            ]
        )
        self.assertEqual(summary, {
            "total_learners": 3, "high_risk": 1, "medium_risk": 1, "low_risk": 1,
        })

    def _dashboard_response(self, query=None):
        with patch(
            "core.views._dashboard_prediction_results",
            return_value=(pd.DataFrame(), self.dashboard_results),
        ), patch(
            "core.views._add_dashboard_indicators",
            side_effect=lambda rows, table, artifacts: rows,
        ):
            return self.client.get(reverse("student_risk_dashboard"), query or {})

    def test_student_risk_dashboard_loads_and_filters_search_module_and_risk(self):
        response = self._dashboard_response(
            {"search": "1001", "risk": "High", "module": "AAA"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Student Risk Dashboard")
        self.assertContains(response, "1001")
        self.assertNotContains(response, "1025")

    def test_dashboard_sorting_pagination_and_view_student_link(self):
        response = self._dashboard_response({"sort": "student", "page": 2})
        content = response.content.decode()
        self.assertContains(response, "Showing 26–30 of 30 learners")
        self.assertIn("id_student=1025", content)
        self.assertLess(content.find("1025"), content.find("1029"))
        self.assertIn("code_module=BBB", content)

    def test_batch_upload_and_download_work(self):
        upload = SimpleUploadedFile(
            "early_warning.csv",
            self.sample.to_csv(index=False).encode("utf-8"),
            content_type="text/csv",
        )
        with patch("core.views.predict_batch", return_value=self.dashboard_results):
            response = self.client.post(reverse("batch_prediction"), {"dataset": upload})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Prediction Complete")
        download = self.client.get(reverse("download_batch_predictions"))
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download["Content-Type"], "text/csv")
        content = download.content.decode()
        self.assertIn("dropout_probability", content)
        self.assertNotIn("final_result", content)

    def test_batch_upload_shows_missing_column_error(self):
        incomplete = SimpleUploadedFile(
            "incomplete.csv", b"id_student,code_module\n1,AAA\n", content_type="text/csv"
        )
        response = self.client.post(reverse("batch_prediction"), {"dataset": incomplete})
        self.assertContains(response, "missing required columns")


class IndividualPredictionTests(TestCase):

    def setUp(self):
        self.user = make_role_user("prediction_admin", "ADMIN")
        self.client.force_login(self.user)

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


class AnalyticsReportingTests(TestCase):

    def setUp(self):
        self.user = make_role_user("analytics_admin", "ADMIN")
        self.client.force_login(self.user)

    @classmethod
    def setUpTestData(cls):
        cls.data_directory = Path(__file__).resolve().parents[1] / "data"
        cls.dataset_path = cls.data_directory / "early_warning_dataset.csv"
        cls.artifact_directory = cls.data_directory / "test_early_warning_artifacts"
        generate_early_warning_dataset(cls.data_directory, cls.dataset_path)
        persist_selected_model(cls.dataset_path, cls.artifact_directory)

    @classmethod
    def tearDownClass(cls):
        for path in cls.artifact_directory.glob("*"):
            path.unlink()
        cls.artifact_directory.rmdir()
        super().tearDownClass()

    def test_model_performance_page_loads_and_displays_metrics(self):
        response = self.client.get(reverse("model_performance"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Model Performance")
        self.assertContains(response, "Logistic Regression")
        self.assertContains(response, "ROC-AUC")
        self.assertContains(response, "The ROC curve shows how model classification performance changes")

    def test_reports_page_loads_and_downloads_summary_csv(self):
        response = self.client.get(reverse("reports"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Learners Analysed")
        self.assertContains(response, "High Risk")
        self.assertContains(response, "Risk Distribution")
        self.assertContains(response, "Download Report")

        download = self.client.get(reverse("download_report"))
        self.assertEqual(download.status_code, 200)
        self.assertIn("text/csv", download["Content-Type"])
        self.assertIn("risk_level", download.content.decode())

    def test_dashboard_includes_attention_summary_and_intervention_overview(self):
        response = self.client.get(reverse("upload_dataset"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "What Needs Attention?")
        self.assertContains(response, "View Risk Dashboard")
        self.assertContains(response, "Intervention Overview")


class AuthenticationAccessControlTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.User = get_user_model()
        cls.groups = {
            name: Group.objects.create(name=name)
            for name in ("ADMIN", "EDUCATOR", "STUDENT")
        }

    def make_user(self, username, password=TEST_PASSWORD, role="ADMIN"):
        user = self.User.objects.create_user(username=username, password=password)
        user.groups.add(self.groups[role])
        return user

    def create_student_profile(self, user, student_id):
        return StudentProfile.objects.create(user=user, student_id=str(student_id))

    def test_unauthenticated_user_is_redirected_to_login(self):
        response = self.client.get(reverse("upload_dataset"))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("login"), response.url)

    def test_valid_login_succeeds_and_invalid_login_fails_safely(self):
        user = self.make_user("adminuser", role="ADMIN")
        login_response = self.client.post(
            reverse("login"),
            {"username": "adminuser", "password": TEST_PASSWORD},
            follow=True,
        )
        self.assertEqual(login_response.status_code, 200)
        self.assertTrue(login_response.wsgi_request.user.is_authenticated)

        invalid_response = self.client.post(
            reverse("login"),
            {"username": "adminuser", "password": "wrong-password"},
            follow=True,
        )
        self.assertContains(invalid_response, "Invalid username or password")
        self.assertContains(invalid_response, "Please check your credentials and try again")

    def test_logout_revokes_access_to_protected_pages(self):
        user = self.make_user("educatoruser", role="EDUCATOR")
        self.client.force_login(user)
        self.client.get(reverse("logout"))
        protected_response = self.client.get(reverse("student_risk_dashboard"))
        self.assertEqual(protected_response.status_code, 302)
        self.assertIn(reverse("login"), protected_response.url)

    def test_admin_can_access_model_training_and_reports(self):
        user = self.make_user("admin1", role="ADMIN")
        self.client.force_login(user)
        training_response = self.client.get(reverse("model_training"))
        reports_response = self.client.get(reverse("reports"))
        self.assertEqual(training_response.status_code, 200)
        self.assertEqual(reports_response.status_code, 200)

    def test_educator_can_access_student_support_pages_but_not_admin_training(self):
        user = self.make_user("educator1", role="EDUCATOR")
        self.client.force_login(user)
        self.assertEqual(self.client.get(reverse("student_risk_dashboard")).status_code, 200)
        self.assertEqual(self.client.get(reverse("recommendations")).status_code, 200)
        self.assertEqual(self.client.get(reverse("interventions")).status_code, 200)
        self.assertEqual(self.client.get(reverse("model_training")).status_code, 403)

    def test_student_cannot_access_other_students_or_raw_datasets_and_reports(self):
        user = self.make_user("student1", role="STUDENT")
        other_user = self.make_user("student2", role="STUDENT")
        self.create_student_profile(user, "60001")
        self.create_student_profile(other_user, "60002")
        self.client.force_login(user)

        own_prediction = self.client.get(reverse("individual_prediction") + "?id_student=60001"
        )
        self.assertEqual(own_prediction.status_code, 200)

        other_student = self.client.get(reverse("individual_prediction") + "?id_student=60002")
        self.assertEqual(other_student.status_code, 403)

        self.assertEqual(self.client.get(reverse("reports")).status_code, 403)
        self.assertEqual(self.client.get(reverse("model_training")).status_code, 403)

    def test_student_id_manipulation_is_rejected_even_when_param_is_present(self):
        user = self.make_user("student3", role="STUDENT")
        self.create_student_profile(user, "70011")
        self.client.force_login(user)
        for route_name in ("individual_prediction", "recommendations"):
            with self.subTest(route_name=route_name):
                response = self.client.get(
                    reverse(route_name),
                    {"id_student": "70012", "code_module": "AAA", "code_presentation": "2013J"},
                )
                self.assertEqual(response.status_code, 403)

    def test_student_learner_options_only_contain_the_mapped_learner(self):
        user = self.make_user("student_options", role="STUDENT")
        self.create_student_profile(user, "60001")
        Intervention.objects.create(
            student_id="60001",
            code_module="AAA",
            code_presentation="2013J",
            intervention_type=Intervention.InterventionType.ASSESSMENT_SUPPORT,
            priority=Intervention.Priority.MEDIUM,
            status=Intervention.Status.PENDING,
            risk_probability_at_intervention=0.20,
        )
        self.client.force_login(user)
        table = pd.DataFrame(
            [
                {"id_student": "60001", "code_module": "AAA", "code_presentation": "2013J",
                 "registration_offset_days": -10, "early_assessment_count": 1,
                 "early_assessment_score_mean": 80.0, "early_assessment_score_std": 0.0,
                 "early_assessment_scheduled_count": 1, "early_assessment_submission_rate": 1.0},
                {"id_student": "60002", "code_module": "BBB", "code_presentation": "2013B",
                 "registration_offset_days": -12, "early_assessment_count": 1,
                 "early_assessment_score_mean": 70.0, "early_assessment_score_std": 0.0,
                 "early_assessment_scheduled_count": 1, "early_assessment_submission_rate": 1.0},
            ]
        )
        prediction = {
            "row": table.iloc[0].to_dict(),
            "dropout_probability": 0.20,
            "risk_band": "Low",
            "predicted_class": 0,
            "selected_model": "Logistic Regression",
        }
        common_patches = (
            patch("core.views.get_early_warning_dataset_path", return_value="dataset.csv"),
            patch("core.views.get_early_warning_artifact_directory", return_value="artifacts"),
            patch("core.views.os.path.exists", return_value=True),
            patch("core.views.pd.read_csv", return_value=table),
            patch("core.views.predict_learner", return_value=prediction),
            patch("core.views.explain_learner", return_value={"higher": [], "lower": []}),
        )
        with common_patches[0], common_patches[1], common_patches[2], common_patches[3], common_patches[4], common_patches[5]:
            responses = [
                self.client.get(
                    reverse("individual_prediction"),
                    {"id_student": "60001", "code_module": "AAA", "code_presentation": "2013J"},
                ),
                self.client.get(
                    reverse("recommendations"),
                    {"id_student": "60001", "code_module": "AAA", "code_presentation": "2013J"},
                ),
                self.client.get(reverse("individual_prediction")),
            ]

        for response in responses:
            self.assertEqual(response.status_code, 200)
            learner_ids = {
                str(option["id_student"])
                for option in response.context["learner_options"]
            }
            self.assertEqual(learner_ids, {"60001"})
            self.assertContains(response, "60001")
            self.assertNotContains(response, "60002")
        self.assertIsNotNone(responses[2].context["prediction"])
        self.assertEqual(
            str(responses[2].context["prediction"]["row"]["id_student"]),
            "60001",
        )
        self.assertContains(responses[0], "Assessment Support")
        self.assertNotContains(responses[0], "Record Intervention")
        self.assertNotContains(responses[0], "Edit follow-up")
        self.assertNotContains(responses[0], "Save update")

    def test_admin_and_educator_retain_full_learner_selection(self):
        table = pd.DataFrame(
            [
                {"id_student": "60001", "code_module": "AAA", "code_presentation": "2013J"},
                {"id_student": "60002", "code_module": "BBB", "code_presentation": "2013B"},
            ]
        )
        with patch("core.views.get_early_warning_dataset_path", return_value="dataset.csv"), patch(
            "core.views.get_early_warning_artifact_directory", return_value="artifacts"
        ), patch("core.views.os.path.exists", return_value=True), patch(
            "core.views.pd.read_csv", return_value=table
        ):
            for role in ("ADMIN", "EDUCATOR"):
                with self.subTest(role=role):
                    user = self.make_user(f"selector_{role.lower()}", role=role)
                    self.client.force_login(user)
                    for route_name in ("individual_prediction", "recommendations"):
                        response = self.client.get(reverse(route_name))
                        self.assertEqual(response.status_code, 200)
                        learner_ids = {
                            str(option["id_student"])
                            for option in response.context["learner_options"]
                        }
                        self.assertEqual(learner_ids, {"60001", "60002"})

    def test_student_root_route_has_no_aggregate_dashboard_or_upload(self):
        user = self.make_user("student_root", role="STUDENT")
        self.create_student_profile(user, "60001")
        self.client.force_login(user)

        dashboard = self.client.get(reverse("upload_dataset"))
        self.assertRedirects(dashboard, reverse("individual_prediction"), fetch_redirect_response=False)

        with patch("core.views.get_dataset_path") as get_dataset_path:
            upload = self.client.post(
                reverse("upload_dataset"),
                {"dataset": SimpleUploadedFile("restricted.csv", b"a,b\n1,2\n", content_type="text/csv")},
            )
        self.assertEqual(upload.status_code, 403)
        get_dataset_path.assert_not_called()

    def test_student_cannot_access_institutional_data_or_model_operations(self):
        user = self.make_user("student_operations", role="STUDENT")
        self.create_student_profile(user, "60001")
        self.client.force_login(user)
        restricted_routes = (
            "remove_dataset",
            "data_preprocessing",
            "model_training",
            "early_warning_training",
            "student_risk_dashboard",
            "batch_prediction",
            "download_batch_predictions",
            "interventions",
            "record_intervention",
            "reports",
            "download_report",
        )
        for route_name in restricted_routes:
            with self.subTest(route_name=route_name):
                self.assertEqual(self.client.get(reverse(route_name)).status_code, 403)
        self.assertEqual(
            self.client.get(reverse("update_intervention", args=[1])).status_code,
            403,
        )

    def test_educator_root_routes_to_support_dashboard_and_admin_operations_are_denied(self):
        user = self.make_user("educator_boundary", role="EDUCATOR")
        self.client.force_login(user)
        dashboard = self.client.get(reverse("upload_dataset"))
        self.assertRedirects(dashboard, reverse("student_risk_dashboard"), fetch_redirect_response=False)
        upload = self.client.post(
            reverse("upload_dataset"),
            {"dataset": SimpleUploadedFile("restricted.csv", b"a,b\n1,2\n", content_type="text/csv")},
        )
        self.assertEqual(upload.status_code, 403)
        for route_name in ("remove_dataset", "data_preprocessing", "model_training", "early_warning_training"):
            with self.subTest(route_name=route_name):
                self.assertEqual(self.client.get(reverse(route_name)).status_code, 403)
        self.assertEqual(self.client.get(reverse("interventions")).status_code, 200)

    def test_admin_retains_dashboard_and_administrative_operations(self):
        user = self.make_user("admin_boundary", role="ADMIN")
        self.client.force_login(user)
        self.assertEqual(self.client.get(reverse("upload_dataset")).status_code, 200)
        self.assertEqual(self.client.get(reverse("data_preprocessing")).status_code, 200)
        self.assertEqual(self.client.get(reverse("model_training")).status_code, 200)
        with patch("core.views.generate_early_warning_dataset"), patch(
            "core.views.train_early_warning_models", return_value={"results": []}
        ), patch("core.views.analyze_early_warning_thresholds", return_value={}), patch(
            "core.views.persist_selected_model",
            return_value={"selected_model": "Logistic Regression"},
        ):
            early_warning_response = self.client.get(reverse("early_warning_training"))
        self.assertEqual(early_warning_response.status_code, 200)

    def test_post_forms_require_csrf_and_unauthorized_access_is_rejected(self):
        user = self.make_user("student4", role="STUDENT")
        self.create_student_profile(user, "80001")
        self.client.force_login(user)
        response = self.client.post(reverse("record_intervention"), {
            "id_student": "80002",
            "code_module": "AAA",
            "code_presentation": "2013J",
            "intervention_type": "assessment_support",
            "priority": "High",
            "status": "Pending",
            "notes": "Attempted unauthorized action",
            "risk_probability_at_intervention": 0.75,
        })
        self.assertEqual(response.status_code, 403)


class InterventionWorkflowTests(TestCase):

    def setUp(self):
        self.user = make_role_user("intervention_educator", "EDUCATOR")
        self.client.force_login(self.user)
        self.student_id = "60001"
        self.module = "AAA"
        self.presentation = "2013J"
        self.learner_key = (self.student_id, self.module, self.presentation)
        self.prediction = {
            "row": {
                "id_student": self.student_id,
                "code_module": self.module,
                "code_presentation": self.presentation,
                "registration_offset_days": -10,
                "early_assessment_count": 2,
                "early_assessment_score_mean": 48.0,
                "early_assessment_score_std": 3.0,
                "early_assessment_scheduled_count": 3,
                "early_assessment_submission_rate": 0.66,
            },
            "dropout_probability": 0.52,
            "risk_band": "Medium",
            "predicted_class": 1,
            "selected_model": "Logistic Regression",
        }

    def create_intervention(self, **values):
        defaults = {
            "student_id": self.student_id,
            "code_module": self.module,
            "code_presentation": self.presentation,
            "intervention_type": Intervention.InterventionType.ASSESSMENT_SUPPORT,
            "priority": Intervention.Priority.MEDIUM,
            "status": Intervention.Status.PENDING,
            "follow_up_date": timezone.localdate() + timedelta(days=7),
            "notes": "Review upcoming assessment participation.",
            "risk_probability_at_intervention": 0.78,
        }
        defaults.update(values)
        return Intervention.objects.create(**defaults)

    def test_intervention_model_creation_and_controlled_choices(self):
        intervention = self.create_intervention()
        self.assertIsNotNone(intervention.pk)
        self.assertEqual(intervention.get_intervention_type_display(), "Assessment Support")
        self.assertEqual(intervention.get_status_display(), "Pending")

    def test_model_rejects_invalid_type_and_probability(self):
        intervention = Intervention(
            student_id=self.student_id,
            code_module=self.module,
            code_presentation=self.presentation,
            intervention_type="unvalidated type",
            priority=Intervention.Priority.HIGH,
            risk_probability_at_intervention=1.2,
        )
        with self.assertRaises(ValidationError):
            intervention.full_clean()

    def test_record_intervention_associates_learner_and_captures_current_risk(self):
        post_data = {
            "id_student": self.student_id,
            "code_module": self.module,
            "code_presentation": self.presentation,
            "intervention_type": Intervention.InterventionType.MENTOR_FOLLOW_UP,
            "priority": Intervention.Priority.HIGH,
            "follow_up_date": (timezone.localdate() + timedelta(days=7)).isoformat(),
            "notes": "Arrange a faculty check-in.",
            "risk_probability_at_intervention": "0.01",
        }
        with patch("core.views._current_learner_prediction", return_value=self.prediction):
            response = self.client.post(reverse("record_intervention"), post_data)
        self.assertEqual(response.status_code, 302)
        intervention = Intervention.objects.get()
        self.assertEqual(
            (intervention.student_id, intervention.code_module, intervention.code_presentation),
            self.learner_key,
        )
        self.assertEqual(intervention.risk_probability_at_intervention, 0.52)
        self.assertEqual(intervention.intervention_type, Intervention.InterventionType.MENTOR_FOLLOW_UP)

    def test_invalid_intervention_form_is_rejected_without_exception_details(self):
        post_data = {
            "id_student": self.student_id,
            "code_module": self.module,
            "code_presentation": self.presentation,
            "intervention_type": "free-form-invalid-value",
            "priority": "urgent-invalid-value",
        }
        with patch("core.views._current_learner_prediction", return_value=self.prediction):
            response = self.client.post(reverse("record_intervention"), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Please try again.")
        self.assertNotContains(response, "ValidationError")
        self.assertEqual(Intervention.objects.count(), 0)

    def test_status_update_preserves_student_identity_and_original_risk(self):
        intervention = self.create_intervention()
        response = self.client.post(
            reverse("update_intervention", args=[intervention.pk]),
            {
                "status": Intervention.Status.COMPLETED,
                "notes": "Support session completed.",
                "follow_up_date": (timezone.localdate() + timedelta(days=14)).isoformat(),
                "student_id": "tampered",
                "risk_probability_at_intervention": "0.05",
            },
        )
        self.assertEqual(response.status_code, 302)
        intervention.refresh_from_db()
        self.assertEqual(intervention.status, Intervention.Status.COMPLETED)
        self.assertEqual(intervention.notes, "Support session completed.")
        self.assertEqual(intervention.student_id, self.student_id)
        self.assertEqual(intervention.risk_probability_at_intervention, 0.78)

    def test_overdue_is_calculated_for_open_actions_only(self):
        overdue_date = timezone.localdate() - timedelta(days=1)
        pending = self.create_intervention(follow_up_date=overdue_date)
        completed = self.create_intervention(
            student_id="60002",
            status=Intervention.Status.COMPLETED,
            follow_up_date=overdue_date,
        )
        no_date = self.create_intervention(
            student_id="60003",
            status=Intervention.Status.IN_PROGRESS,
            follow_up_date=None,
        )
        self.assertTrue(pending.is_overdue)
        self.assertFalse(completed.is_overdue)
        self.assertFalse(no_date.is_overdue)

    def test_intervention_history_and_neutral_reassessment_render_on_profile(self):
        self.create_intervention()
        with patch("core.views.get_early_warning_dataset_path", return_value="unused.csv"), patch(
            "core.views.get_early_warning_artifact_directory", return_value="unused-artifacts"
        ), patch("core.views.os.path.exists", return_value=True), patch(
            "core.views.pd.read_csv", return_value=pd.DataFrame([self.prediction["row"]])
        ), patch("core.views.predict_learner", return_value=self.prediction), patch(
            "core.views.explain_learner", return_value={"higher": [], "lower": []}
        ):
            response = self.client.get(
                reverse("individual_prediction"),
                {"id_student": self.student_id, "code_module": self.module,
                 "code_presentation": self.presentation},
            )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Student Profile")
        self.assertContains(response, "Current Risk")
        self.assertContains(response, "52%")
        self.assertContains(response, "Intervention History")
        self.assertContains(response, "Assessment Support")
        self.assertContains(response, "Risk probability changed after the recorded intervention.")
        self.assertContains(response, "-26 percentage points")
        self.assertContains(response, "Recommendations")

    def test_profile_empty_history_and_recommendations_route_render(self):
        with patch("core.views.get_early_warning_dataset_path", return_value="unused.csv"), patch(
            "core.views.get_early_warning_artifact_directory", return_value="unused-artifacts"
        ), patch("core.views.os.path.exists", return_value=True), patch(
            "core.views.pd.read_csv", return_value=pd.DataFrame([self.prediction["row"]])
        ), patch("core.views.predict_learner", return_value=self.prediction), patch(
            "core.views.explain_learner", return_value={"higher": [], "lower": []}
        ):
            profile = self.client.get(
                reverse("individual_prediction"),
                {"id_student": self.student_id, "code_module": self.module,
                 "code_presentation": self.presentation},
            )
            recommendations = self.client.get(
                reverse("recommendations"),
                {"id_student": self.student_id, "code_module": self.module,
                 "code_presentation": self.presentation},
            )
        self.assertContains(profile, "No interventions recorded yet.")
        self.assertContains(profile, 'href="/interventions/record/?')
        self.assertEqual(recommendations.status_code, 200)
        self.assertContains(recommendations, "Personalized Recommendations")
        self.assertContains(recommendations, "Additional Learning Support")

    def test_intervention_filters_and_pagination(self):
        self.create_intervention(student_id="70001")
        self.create_intervention(
            student_id="70002",
            status=Intervention.Status.COMPLETED,
            priority=Intervention.Priority.HIGH,
            intervention_type=Intervention.InterventionType.LEARNING_SUPPORT,
        )
        self.create_intervention(student_id="70003")
        response = self.client.get(
            reverse("interventions"),
            {
                "search": "70002",
                "status": Intervention.Status.COMPLETED,
                "priority": Intervention.Priority.HIGH,
                "type": Intervention.InterventionType.LEARNING_SUPPORT,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "70002")
        self.assertNotContains(response, "70001")
        self.assertEqual(response.context["page_obj"].paginator.count, 1)

        for student_number in range(71000, 71026):
            self.create_intervention(student_id=str(student_number))
        paged = self.client.get(reverse("interventions"))
        self.assertEqual(paged.context["page_obj"].paginator.count, 29)
        self.assertEqual(len(paged.context["page_obj"].object_list), 25)

    def test_intervention_counts_use_database_and_educator_root_routes_to_risk_dashboard(self):
        self.create_intervention()
        self.create_intervention(
            student_id="80002",
            status=Intervention.Status.IN_PROGRESS,
            follow_up_date=timezone.localdate() - timedelta(days=1),
        )
        self.create_intervention(
            student_id="80003", status=Intervention.Status.COMPLETED
        )
        queue = self.client.get(reverse("interventions"))
        self.assertEqual(
            queue.context["summary"],
            {"total_interventions": 3, "pending": 1, "in_progress": 1,
             "completed": 1, "overdue": 1},
        )
        dashboard = self.client.get(reverse("upload_dataset"))
        self.assertRedirects(
            dashboard,
            reverse("student_risk_dashboard"),
            fetch_redirect_response=False,
        )
        risk_dashboard = self.client.get(reverse("student_risk_dashboard"))
        self.assertEqual(risk_dashboard.status_code, 200)
        self.assertEqual(risk_dashboard.context["intervention_overview"]["overdue"], 1)


@override_settings(ALLOWED_HOSTS=["localhost"])
class NavigationTests(TestCase):

    def setUp(self):
        self.user = make_role_user("navigation_admin", "ADMIN")
        self.client.force_login(self.user)

    navigation_routes = {
        "upload_dataset": "/",
        "data_preprocessing": "/preprocessing/",
        "model_training": "/model-training/",
        "early_warning_training": "/early-warning-training/",
        "individual_prediction": "/individual-prediction/",
        "recommendations": "/recommendations/",
        "student_risk_dashboard": "/student-risk-dashboard/",
        "batch_prediction": "/batch-prediction/",
        "interventions": "/interventions/",
    }

    def test_major_pages_render_all_navigation_links(self):
        pages = [
            reverse("upload_dataset"),
            reverse("data_preprocessing"),
            reverse("model_training"),
            reverse("early_warning_training"),
            reverse("individual_prediction"),
            reverse("recommendations"),
            reverse("student_risk_dashboard"),
            reverse("batch_prediction"),
            reverse("interventions"),
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

    def setUp(self):
        self.user = make_role_user("dataset_admin", "ADMIN")
        self.client.force_login(self.user)

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
