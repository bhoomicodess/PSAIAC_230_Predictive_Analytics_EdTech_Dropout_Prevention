from django.contrib import admin
from django.contrib.auth.views import LogoutView
from django.urls import path

from core import views


urlpatterns = [

    path(
        "health/",
        views.health_check,
        name="health"
    ),

    path(
        "admin/",
        admin.site.urls
    ),

    path(
        "login/",
        views.login_view,
        name="login"
    ),

    path(
        "logout/",
        views.logout_view,
        name="logout"
    ),

    path(
        "access-denied/",
        views.access_denied,
        name="access_denied"
    ),

    path(
        "settings/",
        views.settings_view,
        name="settings"
    ),

    path(
        "help/",
        views.help_view,
        name="help"
    ),

    path(
        "",
        views.upload_dataset,
        name="upload_dataset"
    ),

    path(
        "remove-dataset/",
        views.remove_dataset,
        name="remove_dataset"
    ),

    path(
        "preprocessing/",
        views.data_preprocessing,
        name="data_preprocessing"
    ),

    path(
        "model-training/",
        views.model_training,
        name="model_training"
    ),

    path(
        "early-warning-training/",
        views.early_warning_training,
        name="early_warning_training"
    ),

    path(
        "individual-prediction/",
        views.individual_prediction,
        name="individual_prediction"
    ),

    path(
        "recommendations/",
        views.recommendations,
        name="recommendations"
    ),

    path(
        "student-risk-dashboard/",
        views.student_risk_dashboard,
        name="student_risk_dashboard"
    ),

    path(
        "batch-prediction/",
        views.batch_prediction,
        name="batch_prediction"
    ),

    path(
        "batch-prediction/download/",
        views.download_batch_predictions,
        name="download_batch_predictions"
    ),

    path(
        "model-performance/",
        views.model_performance,
        name="model_performance"
    ),

    path(
        "reports/",
        views.reports,
        name="reports"
    ),

    path(
        "reports/download/",
        views.download_report,
        name="download_report"
    ),

    path(
        "interventions/",
        views.interventions,
        name="interventions"
    ),

    path(
        "interventions/record/",
        views.record_intervention,
        name="record_intervention"
    ),

    path(
        "interventions/<int:intervention_id>/update/",
        views.update_intervention,
        name="update_intervention"
    ),
]
