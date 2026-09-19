from django.contrib import admin
from django.urls import path

from core import views


urlpatterns = [

    path(
        "admin/",
        admin.site.urls
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
]