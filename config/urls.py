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
        "preprocessing/",
        views.data_preprocessing,
        name="data_preprocessing"
    ),

    path(
        "model-training/",
        views.model_training,
        name="model_training"
    ),
]