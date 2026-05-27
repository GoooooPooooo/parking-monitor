from django.urls import path

from . import views

urlpatterns = [
    path("upload/", views.upload_image, name="upload-image"),
    path("detect/", views.run_detection, name="run-detection"),
    path("detect-and-classify/", views.detect_and_classify, name="detect-and-classify"),
    path("classify-only/", views.classify_only, name="classify-only"),
    path("models/", views.list_models, name="list-models"),
    path("mlflow/models/", views.list_mlflow_models, name="list-mlflow-models"),
    path("mlflow/download/", views.download_mlflow_model, name="download-mlflow-model"),
    path("history/", views.list_history, name="list-history"),
    path("mlflow/metrics/", views.get_mlflow_metrics, name="mlflow-metrics"),
    path("results/<str:result_id>/", views.get_result, name="get-result"),
    path("results/<str:result_id>/delete/", views.delete_result, name="delete-result"),
    path("image/<str:result_id>/", views.get_image, name="get-image"),
]
