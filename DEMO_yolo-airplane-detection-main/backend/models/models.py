"""Django models for the application"""

import uuid

from django.db import models
from django.utils import timezone


class DetectionTask(models.Model):
    """Task for tracking detection jobs"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    model_id = models.CharField(max_length=100)
    input_image = models.ImageField(upload_to="inputs/")
    output_image = models.ImageField(upload_to="outputs/", null=True, blank=True)
    error_message = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Task {self.id} - {self.status}"


class DetectionResult(models.Model):
    """Individual detection result"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.ForeignKey(
        DetectionTask, on_delete=models.CASCADE, related_name="results"
    )

    class_name = models.CharField(max_length=100)
    confidence = models.FloatField()

    # Bounding box coordinates (normalized 0-1)
    x_min = models.FloatField()
    y_min = models.FloatField()
    x_max = models.FloatField()
    y_max = models.FloatField()

    # Optional classification result
    classified_type = models.CharField(max_length=100, null=True, blank=True)
    classification_confidence = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.class_name} ({self.confidence:.2f})"
