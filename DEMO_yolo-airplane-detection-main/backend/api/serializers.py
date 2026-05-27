"""Serializers for API"""

from rest_framework import serializers


class DetectionResultSerializer(serializers.Serializer):
    """Serializer for detection results"""

    image_url = serializers.URLField()
    detections = serializers.ListField(child=serializers.DictField(), allow_empty=True)
    model_used = serializers.CharField(max_length=100)
    processing_time = serializers.FloatField(required=False)


class ImageUploadSerializer(serializers.Serializer):
    """Serializer for image upload"""

    image = serializers.ImageField()
    model_id = serializers.ChoiceField(
        choices=["yolov8n-base", "yolov8n-mod1", "yolov8n-mod2"], default="yolov8n-base"
    )
