"""Airplane classifier module"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class AirplaneClassifier:
    """Classifier for airplane types detected by YOLO"""

    def __init__(self, model_path: Path):
        """Initialize classifier with model path"""
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self._load_model()

    def _load_model(self):
        """Load ONNX classification model"""
        # Placeholder
        pass

    def classify(self, cropped_image: np.ndarray) -> Dict:
        """Classify cropped airplane image

        Args:
            cropped_image: cropped airplane image

        Returns:
            Dict with class name and confidence
        """
        # Placeholder
        return {"class": "unknown", "confidence": 0.0}
