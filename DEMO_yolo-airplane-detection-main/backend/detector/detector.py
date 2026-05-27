"""YOLO Detector using ONNX Runtime"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO object detector using ONNX Runtime"""

    def __init__(self, model_path: Path, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """Initialize detector with ONNX model

        Args:
            model_path: Path to ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._load_model()

    def _load_model(self):
        """Load ONNX model with ONNX Runtime"""
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)

            # Get model input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            # Get input shape
            input_shape = self.session.get_inputs()[0].shape
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]

            logger.info(f"Loaded ONNX model: {self.model_path.name}")
            logger.info(f"Input shape: {input_shape}")
            logger.info(f"Providers: {self.session.get_providers()}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for YOLO model

        Args:
            image: Input image (BGR format)

        Returns:
            Preprocessed tensor, scale factor, padding
        """
        # Get original dimensions
        orig_h, orig_w = image.shape[:2]

        # Resize with aspect ratio preservation
        scale = min(self.input_width / orig_w, self.input_height / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - new_w) // 2
        pad_y = (self.input_height - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Convert to RGB and normalize
        image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0

        # Transpose to CHW format and add batch dimension
        image_transposed = np.transpose(image_norm, (2, 0, 1))
        image_batch = np.expand_dims(image_transposed, axis=0)

        return image_batch, scale, (pad_x, pad_y)

    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        padding: Tuple[int, int],
        orig_shape: Tuple[int, int]
    ) -> List[Dict]:
        """Postprocess YOLO outputs

        Args:
            outputs: Raw model outputs
            scale: Scale factor from preprocessing
            padding: Padding from preprocessing
            orig_shape: Original image shape (h, w)

        Returns:
            List of detections with bbox, confidence, class
        """
        # YOLOv8 output shape: (1, 84, 8400) or similar
        # Format: [x, y, w, h, class_scores...]
        output = outputs[0]

        # Transpose to (8400, 84)
        if len(output.shape) == 3:
            output = output[0].T

        # Extract boxes and scores
        boxes = output[:, :4]  # x, y, w, h
        scores = output[:, 4:]  # class scores

        # Get class with max score
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert from xywh to xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Adjust for padding and scale
        pad_x, pad_y = padding
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / scale

        # Clip to image boundaries
        orig_h, orig_w = orig_shape
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        # Apply NMS
        indices = self.nms(boxes_xyxy, confidences, self.iou_threshold)

        # Build detections
        detections = []
        for idx in indices:
            detections.append({
                "bbox": boxes_xyxy[idx].tolist(),
                "confidence": float(confidences[idx]),
                "class_id": int(class_ids[idx]),
                "class_name": "airplane"
            })

        return detections

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression

        Args:
            boxes: Bounding boxes (N, 4) in xyxy format
            scores: Confidence scores (N,)
            iou_threshold: IoU threshold

        Returns:
            Indices of boxes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, image: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        """Run detection on image

        Args:
            image: Input image (BGR format)
            conf_threshold: Optional confidence threshold override

        Returns:
            List of detections
        """
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold

        orig_shape = image.shape[:2]

        # Preprocess
        input_tensor, scale, padding = self.preprocess(image)

        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess
        detections = self.postprocess(outputs, scale, padding, orig_shape)

        return detections
