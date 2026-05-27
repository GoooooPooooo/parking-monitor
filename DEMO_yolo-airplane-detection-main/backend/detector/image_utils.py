"""Image processing utilities"""

import base64
import io

import cv2
import numpy as np
from PIL import Image


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 string to numpy array"""
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def encode_base64_image(image: np.ndarray, format: str = ".jpg") -> str:
    """Encode numpy array to base64 string"""
    _, buffer = cv2.imencode(format, image)
    return base64.b64encode(buffer).decode("utf-8")


def preprocess_image(image: np.ndarray, target_size: int = 640) -> np.ndarray:
    """Preprocess image for YOLO model"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Pad to target size
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return padded, scale


def postprocess_detections(
    outputs: np.ndarray,
    scale: float,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
) -> list:
    """Post-process model outputs"""
    # Placeholder for NMS and filtering
    return []
