"""API views for airplane detection"""

import io
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List

import cv2
import numpy as np
from django.conf import settings
from django.http import FileResponse
from pathlib import Path
from PIL import Image
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from core.storage import minio_client

logger = logging.getLogger(__name__)

RESULTS_STORE: Dict[str, Any] = {}
IMAGE_CACHE: Dict[str, np.ndarray] = {}  # Cache original images for classification


def load_yolo_model(model_id: str = "baseline"):
    """Load YOLO model - uses ONNX Runtime for .onnx files"""
    try:
        from detector.detector import YOLODetector
        from ultralytics import YOLO

        # Check for MLflow model (format: mlflow:<run_id>)
        if model_id.startswith("mlflow:"):
            run_id = model_id.split(":", 1)[1]
            import mlflow
            mlflow.set_tracking_uri("http://mlflow:5000")
            client = mlflow.tracking.MlflowClient()

            # Download model to temp directory
            import tempfile
            temp_dir = tempfile.mkdtemp()

            # Find model artifact
            artifacts = client.list_artifacts(run_id)
            pt_artifact = None
            onnx_artifact = None

            for artifact in artifacts:
                if artifact.path.endswith('best.pt') or artifact.path.endswith('last.pt'):
                    pt_artifact = artifact.path
                elif 'onnx' in artifact.path.lower() and artifact.path.endswith('.onnx'):
                    onnx_artifact = artifact.path

            # Prefer ONNX, fallback to PT
            artifact_path = onnx_artifact if onnx_artifact else pt_artifact
            if not artifact_path:
                logger.error(f"No model artifacts found for run_id: {run_id}")
                return None

            local_path = client.download_artifacts(run_id, artifact_path, dst_path=temp_dir)

            # Load model
            if artifact_path.endswith('.onnx'):
                logger.info(f"Loading ONNX model from MLflow: {run_id}")
                return YOLODetector(local_path, conf_threshold=0.1)
            else:
                logger.info(f"Loading PT model from MLflow: {run_id}")
                return YOLO(local_path)

        # Check for ONNX models in /weights/ (preferred)
        weights_dir = Path("/weights")
        if weights_dir.exists():
            onnx_path = weights_dir / f"{model_id}.onnx"
            if onnx_path.exists():
                logger.info(f"Loading ONNX model with ONNX Runtime: {onnx_path}")
                return YOLODetector(onnx_path, conf_threshold=0.1)

            # Fallback to .pt if ONNX not found
            pt_path = weights_dir / f"{model_id}.pt"
            if pt_path.exists():
                logger.warning(f"ONNX not found, using PyTorch: {pt_path}")
                return YOLO(str(pt_path))

        # Try default pretrained
        logger.info(f"Loading pretrained {model_id}")
        return YOLO(f"{model_id}.pt")
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def detect_airplanes(
    image: np.ndarray, model_id: str = "baseline"
) -> List[Dict[str, Any]]:
    """Run airplane detection on image"""
    model = load_yolo_model(model_id)
    if model is None:
        return []

    try:
        # Check if it's ONNX Runtime detector or Ultralytics YOLO
        if hasattr(model, 'detect') and callable(getattr(model, 'detect', None)):  # ONNX Runtime detector
            logger.info(f"Using ONNX Runtime detector: {model_id}")
            detections = model.detect(image)
        else:  # Ultralytics YOLO
            logger.info(f"Using Ultralytics YOLO: {model_id}")
            # Use lower confidence threshold for all models
            conf = 0.1
            results = model(image, verbose=False, conf=conf)
            detections = []

            for r in results:
                boxes = r.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    # Filter only airplanes for pretrained yolov8n (class_id=4)
                    # For trained models (baseline, mod1, mod2), class_id=0 is airplane
                    if model_id == "yolov8n":
                        # Pretrained COCO model: airplane is class_id=4
                        if class_id != 4:
                            continue

                    detections.append(
                        {
                            "bbox": box.xyxy[0].tolist(),
                            "confidence": float(box.conf[0]),
                            "class_id": class_id,
                            "class_name": class_name,
                        }
                    )

        return detections
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes on image"""
    try:
        import cv2

        img = image.copy()
        colors = {
            "airplane": (255, 0, 0),
        }

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            conf = det["confidence"]
            class_name = det["class_name"]
            label = f"{class_name} {conf:.2f}"

            color = colors.get(class_name, (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        return img
    except Exception as e:
        logger.error(f"Failed to draw detections: {e}")
        return image


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_image(request):
    """Upload image, run detection, save to MinIO"""
    if "image" not in request.FILES:
        return Response(
            {"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST
        )

    image_file = request.FILES["image"]
    model_id = request.data.get("model", "yolov8n")

    allowed_extensions = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}

    try:
        ext = image_file.name.split(".")[-1].lower() if "." in image_file.name else ""
        if not ext or ext not in allowed_extensions:
            return Response(
                {"error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Read and detect
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)
        detections = detect_airplanes(image_np, model_id)

        # Draw detections
        output_image = draw_detections(image_np, detections)

        # Generate IDs
        result_id = str(uuid.uuid4())
        input_key = f"input_{result_id}.{ext}"
        output_key = f"result_{result_id}.jpg"

        # Upload original to 'images' bucket
        original_bytes = io.BytesIO()
        image.save(original_bytes, format=ext.upper() if ext != "jpg" else "JPEG")
        original_bytes.seek(0)
        minio_client.upload_bytes(original_bytes.getvalue(), "images", input_key)

        # Upload result to 'results' bucket
        output_bytes = io.BytesIO()
        Image.fromarray(output_image).save(output_bytes, format="JPEG")
        output_bytes.seek(0)
        minio_client.upload_bytes(output_bytes.getvalue(), "results", output_key)

        # Build response — use localhost for browser access
        minio_host = "localhost"
        image_url = f"http://{minio_host}:9000/results/{output_key}"
        original_url = f"http://{minio_host}:9000/images/{input_key}"

        result = {
            "id": result_id,
            "image_url": image_url,
            "result_image_url": image_url,
            "original_url": original_url,
            "detections": detections,
            "count": len(detections),
            "model_name": model_id,
            "created_at": datetime.now().isoformat(),
        }

        RESULTS_STORE[result_id] = {
            **result,
            "input_key": input_key,
            "output_key": output_key,
        }

        # Cache original image for classification
        IMAGE_CACHE[result_id] = image_np.copy()

        return Response(result)

    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET"])
def list_models(request):
    """List available detection models"""
    import os
    weights_dir = "/weights"
    models_list = [
        {"id": "yolov8n", "name": "YOLOv8n (pretrained)", "description": "Pretrained on COCO", "path": "yolov8n.pt"},
    ]

    # Add trained models
    for name, display_name in [("baseline", "Baseline"), ("mod1", "Mod1"), ("mod2", "Mod2")]:
        pt_path = os.path.join(weights_dir, f"{name}.pt")
        onnx_path = os.path.join(weights_dir, f"{name}.onnx")
        if os.path.exists(pt_path) or os.path.exists(onnx_path):
            models_list.append({
                "id": name,
                "name": f"YOLOv8n {display_name}",
                "description": f"Trained on COCO airplane dataset",
                "path": f"{name}.onnx" if os.path.exists(onnx_path) else f"{name}.pt",
            })

    return Response({"models": models_list})


@api_view(["GET"])
def list_history(request):
    """Get all detection results from history"""
    results = []
    for result_id, data in RESULTS_STORE.items():
        results.append({
            "id": result_id,
            "image_url": data["image_url"],
            "original_url": data.get("original_url", ""),
            "detections": data["detections"],
            "count": data["count"],
            "model_id": data.get("model_name", "yolov8n"),
            "created_at": data.get("created_at", datetime.now().isoformat()),
        })
    # Sort by created_at (newest first)
    results.sort(key=lambda x: x["created_at"], reverse=True)
    return Response({"results": results})


@api_view(["POST"])
@parser_classes([JSONParser])
def run_detection(request):
    """Run detection with selected model on cached image"""
    image_id = request.data.get("image_id")
    model_name = request.data.get("model_name", "yolov8n")
    
    if not image_id:
        return Response({"error": "No image_id provided"}, status=status.HTTP_400_BAD_REQUEST)
    
    image_np = IMAGE_CACHE.get(image_id)
    if image_np is None:
        return Response({"error": "Image not cached. Re-upload."}, status=status.HTTP_400_BAD_REQUEST)
    
    # Run detection with selected model
    detections = detect_airplanes(image_np, model_name)
    
    # Draw detections
    output_image = draw_detections(image_np, detections)
    
    # Save to MinIO
    result_id = str(uuid.uuid4())
    output_key = f"detect_{model_name}_{result_id}.jpg"
    output_bytes = io.BytesIO()
    Image.fromarray(output_image).save(output_bytes, format="JPEG")
    output_bytes.seek(0)
    minio_client.upload_bytes(output_bytes.getvalue(), "results", output_key)

    result = {
        "id": result_id,
        "image_url": f"http://localhost:9000/results/{output_key}",
        "result_image_url": f"http://localhost:9000/results/{output_key}",
        "original_url": f"http://localhost:9000/images/input_{image_id}.jpg",
        "detections": detections,
        "count": len(detections),
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
    }

    # Store in RESULTS_STORE
    RESULTS_STORE[result_id] = {
        **result,
        "output_key": output_key,
    }

    return Response(result)


@api_view(["GET"])
def get_result(request, result_id):
    """Get detection result by ID"""
    result = RESULTS_STORE.get(result_id)
    if not result:
        return Response({"error": "Result not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(result)


@api_view(["DELETE"])
def delete_result(request, result_id):
    """Delete result from MinIO and cache"""
    result = RESULTS_STORE.get(result_id)
    if not result:
        return Response({"error": "Result not found"}, status=status.HTTP_404_NOT_FOUND)

    try:
        # Delete from MinIO
        input_key = result.get("input_key")
        output_key = result.get("output_key")
        
        if input_key:
            minio_client.client.delete_object(Bucket="images", Key=input_key)
            logger.info(f"Deleted from images: {input_key}")
        
        if output_key:
            minio_client.client.delete_object(Bucket="results", Key=output_key)
            logger.info(f"Deleted from results: {output_key}")
        
        # Remove from cache
        del RESULTS_STORE[result_id]
        IMAGE_CACHE.pop(result_id, None)
        
        return Response({"status": "deleted"})
    except Exception as e:
        logger.error(f"Failed to delete result: {e}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET"])
def get_image(request, result_id):
    """Get result image from MinIO"""
    result = RESULTS_STORE.get(result_id)
    if not result:
        return Response({"error": "Result not found"}, status=status.HTTP_404_NOT_FOUND)

    output_key = result.get("output_key")
    if not output_key:
        return Response({"error": "Image not found"}, status=status.HTTP_404_NOT_FOUND)

    try:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=f"http://{settings.MINIO_ENDPOINT}",
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
        )
        response = s3.get_object(Bucket="results", Key=output_key)
        return FileResponse(response["Body"], content_type="image/jpeg")
    except Exception as e:
        logger.error(f"Failed to fetch image from MinIO: {e}")
        return Response({"error": "Image not available"}, status=status.HTTP_404_NOT_FOUND)


# ===== Detect and Classify Pipeline =====

_classifier_cache = {"session": None, "class_names": None}


def _load_classifier():
    """Lazy load classifier ONNX model."""
    if _classifier_cache["session"] is None:
        import onnxruntime as ort
        classifier_path = "/weights/classifier.onnx"
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier not found: {classifier_path}")
        _classifier_cache["session"] = ort.InferenceSession(classifier_path, providers=["CPUExecutionProvider"])
        
        # Load class names from checkpoint
        import torch
        ckpt = torch.load("/weights/classifier.pth", map_location="cpu", weights_only=False)
        _classifier_cache["class_names"] = ckpt["class_names"]
    
    return _classifier_cache["session"], _classifier_cache["class_names"]


def _classify_crop(crop: np.ndarray, top_k: int = 3):
    """Classify cropped airplane image using ONNX. Returns top-K predictions."""
    session, class_names = _load_classifier()
    
    # Preprocess
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(crop_rgb, (224, 224))
    normalized = (resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    
    # Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})
    logits = outputs[0][0]
    
    # Apply softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # Get top-K predictions
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_predictions = []
    for idx in top_indices:
        top_predictions.append({
            "class_id": int(idx),
            "class_name": class_names[idx] if idx < len(class_names) else f"airplane_{idx}",
            "confidence": float(probs[idx]),
        })
    
    return top_predictions


@api_view(["POST"])
@parser_classes([JSONParser, MultiPartParser, FormParser])
def detect_and_classify(request):
    """Detect airplanes and classify their types."""
    # Support JSON (image already uploaded via /api/upload/)
    if request.data.get("image_id"):
        result_id = request.data["image_id"]
        model_name = request.data.get("model_name", "yolov8n")
        
        # Get cached image
        image_np = IMAGE_CACHE.get(result_id)
        if image_np is None:
            return Response({"error": "Original image not cached. Re-upload image."}, status=status.HTTP_400_BAD_REQUEST)

        # Run detection with SELECTED model
        detections = detect_airplanes(image_np, model_name)

        # Создаём новый результат с выбранной моделью
        import copy
        base_result = RESULTS_STORE.get(result_id, {})
        result_with_class = {
            "id": result_id,
            "image_url": base_result.get("image_url", ""),
            "original_url": base_result.get("original_url", ""),
            "detections": detections,
            "count": len(detections),
            "model_name": model_name,
        }
        
        # Classify each detection
        classified_detections = []
        for det in detections:
            try:
                x1, y1, x2, y2 = map(int, det.get("bbox", [0, 0, 0, 0]))
                crop = image_np[max(0,y1):min(image_np.shape[0],y2), max(0,x1):min(image_np.shape[1],x2)]

                if crop.size > 0:
                    top_preds = _classify_crop(crop, top_k=3)
                    det["classification"] = top_preds[0] if top_preds else {"class_name": "unknown", "confidence": 0.0}
                    det["classification_top"] = top_preds
                else:
                    det["classification"] = {"class_name": "unknown", "confidence": 0.0}
                    det["classification_top"] = []
            except Exception as e:
                logger.error(f"Classification error: {e}")
                det["classification"] = {"class_name": "error", "confidence": 0.0}
                det["classification_top"] = []

            classified_detections.append(det)

        logger.info(f"Classified {len(classified_detections)} detections")
        result_with_class["detections"] = classified_detections
        result_with_class["count"] = len(classified_detections)

        # Draw classification on image (без сохранения в RESULTS_STORE)
        if image_np is not None:
            output_image = draw_detections_with_classification(image_np, classified_detections)
            result_id_classify = str(uuid.uuid4())
            output_key = f"result_classified_{result_id_classify}.jpg"
            output_bytes = io.BytesIO()
            Image.fromarray(output_image).save(output_bytes, format="JPEG")
            output_bytes.seek(0)
            minio_client.upload_bytes(output_bytes.getvalue(), "results", output_key)
            result_with_class["result_image_url"] = f"http://localhost:9000/results/{output_key}"
            result_with_class["id"] = result_id_classify
            result_with_class["created_at"] = datetime.now().isoformat()

            # Store in RESULTS_STORE
            RESULTS_STORE[result_id_classify] = {
                **result_with_class,
                "output_key": output_key,
            }
        else:
            result_with_class["result_image_url"] = result_with_class.get("image_url", "")
            result_with_class["created_at"] = datetime.now().isoformat()

        # Возвращаем копию с классификацией, оригинал не тронут
        return Response(result_with_class)
    
    # Multipart fallback - process new image
    if "image" not in request.FILES:
        return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

    image_file = request.FILES["image"]

    try:
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)
        
        # Run detection
        detections = detect_airplanes(image_np)
        
        # Classify each detection
        results_with_classification = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            crop = image_np[max(0,y1):min(image_np.shape[0],y2), max(0,x1):min(image_np.shape[1],x2)]
            
            if crop.size > 0:
                classification = _classify_crop(crop)
                det["classification"] = classification
            else:
                det["classification"] = {"class_name": "unknown", "confidence": 0.0}
            
            results_with_classification.append(det)
        
        # Draw results
        output_image = draw_detections_with_classification(image_np, results_with_classification)
        
        # Upload to MinIO
        result_id = str(uuid.uuid4())
        output_key = f"result_classified_{result_id}.jpg"
        output_bytes = io.BytesIO()
        Image.fromarray(output_image).save(output_bytes, format="JPEG")
        output_bytes.seek(0)
        minio_client.upload_bytes(output_bytes.getvalue(), "results", output_key)
        
        image_url = f"http://localhost:9000/results/{output_key}"
        
        result = {
            "id": result_id,
            "image_url": image_url,
            "detections": results_with_classification,
            "count": len(results_with_classification),
        }
        
        RESULTS_STORE[result_id] = {
            **result,
            "output_key": output_key,
        }
        
        return Response(result)

    except Exception as e:
        logger.error(f"Detect and classify error: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
@parser_classes([JSONParser])
def classify_only(request):
    """Classify the entire image using a classifier model (no detection)."""
    try:
        result_id = request.data.get("image_id")
        model_name = request.data.get("model_name", "classifier_resnet18")

        # Get cached image
        image_np = IMAGE_CACHE.get(result_id)
        if image_np is None:
            return Response({"error": "Original image not cached. Re-upload image."}, status=status.HTTP_400_BAD_REQUEST)

        # Load classifier
        classifier_path = "/weights/classifier.pth"
        if not os.path.exists(classifier_path):
            return Response({"error": f"Classifier model not found at {classifier_path}"}, status=status.HTTP_404_NOT_FOUND)

        # Classify image
        import torch
        from torchvision import transforms
        from torchvision.models import resnet18
        from PIL import Image as PILImage
        
        checkpoint = torch.load(classifier_path, map_location='cpu', weights_only=False)
        
        # Определяем количество классов и имена из checkpoint
        num_classes = checkpoint.get('num_classes', 2)
        class_names = checkpoint.get('class_names', ['airplane', 'not_airplane'])
        
        # Создаём модель и загружаем state_dict
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        # State_dict может быть под разными ключами
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Если checkpoint содержит напрямую state_dict без обёртки
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Preprocess
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(image_np).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probabilities, min(5, len(probabilities)))
        
        # Формируем предсказания
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            class_idx = idx.item()
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            predictions.append({
                "class_name": class_name,
                "confidence": round(prob.item(), 4),
            })
        
        # Создаём результат без bbox (только классификация)
        result_id_classify = str(uuid.uuid4())
        
        # Рисуем результат на изображении
        img_with_result = image_np.copy()
        import cv2
        from PIL import Image as PILImage
        
        # Создаём overlay с результатом
        overlay = img_with_result.copy()
        cv2.rectangle(overlay, (10, 10), (400, 80 + len(predictions) * 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img_with_result, 0.3, 0, img_with_result)
        
        # Рисуем текст
        y_offset = 40
        cv2.putText(img_with_result, "Classification Result:", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        for i, pred in enumerate(predictions):
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            label = f"{pred['class_name']}: {pred['confidence']:.2%}"
            cv2.putText(img_with_result, label, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25
        
        # Сохраняем результат
        output_key = f"result_classified_only_{result_id_classify}.jpg"
        output_bytes = io.BytesIO()
        Image.fromarray(img_with_result).save(output_bytes, format="JPEG")
        output_bytes.seek(0)
        minio_client.upload_bytes(output_bytes.getvalue(), "results", output_key)
        
        # Получаем original_url из RESULTS_STORE (там где загруженное изображение)
        original_result = RESULTS_STORE.get(result_id, {})
        original_url = original_result.get("original_url", original_result.get("image_url", ""))

        result = {
            "id": result_id_classify,
            "image_url": f"http://localhost:9000/results/{output_key}",
            "original_url": original_url,
            "detections": [],  # No detections, only classification
            "count": 0,
            "model_name": model_name,
            "classification": predictions[0] if predictions else None,
            "classification_top": predictions,
            "result_image_url": f"http://localhost:9000/results/{output_key}",
            "created_at": datetime.now().isoformat(),
        }
        
        RESULTS_STORE[result_id_classify] = result
        
        return Response(result)
        
    except Exception as e:
        logger.error(f"Classify only error: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def draw_detections_with_classification(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes with classification labels."""
    import cv2
    img = image.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["bbox"])
        classification = det.get("classification", {})
        cls_name = classification.get("class_name", "unknown")
        cls_conf = classification.get("confidence", 0)
        
        # Color based on classification confidence
        if cls_conf > 0.5:
            color = (0, 255, 0)  # Green - high confidence
        elif cls_conf > 0.2:
            color = (0, 255, 255)  # Yellow - medium
        else:
            color = (0, 165, 255)  # Orange - low confidence
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Classification label with percentage
        label = f"{cls_name} {cls_conf*100:.1f}%"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Background for label
        cv2.rectangle(img, (x1, y1 - 25), (x1 + label_w + 5, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Detection number badge
        badge = f"#{i+1}"
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x2 - bw - 4, y1), (x2, y1 + bh + 4), (255, 255, 255), -1)
        cv2.putText(img, badge, (x2 - bw - 2, y1 + bh + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return img


@api_view(["GET"])
def get_mlflow_metrics(request):
    """Get MLflow experiment metrics for dashboard"""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://mlflow:5000")

        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        runs_data = []

        for exp in experiments:
            if "yolov8" in exp.name.lower() or "aircraft" in exp.name.lower():
                runs = client.search_runs(exp.experiment_id)
                for run in runs:
                    metrics = []
                    for key, value in run.data.metrics.items():
                        metrics.append({
                            "id": f"{run.info.run_id}_{key}",
                            "name": key,
                            "value": value,
                            "step": 0,
                            "timestamp": run.info.start_time,
                            "model_name": run.data.params.get("model_type", "unknown"),
                        })

                    runs_data.append({
                        "id": run.info.run_id,
                        "name": run.info.run_name or "unknown",
                        "status": run.info.status,
                        "metrics": metrics,
                        "start_time": run.info.start_time,
                        "end_time": run.info.end_time or run.info.start_time,
                    })

        return Response(runs_data)
    except Exception as e:
        logger.error(f"MLflow metrics error: {e}")
        return Response([], status=200)


@api_view(["GET"])
def list_mlflow_models(request):
    """Get list of available models from MLflow artifacts"""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://mlflow:5000")

        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        models_list = []

        for exp in experiments:
            if "yolov8" in exp.name.lower() or "aircraft" in exp.name.lower() or "classifier" in exp.name.lower():
                runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])

                for run in runs:
                    # Skip non-FINISHED runs
                    if run.info.status != "FINISHED":
                        continue

                    # Проверяем наличие модели через параметры run, а не артефакты (S3 может быть недоступен)
                    model_type = run.data.params.get("model_type", "unknown")
                    
                    # Пропускаем runs без model_type
                    if model_type == "unknown":
                        continue

                    epochs = run.data.params.get("epochs", "?")

                    # Определяем тип модели: detector или classifier
                    is_classifier = "resnet" in model_type.lower() or "classifier" in (run.info.run_name or "").lower()

                    if is_classifier:
                        # Метрики классификатора
                        accuracy = run.data.metrics.get("best/val_accuracy", 0)
                        top5_accuracy = run.data.metrics.get("final/val_top5_accuracy", 0)
                    else:
                        # Метрики детектора (YOLO)
                        accuracy = run.data.metrics.get("final/mAP50", 0)
                        top5_accuracy = run.data.metrics.get("final/mAP50-95", 0)

                    # Нормализуем к 0-1 (метрики могут быть уже в процентах)
                    if accuracy > 1:
                        accuracy = accuracy / 100
                    if top5_accuracy > 1:
                        top5_accuracy = top5_accuracy / 100

                    primary_metric = accuracy  # Для сортировки
                    
                    # Определяем наличие файлов через проверку локальных весов
                    has_pt = True  # Предполагаем, что модель доступна
                    has_onnx = False  # Будет проверено при скачивании

                    models_list.append({
                        "run_id": run.info.run_id,
                        "run_name": run.info.run_name or "unknown",
                        "model_type": model_type,
                        "epochs": epochs,
                        "mAP50": round(primary_metric, 4) if primary_metric else 0,
                        "accuracy": round(accuracy, 4) if accuracy else 0,
                        "top5_accuracy": round(top5_accuracy, 4) if top5_accuracy else 0,
                        "is_classifier": is_classifier,
                        "status": run.info.status,
                        "created_at": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                        "has_pt": has_pt,
                        "has_onnx": has_onnx,
                    })

        # Sort by primary metric descending
        models_list.sort(key=lambda x: x["mAP50"], reverse=True)

        return Response({"models": models_list})
    except Exception as e:
        logger.error(f"MLflow models list error: {e}")
        import traceback
        traceback.print_exc()
        return Response({"models": [], "error": str(e)}, status=200)


@api_view(["POST"])
@parser_classes([JSONParser])
def download_mlflow_model(request):
    """Download model from MLflow and save to /weights/"""
    import os
    run_id = request.data.get("run_id")
    model_name = request.data.get("model_name", "mlflow_model")

    if not run_id:
        return Response({"error": "No run_id provided"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        weights_dir = "/weights"

        # Check if model already exists locally
        existing_pt = os.path.join(weights_dir, f"{model_name}.pt")
        existing_onnx = os.path.join(weights_dir, f"{model_name}.onnx")

        if os.path.exists(existing_pt):
            return Response({
                "status": "already_exists",
                "model_name": model_name,
                "path": existing_pt,
                "format": "pt",
            })
        if os.path.exists(existing_onnx):
            return Response({
                "status": "already_exists",
                "model_name": model_name,
                "path": existing_onnx,
                "format": "onnx",
            })

        import mlflow
        mlflow.set_tracking_uri("http://mlflow:5000")

        client = mlflow.tracking.MlflowClient()

        # Find .pt or .onnx artifact
        artifacts = client.list_artifacts(run_id)

        pt_artifact = None
        onnx_artifact = None

        for artifact in artifacts:
            if artifact.path.endswith('best.pt') or artifact.path.endswith('last.pt'):
                pt_artifact = artifact.path
            elif 'onnx' in artifact.path.lower() and artifact.path.endswith('.onnx'):
                onnx_artifact = artifact.path

        if not pt_artifact and not onnx_artifact:
            return Response({"error": "No model artifacts found"}, status=status.HTTP_404_NOT_FOUND)

        # Download preferred format (ONNX > PT)
        artifact_path = onnx_artifact if onnx_artifact else pt_artifact
        extension = ".onnx" if onnx_artifact else ".pt"

        # Download artifact
        local_path = client.download_artifacts(run_id, artifact_path)

        # Copy to /weights/
        import shutil
        dest_path = f"/weights/{model_name}{extension}"
        shutil.copy2(local_path, dest_path)

        logger.info(f"Downloaded model from MLflow: {dest_path}")

        return Response({
            "status": "success",
            "model_name": model_name,
            "path": dest_path,
            "format": extension[1:],
        })

    except Exception as e:
        logger.error(f"Download model error: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
