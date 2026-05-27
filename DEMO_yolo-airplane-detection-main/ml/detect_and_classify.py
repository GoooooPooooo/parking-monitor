"""
Пайплайн: детекция + классификация самолётов.

1. Детекция через YOLO ONNX → bounding box
2. Crop объекта
3. Классификация через ResNet ONNX → тип самолёта
4. Отрисовка рамки + confidence + имя класса
5. Трассировка в MLflow
"""

import argparse
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml


class DetectionPipeline:
    """Пайплайн детекции и классификации."""
    
    def __init__(self, detector_path: str, classifier_path: str):
        self.detector = ort.InferenceSession(detector_path, providers=["CPUExecutionProvider"])
        self.detector_input = self.detector.get_inputs()[0].name
        
        classifier = ort.InferenceSession(classifier_path, providers=["CPUExecutionProvider"])
        self.classifier_session = classifier
        self.classifier_input = classifier.get_inputs()[0].name
        
        self.imgsz = 320
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
    
    def detect(self, image: np.ndarray) -> list:
        """Детекция объектов."""
        h, w = image.shape[:2]
        blob = cv2.resize(image, (self.imgsz, self.imgsz))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        
        outputs = self.detector.run(None, {self.detector_input: blob})
        output = outputs[0][0].T
        
        detections = []
        for row in output:
            scores = row[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > self.conf_threshold:
                cx, cy, bw, bh = row[:4]
                x1 = int((cx - bw / 2) * w / self.imgsz)
                y1 = int((cy - bh / 2) * h / self.imgsz)
                x2 = int((cx + bw / 2) * w / self.imgsz)
                y2 = int((cy + bh / 2) * h / self.imgsz)
                
                detections.append({
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "bbox": [x1, y1, x2, y2],
                })
        
        # NMS
        if detections:
            boxes = [[d["bbox"][0], d["bbox"][1], d["bbox"][2]-d["bbox"][0], d["bbox"][3]-d["bbox"][1]] for d in detections]
            scores = [d["confidence"] for d in detections]
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                detections = [detections[i] for i in indices]
        
        return detections
    
    def classify(self, crop: np.ndarray) -> dict:
        """Классификация cropped изображения."""
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        batch = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        
        outputs = self.classifier_session.run(None, {self.classifier_input: batch})
        probs = outputs[0][0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        
        return {"class_id": class_id, "confidence": confidence}
    
    def process(self, image: np.ndarray) -> dict:
        """Полный пайплайн: detect + classify."""
        start_time = time.time()
        
        # Step 1: Detect
        detections = self.detect(image)
        detect_time = time.time() - start_time
        
        results = []
        classify_times = []
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = image[max(0,y1):min(image.shape[0],y2), max(0,x1):min(image.shape[1],x2)]
            
            if crop.size > 0:
                classify_start = time.time()
                classification = self.classify(crop)
                classify_times.append(time.time() - classify_start)
                det["classification"] = classification
            
            results.append(det)
        
        total_time = time.time() - start_time
        
        return {
            "detections": results,
            "count": len(results),
            "timing": {
                "detection": detect_time,
                "classification_avg": np.mean(classify_times) if classify_times else 0,
                "total": total_time,
            }
        }
    
    def draw_results(self, image: np.ndarray, results: dict) -> np.ndarray:
        """Отрисовка результатов."""
        img = image.copy()
        
        for det in results["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            
            class_name = "airplane"
            if "classification" in det:
                class_name = f"airplane_{det['classification']['class_id']}"
                conf = det["classification"]["confidence"]
            
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + lw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img


def main():
    parser = argparse.ArgumentParser(description="Detect + Classify pipeline")
    parser.add_argument("--detector", type=str, default="/weights/baseline.onnx")
    parser.add_argument("--classifier", type=str, default="/weights/classifier.onnx")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="/examples")
    args = parser.parse_args()
    
    pipeline = DetectionPipeline(args.detector, args.classifier)
    
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isfile(args.source):
        sources = [args.source]
    else:
        sources = [os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith((".jpg", ".png"))]
    
    for src in sources:
        image = cv2.imread(src)
        if image is None:
            continue
        
        results = pipeline.process(image)
        print(f"{src}: {results['count']} detections, {results['timing']['total']:.3f}s")
        
        output_img = pipeline.draw_results(image, results)
        filename = Path(src).stem + "_result.jpg"
        cv2.imwrite(os.path.join(args.output, filename), output_img)
        print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
