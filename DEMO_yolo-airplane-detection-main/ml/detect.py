"""
Инференс детектора на тестовых изображениях.

Использует ONNX модель для детекции самолётов.
Отрисовывает bounding box и confidence на изображении.

Пример:
    python detect.py --model ../weights/baseline.onnx --source test.jpg --output results/
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class YOLODetector:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.class_names = ["airplane"]

    def preprocess(self, image: np.ndarray):
        h, w = image.shape[:2]
        self.original_shape = (h, w)
        input_size = self.input_shape[2]
        if isinstance(input_size, str):
            input_size = 640

        blob = cv2.resize(image, (input_size, input_size))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob

    def postprocess(self, outputs: list):
        input_size = self.input_shape[2]
        if isinstance(input_size, str):
            input_size = 640

        detections = []
        output = outputs[0]

        if len(output.shape) == 3:
            output = output[0].T
        else:
            output = output.T

        for row in output:
            scores = row[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > self.conf_threshold:
                cx, cy, bw, bh = row[:4]
                x1 = int((cx - bw / 2) * self.original_shape[1] / input_size)
                y1 = int((cy - bh / 2) * self.original_shape[0] / input_size)
                x2 = int((cx + bw / 2) * self.original_shape[1] / input_size)
                y2 = int((cy + bh / 2) * self.original_shape[0] / input_size)

                detections.append(
                    {
                        "class_id": int(class_id),
                        "class_name": self.class_names[class_id],
                        "confidence": float(confidence),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return detections

    def nms(self, detections: list):
        if not detections:
            return []

        boxes = [
            [
                d["bbox"][0],
                d["bbox"][1],
                d["bbox"][2] - d["bbox"][0],
                d["bbox"][3] - d["bbox"][1],
            ]
            for d in detections
        ]
        scores = [d["confidence"] for d in detections]

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.iou_threshold
        )

        if len(indices) == 0:
            return []
        if isinstance(indices, tuple):
            indices = indices[0]

        return [detections[i] for i in indices]

    def detect(self, image: np.ndarray):
        blob = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        detections = self.postprocess(outputs)
        return self.nms(detections)

    def draw_detections(self, image: np.ndarray, detections: list):
        result = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            class_name = det.get("class_name", "airplane")

            color = (0, 255, 0)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(result, (x1, y1 - 20), (x1 + label_w, y1), color, -1)
            cv2.putText(
                result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        return result


def main():
    parser = argparse.ArgumentParser(description="YOLO ONNX inference")
    parser.add_argument("--model", type=str, default="../weights/baseline.onnx")
    parser.add_argument("--source", type=str, required=True, help="Image or directory")
    parser.add_argument("--output", type=str, default="../examples")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    detector = YOLODetector(args.model)
    detector.conf_threshold = args.conf
    detector.iou_threshold = args.iou

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.source):
        sources = [args.source]
    else:
        sources = [
            os.path.join(args.source, f)
            for f in os.listdir(args.source)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    for src in sources:
        image = cv2.imread(src)
        if image is None:
            print(f"Failed to read {src}")
            continue

        detections = detector.detect(image)
        print(f"{src}: {len(detections)} detections")

        result = detector.draw_detections(image, detections)

        filename = Path(src).stem + "_result.jpg"
        output_path = os.path.join(args.output, filename)
        cv2.imwrite(output_path, result)
        print(f"Saved: {output_path}")

        for det in detections:
            print(f"  {det['class_name']}: {det['confidence']:.3f} at {det['bbox']}")


if __name__ == "__main__":
    main()
