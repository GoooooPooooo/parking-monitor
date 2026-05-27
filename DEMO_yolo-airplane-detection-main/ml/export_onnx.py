"""
Экспорт всех обученных моделей в формат ONNX.

Модели:
- baseline.pt → baseline.onnx
- mod1.pt → mod1.onnx
- mod2.pt → mod2.onnx
- classifier.pt → classifier.onnx
"""

import argparse
import os

from ultralytics import YOLO


def export_model(pt_path: str, onnx_path: str, imgsz: int = 640):
    """Экспорт одной модели в ONNX."""
    if not os.path.exists(pt_path):
        print(f"Warning: {pt_path} not found, skipping.")
        return False

    print(f"Exporting {pt_path} → {onnx_path}")
    model = YOLO(pt_path)

    model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=12,
        dynamic=False,
    )

    expected_onnx = pt_path.replace(".pt", ".onnx")
    if os.path.exists(expected_onnx):
        os.rename(expected_onnx, onnx_path)
        print(f"Success: {onnx_path}")
        return True
    else:
        print(f"Warning: ONNX file not found at {expected_onnx}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--weights-dir", type=str, default="../weights")
    parser.add_argument("--output-dir", type=str, default="../weights")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    models = [
        ("baseline.pt", "baseline.onnx", 320),
        ("mod1.pt", "mod1.onnx", 320),
        ("mod2.pt", "mod2.onnx", 320),
    ]

    for pt, onnx, imgsz in models:
        pt_path = os.path.join(args.weights_dir, pt)
        onnx_path = os.path.join(args.output_dir, onnx)
        export_model(pt_path, onnx_path, imgsz)

    print("\nExport complete!")


if __name__ == "__main__":
    main()
