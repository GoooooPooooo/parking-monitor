"""
Измерение FPS для всех моделей (детекторы YOLO и классификатор).
"""

import time
import os
import numpy as np
from pathlib import Path
import json
import yaml
import onnxruntime as ort
import argparse


def benchmark_onnx_detector(model_path: str, model_name: str, num_runs: int = 100):
    """Измерить FPS для ONNX детектора YOLO"""
    print(f"📦 Загрузка детектора: {model_name}")

    # Создать ONNX Runtime сессию
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Получить размер входа
    input_name = session.get_inputs()[0].name

    # Создать случайное изображение 320x320
    img_size = 320
    dummy_image = np.random.rand(1, 3, img_size, img_size).astype(np.float32)

    print(f"🔥 Warmup (10 итераций)...")
    for _ in range(10):
        session.run(None, {input_name: dummy_image})

    print(f"⏱️  Benchmark ({num_runs} итераций)...")
    start = time.time()
    for _ in range(num_runs):
        outputs = session.run(None, {input_name: dummy_image})
    elapsed = time.time() - start

    fps = num_runs / elapsed
    avg_time = elapsed / num_runs * 1000  # ms

    # Получить размер модели
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)

    print(f"✅ FPS: {fps:.2f}")
    print(f"   Среднее время: {avg_time:.2f} ms")
    print(f"   Размер модели: {size_mb:.2f} MB")
    print()

    return {
        "model": model_name,
        "type": "detector",
        "fps": fps,
        "avg_time_ms": avg_time,
        "size_mb": size_mb
    }


def benchmark_onnx_classifier(model_path: str, model_name: str, num_runs: int = 100):
    """Измерить FPS для ONNX классификатора"""
    print(f"📦 Загрузка классификатора: {model_name}")

    # Создать ONNX Runtime сессию
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Получить размер входа
    input_name = session.get_inputs()[0].name

    # Создать случайное изображение 224x224
    img_size = 224
    dummy_image = np.random.rand(1, 3, img_size, img_size).astype(np.float32)

    print(f"🔥 Warmup (10 итераций)...")
    for _ in range(10):
        session.run(None, {input_name: dummy_image})

    print(f"⏱️  Benchmark ({num_runs} итераций)...")
    start = time.time()
    for _ in range(num_runs):
        outputs = session.run(None, {input_name: dummy_image})
    elapsed = time.time() - start

    fps = num_runs / elapsed
    avg_time = elapsed / num_runs * 1000  # ms

    # Получить размер модели
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)

    print(f"✅ FPS: {fps:.2f}")
    print(f"   Среднее время: {avg_time:.2f} ms")
    print(f"   Размер модели: {size_mb:.2f} MB")
    print()

    return {
        "model": model_name,
        "type": "classifier",
        "fps": fps,
        "avg_time_ms": avg_time,
        "size_mb": size_mb
    }


def find_all_models(mlruns_dir: Path, filter_pattern: str = None):
    """Автоматически найти все ONNX модели в MLflow артефактах"""
    models = {
        "detectors": [],
        "classifiers": []
    }

    # Пройтись по всем экспериментам
    for exp_dir in mlruns_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in ["0", "models", ".trash"]:
            continue

        # Пройтись по всем runs в эксперименте
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Проверить наличие meta.yaml
            meta_file = run_dir / "meta.yaml"
            if not meta_file.exists():
                continue

            # Прочитать метаданные
            with open(meta_file, 'r') as f:
                meta = yaml.safe_load(f)

            run_name = meta.get('run_name', 'unknown')
            run_id = meta.get('run_id', run_dir.name)

            # Поиск ONNX моделей в артефактах
            onnx_dir = run_dir / "artifacts" / "onnx"
            if not onnx_dir.exists():
                continue

            # Детекторы YOLO
            yolo_models = list(onnx_dir.glob("yolov8n_*.onnx"))
            for model_path in yolo_models:
                # Попытаться определить разрешение из параметров
                params_dir = run_dir / "params"
                imgsz = "?"
                if params_dir.exists():
                    imgsz_file = params_dir / "imgsz"
                    if imgsz_file.exists():
                        with open(imgsz_file, 'r') as f:
                            imgsz = f.read().strip()

                model_display_name = f"{run_name} ({imgsz}px)"

                # Применить фильтр если указан (проверяем и run_name и imgsz)
                if filter_pattern:
                    filter_lower = filter_pattern.lower()
                    if (filter_lower not in run_name.lower() and
                        filter_lower not in imgsz.lower() and
                        filter_lower not in model_display_name.lower()):
                        continue

                models["detectors"].append({
                    "name": model_display_name,
                    "path": str(model_path),
                    "run_id": run_id,
                    "run_name": run_name,
                    "imgsz": imgsz
                })

            # Классификаторы
            classifier_models = list(onnx_dir.glob("classifier.onnx"))
            for model_path in classifier_models:
                # Применить фильтр если указан
                if filter_pattern:
                    filter_lower = filter_pattern.lower()
                    if filter_lower not in run_name.lower():
                        continue

                models["classifiers"].append({
                    "name": run_name,
                    "path": str(model_path),
                    "run_id": run_id,
                    "run_name": run_name
                })

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Бенчмарк FPS для ONNX моделей из MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python benchmark_fps.py                    # Все модели
  python benchmark_fps.py --filter mod1      # Только mod1
  python benchmark_fps.py --filter 320       # Только 320px
  python benchmark_fps.py --filter classifier # Только классификаторы
  python benchmark_fps.py --runs 200         # 200 итераций
        """
    )
    parser.add_argument(
        "--mlruns-dir",
        type=str,
        default="/mlruns",
        help="Путь к директории mlruns (по умолчанию: /mlruns)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Фильтр по имени модели (например: mod1, 320, classifier)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Количество итераций для бенчмарка (по умолчанию: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/ml/fps_benchmark.json",
        help="Путь для сохранения результатов JSON"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("BENCHMARK FPS ДЛЯ ВСЕХ МОДЕЛЕЙ")
    print("=" * 80)
    print()

    if args.filter:
        print(f"🔍 Фильтр: {args.filter}")
        print()

    # Найти модели
    mlruns_path = Path(args.mlruns_dir)
    if not mlruns_path.exists():
        print(f"❌ Директория {args.mlruns_dir} не найдена")
        return

    models = find_all_models(mlruns_path, filter_pattern=args.filter)

    if not models["detectors"] and not models["classifiers"]:
        print(f"❌ ONNX модели не найдены в {args.mlruns_dir}")
        if args.filter:
            print(f"   Попробуйте без фильтра или измените фильтр: {args.filter}")
        return

    results = []

    # Бенчмарк детекторов
    if models["detectors"]:
        print("=" * 80)
        print(f"ДЕТЕКТОРЫ YOLO (найдено: {len(models['detectors'])})")
        print("=" * 80)
        print()

        for model in models["detectors"]:
            result = benchmark_onnx_detector(model["path"], model["name"], num_runs=args.runs)
            if result:
                result["run_id"] = model["run_id"]
                result["imgsz"] = model["imgsz"]
                results.append(result)

    # Бенчмарк классификаторов
    if models["classifiers"]:
        print("=" * 80)
        print(f"КЛАССИФИКАТОРЫ (найдено: {len(models['classifiers'])})")
        print("=" * 80)
        print()

        # Группируем одинаковые классификаторы
        unique_classifiers = {}
        for model in models["classifiers"]:
            if model["run_name"] not in unique_classifiers:
                unique_classifiers[model["run_name"]] = model

        for model in unique_classifiers.values():
            result = benchmark_onnx_classifier(model["path"], model["run_name"], num_runs=args.runs)
            if result:
                result["run_id"] = model["run_id"]
                results.append(result)

    # Вывод сводной таблицы
    if results:
        print("=" * 80)
        print("СВОДНАЯ ТАБЛИЦА FPS")
        print("=" * 80)
        print()

        # Детекторы
        detector_results = [r for r in results if r["type"] == "detector"]
        if detector_results:
            print("ДЕТЕКТОРЫ:")
            print(f"{'Модель':<30} {'FPS':>10} {'Время (ms)':>15} {'Размер (MB)':>15}")
            print("-" * 80)
            for r in detector_results:
                print(f"{r['model']:<30} {r['fps']:>10.2f} {r['avg_time_ms']:>15.2f} {r['size_mb']:>15.2f}")
            print()

        # Классификаторы
        classifier_results = [r for r in results if r["type"] == "classifier"]
        if classifier_results:
            print("КЛАССИФИКАТОРЫ:")
            print(f"{'Модель':<30} {'FPS':>10} {'Время (ms)':>15} {'Размер (MB)':>15}")
            print("-" * 80)
            for r in classifier_results:
                print(f"{r['model']:<30} {r['fps']:>10.2f} {r['avg_time_ms']:>15.2f} {r['size_mb']:>15.2f}")
            print()

        # Сохранить результаты
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Результаты сохранены: {output_path}")


if __name__ == "__main__":
    main()
