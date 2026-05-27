"""
Тестирование ONNX Runtime детектора.
Демонстрирует работу с ONNX Runtime для детекции самолётов.
"""

import sys
sys.path.insert(0, '/app')

import cv2
from pathlib import Path
from detector.detector import YOLODetector


def main():
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ONNX RUNTIME ДЕТЕКТОРА")
    print("=" * 70)
    print()
    
    # Загрузить модель
    model_path = Path("/weights/baseline.onnx")
    
    if not model_path.exists():
        print(f"❌ Модель не найдена: {model_path}")
        print("   Сначала экспортируйте модель: make export")
        return
    
    print(f"📦 Загрузка модели: {model_path}")
    detector = YOLODetector(model_path, conf_threshold=0.25)
    print(f"✅ Модель загружена")
    print(f"   Input shape: {detector.input_width}x{detector.input_height}")
    print(f"   Providers: {detector.session.get_providers()}")
    print()
    
    # Загрузить изображение
    image_path = "/examples/airplane_demo.jpg"
    
    if not Path(image_path).exists():
        print(f"❌ Изображение не найдено: {image_path}")
        return
    
    print(f"🖼️  Загрузка изображения: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Не удалось загрузить изображение")
        return
    
    print(f"✅ Изображение загружено: {image.shape}")
    print()
    
    # Детекция
    print("🔍 Запуск детекции...")
    detections = detector.detect(image)
    
    print(f"✅ Детекция завершена")
    print(f"   Найдено объектов: {len(detections)}")
    print()
    
    if len(detections) > 0:
        print("📊 Результаты детекции:")
        print("-" * 70)
        for i, det in enumerate(detections, 1):
            bbox = det['bbox']
            conf = det['confidence']
            cls = det['class_name']
            print(f"  {i}. {cls}")
            print(f"     Confidence: {conf:.3f}")
            print(f"     BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            print()
        
        # Отрисовка
        print("🎨 Отрисовка результатов...")
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{conf:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Сохранить результат
        output_path = "/examples/output_onnx_test.jpg"
        cv2.imwrite(output_path, image)
        print(f"✅ Результат сохранён: {output_path}")
    else:
        print("⚠️  Объекты не найдены")
    
    print()
    print("=" * 70)
    print("✅ ТЕСТ ЗАВЕРШЁН УСПЕШНО!")
    print("=" * 70)


if __name__ == "__main__":
    main()
