"""
Генерация отчёта по результатам бенчмарка с использованием LLM.
"""

import json
import sys
from pathlib import Path


def load_benchmark_results(json_path: str):
    """Загрузить результаты бенчмарка из JSON"""
    with open(json_path, "r") as f:
        return json.load(f)


def format_results_for_llm(results):
    """Форматировать результаты для передачи в LLM"""

    # Группировка по типам
    detectors = [r for r in results if r["type"] == "detector"]
    classifiers = [r for r in results if r["type"] == "classifier"]

    report = []
    report.append("# РЕЗУЛЬТАТЫ БЕНЧМАРКА ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ")
    report.append("")
    report.append("## 1. ДЕТЕКТОРЫ YOLO")
    report.append("")

    if detectors:
        report.append("| Модель | FPS | Время (ms) | Размер (MB) | Run ID |")
        report.append("|--------|-----|------------|-------------|--------|")
        for d in detectors:
            report.append(
                f"| {d['model']} | {d['fps']:.2f} | {d['avg_time_ms']:.2f} | {d['size_mb']:.2f} | {d['run_id'][:8]}... |"
            )
        report.append("")

        # Статистика
        report.append("### Статистика детекторов:")
        report.append(f"- Всего моделей: {len(detectors)}")
        report.append(f"- Максимальный FPS: {max(d['fps'] for d in detectors):.2f}")
        report.append(f"- Минимальный FPS: {min(d['fps'] for d in detectors):.2f}")
        report.append(
            f"- Средний FPS: {sum(d['fps'] for d in detectors) / len(detectors):.2f}"
        )
        report.append("")
    else:
        report.append("Детекторы не найдены.")
        report.append("")

    report.append("## 2. КЛАССИФИКАТОРЫ")
    report.append("")

    if classifiers:
        report.append("| Модель | FPS | Время (ms) | Размер (MB) | Run ID |")
        report.append("|--------|-----|------------|-------------|--------|")
        for c in classifiers:
            report.append(
                f"| {c['model']} | {c['fps']:.2f} | {c['avg_time_ms']:.2f} | {c['size_mb']:.2f} | {c['run_id'][:8]}... |"
            )
        report.append("")

        # Статистика
        report.append("### Статистика классификаторов:")
        report.append(f"- Всего моделей: {len(classifiers)}")
        report.append(f"- FPS: {classifiers[0]['fps']:.2f}")
        report.append(f"- Время инференса: {classifiers[0]['avg_time_ms']:.2f} ms")
        report.append("")
    else:
        report.append("Классификаторы не найдены.")
        report.append("")

    return "\n".join(report)


def generate_analysis_prompt(formatted_results):
    """Создать промпт для LLM анализа"""

    prompt = f"""Проанализируй результаты бенчмарка производительности моделей детекции и классификации самолётов.

{formatted_results}

## ЗАДАЧА:
Создай краткий аналитический отчёт на русском языке, который включает:

1. **Сравнение детекторов YOLO**:
   - Какая модель самая быстрая и почему
   - Соотношение размера модели и скорости
   - Рекомендации по выбору модели для разных сценариев

2. **Анализ классификатора**:
   - Оценка производительности
   - Сравнение с детекторами

3. **Общие выводы**:
   - Лучшая модель для production
   - Лучшая модель для edge устройств
   - Компромисс скорость/размер/качество

Формат ответа: структурированный markdown отчёт с заголовками и списками.
"""

    return prompt


def main():
    # ANSI цвета для вывода
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    # Путь к результатам бенчмарка
    json_path = Path("/ml/fps_benchmark.json")

    if not json_path.exists():
        print("❌ Файл fps_benchmark.json не найден")
        print("   Сначала запустите: make benchmark")
        sys.exit(1)

    # Загрузить результаты
    results = load_benchmark_results(json_path)

    if not results:
        print("❌ Результаты бенчмарка пусты")
        sys.exit(1)

    # Форматировать результаты
    formatted_results = format_results_for_llm(results)

    # Вывести форматированные результаты
    print("=" * 80)
    print("ФОРМАТИРОВАННЫЕ РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    print("=" * 80)
    print()
    print(formatted_results)
    print()

    # Создать промпт для LLM
    prompt = generate_analysis_prompt(formatted_results)

    # Сохранить результаты в виде текста
    prompt_path = Path("/ml/benchmark_analysis.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    print("=" * 80)
    print("ФАЙЛЫ СОХРАНЕНЫ")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
