"""
Сбор метрик из MLflow для всех моделей.
"""

import mlflow
import pandas as pd
from pathlib import Path


def collect_metrics():
    """Собрать метрики из MLflow для всех экспериментов"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = mlflow.tracking.MlflowClient()
    
    print("=" * 70)
    print("СБОР МЕТРИК ИЗ MLFLOW")
    print("=" * 70)
    print()
    
    experiments = client.search_experiments()
    data = []
    
    for exp in experiments:
        if "yolov8" in exp.name.lower():
            print(f"📊 Эксперимент: {exp.name}")
            runs = client.search_runs(exp.experiment_id)
            
            for run in runs:
                model_type = run.data.params.get("model_type", "unknown")
                modification = run.data.params.get("modification", "baseline")
                
                metrics = {
                    "model": model_type,
                    "modification": modification,
                    "mAP50": run.data.metrics.get("mAP50", 0),
                    "mAP50-95": run.data.metrics.get("mAP50-95", 0),
                    "precision": run.data.metrics.get("precision", 0),
                    "recall": run.data.metrics.get("recall", 0),
                    "run_id": run.info.run_id,
                    "start_time": run.info.start_time,
                }
                
                data.append(metrics)
                print(f"  ✓ {model_type}: mAP50={metrics['mAP50']:.3f}")
    
    print()
    
    if not data:
        print("⚠️  Метрики не найдены. Убедитесь, что модели обучены.")
        return None
    
    df = pd.DataFrame(data)
    df = df.sort_values('start_time')
    
    print("=" * 70)
    print("СВОДНАЯ ТАБЛИЦА МЕТРИК")
    print("=" * 70)
    print()
    print(df.to_string(index=False))
    print()
    
    # Сохранить в CSV
    output_path = Path("/ml/metrics_summary.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Метрики сохранены: {output_path}")
    
    # Сохранить в Markdown
    md_path = Path("/ml/metrics_summary.md")
    with open(md_path, 'w') as f:
        f.write("# Сводная таблица метрик\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"✅ Markdown таблица: {md_path}")
    
    return df


if __name__ == "__main__":
    collect_metrics()
