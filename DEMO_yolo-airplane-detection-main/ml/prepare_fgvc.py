"""
Подготовка датасета FGVC Aircraft для классификации.

Создаёт структуру директорий для PyTorch ImageFolder:
  fgvc_aircraft_imagefolder/
    train/
      class_01/
        image1.jpg
      class_02/
    val/
"""

import os
import shutil
from pathlib import Path


def prepare_fgvc(src_dir: str, output_dir: str):
    """Подготовить FGVC Aircraft датасет."""
    src = Path(src_dir)
    images_dir = src / "data" / "images"
    
    # Читать train/val списки
    with open(src / "data" / "images_variant_train.txt") as f:
        train_list = [line.strip() for line in f]
    with open(src / "data" / "images_variant_val.txt") as f:
        val_list = [line.strip() for line in f]
    
    # Читать mapping image -> variant
    variant_map = {}
    with open(src / "data" / "images_variant_trainval.txt") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                variant_map[parts[0]] = parts[1]
    
    # Создать output директории
    out = Path(output_dir)
    for split in ["train", "val"]:
        (out / split).mkdir(parents=True, exist_ok=True)
    
    processed = {"train": 0, "val": 0}
    
    for img_name, split_list in [("train", train_list), ("val", val_list)]:
        for item in split_list:
            parts = item.split(" ", 1)
            if len(parts) != 2:
                continue
            img_id, variant = parts[0], parts[1]
            
            src_img = images_dir / f"{img_id}.jpg"
            if not src_img.exists():
                continue
            
            # Создать директорию для класса
            class_dir = out / img_name / variant
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Копировать изображение
            dst = class_dir / f"{img_id}.jpg"
            shutil.copy2(src_img, dst)
            processed[img_name] += 1
    
    print(f"Train images: {processed['train']}")
    print(f"Val images: {processed['val']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    prepare_fgvc(
        src_dir="/ml/data/fgvc-aircraft-2013b",
        output_dir="/ml/data/fgvc_aircraft_imagefolder",
    )
