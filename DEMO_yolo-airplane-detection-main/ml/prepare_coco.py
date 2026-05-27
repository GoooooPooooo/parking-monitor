"""
Подготовка датасета COCO airplane для обучения YOLOv8.

Скрипт:
1. Скачивает COCO 2017 dataset (train + val)
2. Фильтрует только класс airplane (category_id=5)
3. Конвертирует аннотации в формат YOLO
4. Создаёт структуру папок для ultralytics
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

from pycocotools.coco import COCO
from tqdm import tqdm

COCO_AIRPLANE_ID = 5


def _verify_zip(path: str) -> bool:
    """Проверить что файл валидный zip."""
    try:
        with zipfile.ZipFile(path, "r") as z:
            z.testzip()
        return True
    except Exception:
        return False


def download_coco(data_dir: str):
    """Распаковать COCO 2017 dataset если ещё не распакован.
    
    Примечание: скачивание выполняется на хосте через Makefile.
    Эта функция только распаковывает уже скачанные архивы.
    """
    coco_dir = os.path.join(data_dir, "coco")
    train_images = os.path.join(coco_dir, "train2017")
    val_images = os.path.join(coco_dir, "val2017")
    annotations_dir = os.path.join(coco_dir, "annotations")

    # Проверить что уже есть
    if (os.path.exists(os.path.join(train_images, "000000000009.jpg")) and
        os.path.exists(os.path.join(val_images, "000000000139.jpg")) and
        os.path.exists(os.path.join(annotations_dir, "instances_val2017.json"))):
        print("✓ COCO dataset уже распакован.")
        return coco_dir

    os.makedirs(coco_dir, exist_ok=True)

    # Распаковать
    import zipfile
    for zip_path in [os.path.join(coco_dir, "train2017.zip"), os.path.join(coco_dir, "val2017.zip")]:
        if not os.path.exists(zip_path):
            continue
        if not _verify_zip(zip_path):
            print(f"⚠ Повреждённый архив после скачивания: {os.path.basename(zip_path)}")
            os.remove(zip_path)
            continue
        print(f"Распаковка: {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(coco_dir)

    ann_zip = os.path.join(coco_dir, "annotations_trainval2017.zip")
    if os.path.exists(ann_zip) and _verify_zip(ann_zip):
        print("Распаковка аннотаций...")
        with zipfile.ZipFile(ann_zip, "r") as z:
            z.extractall(coco_dir)

    # Финальная проверка
    if not os.path.exists(os.path.join(train_images, "000000000009.jpg")):
        # Может быть в поддиректории
        subdir = os.path.join(train_images, "train2017")
        if os.path.exists(subdir):
            print(f"⚠ Изображения в поддиректории: {subdir}")

    print("✓ COCO dataset готов!")
    return coco_dir


def filter_and_convert(
    coco_dir: str,
    output_dir: str,
    split: str = "train2017",
):
    """Фильтрация airplane и конвертация в YOLO формат."""
    annotations_file = os.path.join(coco_dir, "annotations", f"instances_{split}.json")
    images_dir = os.path.join(coco_dir, split)

    # Check if images are in subdirectory (e.g., val2017/val2017/)
    subdir_images = os.path.join(coco_dir, split, split)
    if os.path.exists(subdir_images):
        images_dir = subdir_images

    coco = COCO(annotations_file)
    cat_ids = coco.getCatIds(catNms=["airplane"])

    if not cat_ids:
        print(f"No airplane annotations found in {split}")
        return 0, 0

    img_ids = coco.getImgIds(catIds=cat_ids)
    print(f"Found {len(img_ids)} images with airplanes in {split}")

    yolo_images_dir = os.path.join(output_dir, "images", split)
    yolo_labels_dir = os.path.join(output_dir, "labels", split)
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)

    image_count = 0
    annotation_count = 0

    for img_id in tqdm(img_ids, desc=f"Processing {split}"):
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info["file_name"]
        src_image = os.path.join(images_dir, filename)

        if not os.path.exists(src_image):
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        annotations = coco.loadAnns(ann_ids)

        if not annotations:
            continue

        img_width = img_info["width"]
        img_height = img_info["height"]

        shutil.copy2(src_image, os.path.join(yolo_images_dir, filename))

        label_lines = []
        for ann in annotations:
            bbox = ann["bbox"]  # [x, y, width, height]
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height

            label_lines.append(
                f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            annotation_count += 1

        label_filename = Path(filename).stem + ".txt"
        with open(os.path.join(yolo_labels_dir, label_filename), "w") as f:
            f.write("\n".join(label_lines))

        image_count += 1

    print(f"{split}: {image_count} images, {annotation_count} annotations")
    return image_count, annotation_count


def create_yaml(output_dir: str):
    """Создать dataset.yaml для ultralytics."""
    yaml_content = """path: .
train: images/train2017
val: images/val2017

names:
  0: airplane
"""
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Created dataset.yaml at {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare COCO airplane dataset for YOLO"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Base data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/coco_airplane",
        help="Output directory",
    )
    parser.add_argument(
        "--val-only", action="store_true", help="Only process val2017 (for testing)"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip download, use existing data"
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_download:
        download_coco(args.data_dir)

    coco_dir = os.path.join(args.data_dir, "coco")

    if not args.val_only:
        print("\nProcessing train2017...")
        filter_and_convert(coco_dir, args.output_dir, "train2017")

    print("\nProcessing val2017...")
    filter_and_convert(coco_dir, args.output_dir, "val2017")

    create_yaml(args.output_dir)
    print("\nDataset preparation complete!")


def ensure_coco_airplane(data_dir: str = "/ml/data", output_dir: str = "/ml/data/coco_airplane"):
    """Проверить наличие подготовленного COCO airplane датасета, при необходимости распаковать и подготовить."""
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    train_img_dir = os.path.join(output_dir, "images", "train2017")
    val_img_dir = os.path.join(output_dir, "images", "val2017")

    # Проверить уже подготовленный датасет
    if (os.path.exists(yaml_path) and
        os.path.isdir(train_img_dir) and
        os.path.isdir(val_img_dir) and
        len(os.listdir(train_img_dir)) > 0 and
        len(os.listdir(val_img_dir)) > 0):
        train_count = len(os.listdir(train_img_dir))
        val_count = len(os.listdir(val_img_dir))
        print(f"✓ COCO airplane датасет уже подготовлен: {train_count} train, {val_count} val изображений")
        return output_dir

    print("⚠ COCO airplane датасет не обнаружен.")
    print("Распаковка и подготовка...")
    print()

    # Проверить что исходные архивы существуют
    coco_dir = os.path.join(data_dir, "coco")
    train_zip = os.path.join(coco_dir, "train2017.zip")
    val_zip = os.path.join(coco_dir, "val2017.zip")
    ann_zip = os.path.join(coco_dir, "annotations_trainval2017.zip")
    
    missing = []
    for z, name in [(train_zip, "train2017.zip"), (val_zip, "val2017.zip"), (ann_zip, "annotations_trainval2017.zip")]:
        if not os.path.exists(z) or os.path.getsize(z) < 1000:
            missing.append(name)
    
    if missing:
        print(f"✗ Отсутствуют архивы COCO: {', '.join(missing)}")
        print("Запустите: make download-coco")
        sys.exit(1)

    # Распаковать
    download_coco(data_dir)

    # Подготовить
    coco_dir = os.path.join(data_dir, "coco")
    print("\nФильтрация класса airplane и конвертация в YOLO формат...")
    filter_and_convert(coco_dir, output_dir, "train2017")
    filter_and_convert(coco_dir, output_dir, "val2017")
    create_yaml(output_dir)

    print("\n✓ COCO airplane датасет готов!")
    return output_dir


if __name__ == "__main__":
    main()
