"""
Скачивание и подготовка датасета FGVC Aircraft.

Автоматически скачивает датасет если он отсутствует и подготавливает
структуру директорий для PyTorch ImageFolder.

Dataset: FGVC Aircraft (2013b)
Source: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
"""

import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
URL_FALLBACK = "http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
DEFAULT_DATA_ROOT = "/ml/data"
DEFAULT_SRC_DIR = os.path.join(DEFAULT_DATA_ROOT, "fgvc-aircraft-2013b")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATA_ROOT, "fgvc_aircraft_imagefolder")


def download_dataset(dest_dir: str = DEFAULT_DATA_ROOT, max_retries: int = 3):
    """Распаковать датасет FGVC Aircraft.
    
    Примечание: скачивание выполняется на хосте через Makefile.
    Эта функция только распаковывает уже скачанный архив.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    tar_path = dest / "fgvc-aircraft-2013b.tar.gz"

    # Проверить есть ли уже распакованная версия
    src_dir = dest / "fgvc-aircraft-2013b"
    if src_dir.exists() and (src_dir / "data").exists():
        print(f"✓ Датасет уже существует: {src_dir}")
        return str(src_dir)

    # Проверить наличие архива (с хоста через volume)
    if not tar_path.exists():
        print(f"✗ Архив не найден: {tar_path}")
        print("Запустите: make download-fgvc")
        sys.exit(1)

    # Проверить целостность
    if not _verify_tar_gz(str(tar_path)):
        print(f"⚠ Архив повреждён: {tar_path}")
        print("Запустите: make download-fgvc (файл будет перескачан на хосте)")
        sys.exit(1)

    # Распаковать
    _extract_tar_gz(str(tar_path), str(dest))

    return str(src_dir)


def _download_with_resume(url: str, dest: str, chunk_size: int = 65536):
    """Скачать файл с поддержкой resume и прогресс-баром."""
    import http.client
    import ssl

    # Проверить существующий размер
    existing_size = 0
    if os.path.exists(dest):
        existing_size = os.path.getsize(dest)

    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")
        print(f"  Resume с {existing_size / 1024 / 1024:.1f} MB")

    # Увеличенный таймаут + отключение verify для проблемных серверов
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, timeout=300, context=ctx) as response:
            total = response.getheader("Content-Length")
            total = int(total) + existing_size if total else 0

            content_range = response.getheader("Content-Range")
            if content_range:
                # Parse "bytes 12345-67890/total"
                total = int(content_range.split("/")[1])

            mode = "ab" if existing_size > 0 else "wb"
            downloaded = existing_size

            with open(dest, mode) as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = min(100, downloaded * 100 // total)
                        mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        print(f"\r  {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

        print()  # newline
    except (http.client.IncompleteRead, ConnectionResetError, TimeoutError) as e:
        raise ConnectionError(f"Соединение разорвано: {e}. Попробуйте снова — скачивание будет продолжено.")


def _verify_tar_gz(path: str) -> bool:
    """Проверить что файл валидный tar.gz."""
    try:
        with tarfile.open(path, "r:gz") as t:
            t.getmembers()[:1]  # прочитать первый entry
        return True
    except Exception:
        return False


def _extract_tar_gz(path: str, dest: str):
    """Распаковать tar.gz файл."""
    print(f"Распаковка: {os.path.basename(path)}...")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path=dest)
    print("✓ Распаковка завершена")


def prepare_fgvc(src_dir: str, output_dir: str):
    """Подготовить FGVC Aircraft датасет для ImageFolder."""
    src = Path(src_dir)
    images_dir = src / "data" / "images"

    if not images_dir.exists():
        print(f"✗ Директория изображений не найдена: {images_dir}")
        sys.exit(1)

    train_file = src / "data" / "images_variant_train.txt"
    val_file = src / "data" / "images_variant_val.txt"

    if not train_file.exists() or not val_file.exists():
        print(f"✗ Файлы разметки не найдены в: {src / 'data'}")
        sys.exit(1)

    with open(train_file) as f:
        train_list = [line.strip() for line in f]
    with open(val_file) as f:
        val_list = [line.strip() for line in f]

    out = Path(output_dir)
    for split in ["train", "val"]:
        (out / split).mkdir(parents=True, exist_ok=True)

    processed = {"train": 0, "val": 0}

    for split_name, split_list in [("train", train_list), ("val", val_list)]:
        for item in split_list:
            parts = item.split(" ", 1)
            if len(parts) != 2:
                continue
            img_id, variant = parts[0], parts[1]

            src_img = images_dir / f"{img_id}.jpg"
            if not src_img.exists():
                continue

            class_dir = out / split_name / variant
            class_dir.mkdir(parents=True, exist_ok=True)

            dst = class_dir / f"{img_id}.jpg"
            if not dst.exists():
                try:
                    os.link(src_img, dst)
                except OSError:
                    import shutil
                    shutil.copy2(src_img, dst)
            processed[split_name] += 1

    print(f"✓ Train images: {processed['train']}")
    print(f"✓ Val images: {processed['val']}")
    print(f"✓ Output: {output_dir}")


def ensure_dataset(
    data_root: str = DEFAULT_DATA_ROOT,
    src_dir: str = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    """Проверить и при необходимости скачать/подготовить датасет."""
    output_path = Path(output_dir)

    if (output_path / "train").exists() and (output_path / "val").exists():
        train_count = sum(1 for _ in (output_path / "train").rglob("*.jpg"))
        val_count = sum(1 for _ in (output_path / "val").rglob("*.jpg"))
        print(f"✓ Датасет уже подготовлен: {train_count} train, {val_count} val изображений")
        return output_dir

    print("Датасет FGVC Aircraft не обнаружен.")
    print("Скачивание и подготовка...")
    print()

    if src_dir is None:
        src_dir = data_root + "/fgvc-aircraft-2013b"

    if not Path(src_dir).exists() or not (Path(src_dir) / "data").exists():
        src_dir = download_dataset(data_root)

    print("\nПодготовка структуры ImageFolder...")
    prepare_fgvc(src_dir, output_dir)

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Скачать и подготовить датасет FGVC Aircraft")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Корневая директория для данных")
    parser.add_argument("--src-dir", default=None, help="Директория с распакованным датасетом")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Директория для ImageFolder структуры")
    parser.add_argument("--download-only", action="store_true", help="Только скачать, без подготовки")
    args = parser.parse_args()

    if args.download_only:
        download_dataset(args.data_root)
    else:
        ensure_dataset(
            data_root=args.data_root,
            src_dir=args.src_dir,
            output_dir=args.output_dir,
        )
