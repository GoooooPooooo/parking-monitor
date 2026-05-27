#!/usr/bin/env python3
"""Upload/download MLflow artifacts to/from MinIO using boto3."""
import argparse
import os
import sys
import time
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


def get_client():
    return boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=Config(signature_version="s3v4"),
    )


def wait_for_bucket(bucket="mlflow", timeout=60, interval=2):
    """Wait for MinIO bucket to be available."""
    s3 = get_client()
    print(f"Ожидание доступности бакета '{bucket}'...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"✓ Бакет '{bucket}' доступен")
            return True
        except ClientError:
            time.sleep(interval)
    print(f"✗ Бакет '{bucket}' не стал доступен за {timeout} секунд")
    return False


def upload_artifacts(experiment_id, run_id, artifacts_dir):
    """Upload local artifacts to MinIO."""
    s3 = get_client()
    bucket = "mlflow"
    prefix = f"{experiment_id}/{run_id}/artifacts/"

    if not os.path.isdir(artifacts_dir):
        print(f"  ⚠ Директория не найдена: {artifacts_dir}")
        return False

    files = []
    for root, _, filenames in os.walk(artifacts_dir):
        for fn in filenames:
            files.append(os.path.join(root, fn))

    if not files:
        print(f"  ⚠ Нет файлов для загрузки")
        return False

    for local_path in files:
        rel = os.path.relpath(local_path, artifacts_dir)
        key = prefix + rel
        s3.upload_file(local_path, bucket, key)
        print(f"  ✓ {rel}")

    print(f"  Загружено {len(files)} файлов")
    return True


def download_artifacts(experiment_id, run_id, artifacts_dir):
    """Download artifacts from MinIO to local directory."""
    s3 = get_client()
    bucket = "mlflow"
    prefix = f"{experiment_id}/{run_id}/artifacts/"
    os.makedirs(artifacts_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            keys.append(key)

    if not keys:
        print("  ⚠ Артефакты не найдены в MinIO")
        return

    for key in keys:
        rel = key[len(prefix):]
        dest = os.path.join(artifacts_dir, rel)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        s3.download_file(bucket, key, dest)
        print(f"  ✓ {rel}")

    print(f"  Скачано {len(keys)} файлов")


def fix_artifact_uri(mlruns_dir):
    """Fix artifact_uri in meta.yaml files from file:// to s3://."""
    for root, dirs, filenames in os.walk(mlruns_dir):
        if "meta.yaml" in filenames and "artifacts" in dirs:
            meta_path = os.path.join(root, "meta.yaml")
            try:
                with open(meta_path, "r") as f:
                    content = f.read()
                if "artifact_uri: file://" in content:
                    # Extract experiment_id and run_id from path
                    parts = os.path.normpath(root).split(os.sep)
                    if len(parts) >= 2:
                        run_id = parts[-1]
                        exp_id = parts[-2]
                        new_uri = f"artifact_uri: s3://mlflow/{exp_id}/{run_id}/artifacts"
                        content = content.replace(
                            content.split("artifact_uri:")[1].split("\n")[0].strip(),
                            new_uri.replace("artifact_uri: ", ""),
                        )
                        # Simpler approach
                        import re
                        content = re.sub(
                            r"artifact_uri: file://.*",
                            new_uri,
                            content,
                        )
                        with open(meta_path, "w") as f:
                            f.write(content)
                        print(f"  ✓ Обновлён artifact_uri в {meta_path}")
            except (PermissionError, Exception) as e:
                print(f"  ⚠ Не удалось обновить {meta_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Sync MLflow artifacts with MinIO")
    sub = parser.add_subparsers(dest="command", required=True)

    # upload
    up = sub.add_parser("upload")
    up.add_argument("mlruns_dir", nargs="?", default="mlruns")
    up.add_argument("--fix-uri", action="store_true", help="Fix artifact_uri in meta.yaml")

    # download
    dl = sub.add_parser("download")
    dl.add_argument("experiment_id")
    dl.add_argument("run_id")
    dl.add_argument("artifacts_dir")

    args = parser.parse_args()

    if args.command == "upload":
        # Wait for bucket to be available
        if not wait_for_bucket():
            sys.exit(1)
        
        found = False
        for exp_dir in sorted(os.listdir(args.mlruns_dir)):
            exp_path = os.path.join(args.mlruns_dir, exp_dir)
            if not os.path.isdir(exp_path) or exp_dir in ("0", "models", ".trash"):
                continue
            print(f"\nЭксперимент: {exp_dir}")
            for run_id in sorted(os.listdir(exp_path)):
                run_path = os.path.join(exp_path, run_id)
                art_dir = os.path.join(run_path, "artifacts")
                if os.path.isdir(art_dir) and os.path.exists(os.path.join(run_path, "meta.yaml")):
                    if os.listdir(art_dir):
                        print(f"  Загрузка артефактов для {run_id}...")
                        if upload_artifacts(exp_dir, run_id, art_dir):
                            found = True
        if args.fix_uri:
            print("\nИсправление artifact_uri...")
            fix_artifact_uri(args.mlruns_dir)
        if found:
            print("\n✓ Все артефакты загружены в MinIO!")
        else:
            print("Артефакты не найдены")

    elif args.command == "download":
        download_artifacts(args.experiment_id, args.run_id, args.artifacts_dir)


if __name__ == "__main__":
    main()
