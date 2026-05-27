#!/usr/bin/env python3
"""Download MLflow artifacts from MinIO for a given experiment/run using boto3."""
import os
import sys
import boto3
from botocore.client import Config


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <experiment_id> <run_id> <artifacts_dir>")
        sys.exit(1)

    experiment_id = sys.argv[1]
    run_id = sys.argv[2]
    artifacts_dir = sys.argv[3]

    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=Config(signature_version="s3v4"),
    )

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
        sys.exit(0)

    for key in keys:
        rel = key[len(prefix):]
        dest = os.path.join(artifacts_dir, rel)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        s3.download_file(bucket, key, dest)
        print(f"  ✓ {rel}")

    print(f"  Скачано {len(keys)} файлов")


if __name__ == "__main__":
    main()
