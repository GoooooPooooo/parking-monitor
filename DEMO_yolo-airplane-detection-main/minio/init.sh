#!/bin/bash
set -e

mc alias set local http://minio:9000 minioadmin minioadmin

mc mb local/models --ignore-existing
mc mb local/images --ignore-existing
mc mb local/results --ignore-existing

echo "MinIO buckets created: models, images, results"
