"""MinIO storage client"""

import io
import logging
from pathlib import Path
from typing import Optional

from django.conf import settings

logger = logging.getLogger(__name__)


class MinIOClient:
    """Client for MinIO operations"""

    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize boto3 client"""
        try:
            import boto3
            from botocore.exceptions import ClientError

            self.client = boto3.client(
                "s3",
                endpoint_url=f"http://{settings.MINIO_ENDPOINT}",
                aws_access_key_id=settings.MINIO_ACCESS_KEY,
                aws_secret_access_key=settings.MINIO_SECRET_KEY,
            )
            self.ClientError = ClientError
            self._ensure_buckets()
        except ImportError:
            logger.warning("boto3 not installed, MinIO client disabled")
        except Exception as e:
            logger.warning(f"MinIO connection failed: {e}")

    def _ensure_buckets(self):
        """Create required buckets"""
        for bucket in ["models", "images", "results"]:
            try:
                self.client.head_bucket(Bucket=bucket)
            except self.ClientError:
                logger.info(f"Creating bucket: {bucket}")
                self.client.create_bucket(Bucket=bucket)

    def upload_file(
        self, file_path: Path, bucket: str = "images", object_name: Optional[str] = None
    ) -> str:
        """Upload file to MinIO bucket"""
        if not self.client:
            raise RuntimeError("MinIO client not initialized")

        if object_name is None:
            object_name = file_path.name

        try:
            self.client.upload_file(str(file_path), bucket, object_name)
            logger.info(f"Uploaded {file_path} to {bucket}/{object_name}")
            return f"http://{settings.MINIO_ENDPOINT}/{bucket}/{object_name}"
        except self.ClientError as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            raise

    def upload_bytes(
        self, data: bytes, bucket: str, object_name: str, content_type: str = "image/jpeg"
    ) -> str:
        """Upload bytes directly to MinIO"""
        if not self.client:
            raise RuntimeError("MinIO client not initialized")

        try:
            self.client.put_object(
                Bucket=bucket,
                Key=object_name,
                Body=io.BytesIO(data),
                ContentType=content_type,
            )
            logger.info(f"Uploaded {object_name} to {bucket}")
            return f"http://{settings.MINIO_ENDPOINT}/{bucket}/{object_name}"
        except self.ClientError as e:
            logger.error(f"Failed to upload bytes to {bucket}/{object_name}: {e}")
            raise

    def download_file(self, object_name: str, download_path: Path, bucket: str = "images") -> Path:
        """Download file from MinIO bucket"""
        if not self.client:
            raise RuntimeError("MinIO client not initialized")

        try:
            self.client.download_file(bucket, object_name, str(download_path))
            logger.info(f"Downloaded {object_name} to {download_path}")
            return download_path
        except self.ClientError as e:
            logger.error(f"Failed to download {object_name}: {e}")
            raise

    def get_presigned_url(self, object_name: str, bucket: str = "images", expiration: int = 3600) -> str:
        """Generate presigned URL for object"""
        if not self.client:
            raise RuntimeError("MinIO client not initialized")

        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": object_name},
                ExpiresIn=expiration,
            )
        except self.ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise


# Singleton instance
minio_client = MinIOClient()
