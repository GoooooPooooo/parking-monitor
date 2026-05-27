import React from 'react';
import { Card, Image, Button } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import { UploadImageFeature } from '../../../features/upload-image';
import { useImageStore } from '../../../entities/image/model';

export const ImageUploaderWidget: React.FC = () => {
  const { uploadedImage, fileName, clearImage } = useImageStore();

  const handleChange = () => {
    clearImage();
  };

  return (
    <Card
      title="Загрузка изображения"
      extra={
        uploadedImage && (
          <Button
            icon={<ReloadOutlined />}
            onClick={handleChange}
            size="small"
          >
            Сменить
          </Button>
        )
      }
    >
      {!uploadedImage ? (
        <UploadImageFeature />
      ) : (
        <div style={{ textAlign: 'center' }}>
          <div
            style={{
              display: 'inline-block',
              border: '1px solid #f0f0f0',
              borderRadius: '8px',
              overflow: 'hidden',
              maxWidth: '100%',
            }}
          >
            <Image
              src={uploadedImage.src}
              alt={fileName || 'Загруженное изображение'}
              style={{
                display: 'block',
                maxWidth: '100%',
                maxHeight: '400px',
                objectFit: 'contain',
              }}
              preview={{
                classNames: { cover: 'custom-mask' },
              }}
            />
          </div>
          <p style={{
            marginTop: '12px',
            textAlign: 'center',
            color: '#666',
            fontSize: '14px',
          }}>
            {fileName}
          </p>
        </div>
      )}
    </Card>
  );
};
