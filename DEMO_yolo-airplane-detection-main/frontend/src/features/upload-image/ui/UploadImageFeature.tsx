import React from 'react';
import { Upload } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import { useUploadImage } from '../lib/useUploadImage';

const { Dragger } = Upload;

export const UploadImageFeature: React.FC = () => {
  const { uploadImage } = useUploadImage();

  const props = {
    name: 'file',
    multiple: false,
    accept: 'image/*',
    beforeUpload: (file: File) => {
      uploadImage(file);
      return false;
    },
  };

  return (
    <Dragger {...props}>
      <p className="ant-upload-drag-icon">
        <InboxOutlined />
      </p>
      <p className="ant-upload-text">Нажмите или перетащите изображение для загрузки</p>
      <p className="ant-upload-hint">
        Поддерживаются форматы: JPEG, PNG, JPG, WebP. Максимальный размер: 10MB
      </p>
    </Dragger>
  );
};
