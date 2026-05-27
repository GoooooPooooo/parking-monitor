import { useState } from 'react';
import { App } from 'antd';
import { useImageStore } from '../../../entities/image/model';
import { detectionApi } from '../../../entities/detection/api';
import { validateImageFile } from '../../../shared/lib/helpers';

export const useUploadImage = () => {
  const { message } = App.useApp();
  const { setUploadedImage, setFileName, setLoading, setError, setUploadedId } = useImageStore();
  const [isUploading, setIsUploading] = useState(false);

  const uploadImage = async (file: File) => {
    const validationError = validateImageFile(file);
    if (validationError) {
      message.error(validationError);
      return;
    }

    setIsUploading(true);
    setLoading(true);

    try {
      const result = await detectionApi.uploadImage(file);

      // Сохраняем ID загруженного изображения для детекции
      setUploadedId(result.id);

      // Загружаем изображение для превью
      const img = new Image();
      img.onload = () => {
        setUploadedImage(img);
        setFileName(file.name);
        message.success('Изображение успешно загружено');
      };
      img.onerror = () => {
        setFileName(file.name);
        message.success('Изображение загружено');
      };
      img.src = result.original_url || URL.createObjectURL(file);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Ошибка загрузки изображения';
      setError(errorMessage);
      message.error(errorMessage);
    } finally {
      setIsUploading(false);
      setLoading(false);
    }
  };

  return { uploadImage, isUploading };
};
