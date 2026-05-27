import { useState } from 'react';
import { message } from 'antd';
import { useDetectionStore } from '../../../entities/detection/model';
import { useImageStore } from '../../../entities/image/model';
import { useModelStore, modelApi } from '../../../entities/model';
import { detectionApi } from '../../../entities/detection/api';
import { useHistoryStore } from '../../../entities/history';

export const useRunDetection = () => {
  const { setResult, setLoading, setError } = useDetectionStore();
  const { uploadedImage, uploadedId } = useImageStore();
  const { selectedModel, setModels } = useModelStore();
  const { addHistoryItem } = useHistoryStore();
  const [isRunning, setIsRunning] = useState(false);

  const runDetection = async () => {
    if (!uploadedImage) {
      message.warning('Сначала загрузите изображение');
      return;
    }

    if (!uploadedId) {
      message.warning('ID изображения не найден. Загрузите заново.');
      return;
    }

    if (!selectedModel) {
      message.warning('Выберите модель для детекции');
      return;
    }

    setIsRunning(true);
    setLoading(true);

    try {
      // Обновляем список моделей
      try {
        const models = await modelApi.getModels();
        setModels(models);
      } catch {}

      // Загружаем детекцию с выбранной моделью
      const result = await detectionApi.runDetection(uploadedId, selectedModel.id);
      
      setResult(result);

      // Добавляем в историю с помощью функционального обновления
      addHistoryItem({
        id: result.id,
        image_url: result.result_image_url || result.image_url,
        result_image_url: result.result_image_url,
        original_url: result.original_url || '',
        detections: result.detections,
        count: result.count,
        model_id: selectedModel.id,
        created_at: result.created_at,
      });

      message.success('Детекция завершена успешно');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Ошибка при детекции';
      setError(errorMessage);
      message.error(errorMessage);
    } finally {
      setIsRunning(false);
      setLoading(false);
    }
  };

  return { runDetection, isRunning };
};
