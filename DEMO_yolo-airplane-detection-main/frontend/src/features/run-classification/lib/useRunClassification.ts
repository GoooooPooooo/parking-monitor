import { useState } from 'react';
import { message } from 'antd';
import { useDetectionStore } from '../../../entities/detection/model';
import { detectionApi } from '../../../entities/detection/api';
import { useModelStore, modelApi } from '../../../entities/model';
import { useImageStore } from '../../../entities/image/model';
import { useHistoryStore } from '../../../entities/history';

export const useRunClassification = () => {
  const { setResult, setLoading, setError } = useDetectionStore();
  const { selectedModel, setModels } = useModelStore();
  const { uploadedId, uploadedImage } = useImageStore();
  const { addHistoryItem, removeLastDetectionResult } = useHistoryStore();
  const [isRunning, setIsRunning] = useState(false);

  const runClassification = async () => {
    if (!uploadedImage) {
      message.warning('Сначала загрузите изображение');
      return;
    }

    if (!uploadedId) {
      message.warning('Изображение не загружено. Загрузите заново.');
      return;
    }

    if (!selectedModel) {
      message.warning('Выберите модель для классификации');
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

      // Определяем тип модели из поля _type (устанавливается при выборе)
      const modelType = selectedModel._type || 'detector';
      
      let result;
      if (modelType === 'classifier') {
        // Для классификатора: удаляем предыдущий результат детекции из истории
        removeLastDetectionResult();
        
        // Используем новый эндпоинт для чистой классификации
        result = await detectionApi.runClassifyOnly(uploadedId, selectedModel.id);
      } else {
        // Старый пайплайн: детекция + классификация кропов
        result = await detectionApi.runDetectionAndClassify(uploadedId, selectedModel.id);
      }
      
      setResult(result);

      // Добавляем в историю с помощью функционального обновления
      addHistoryItem({
        id: result.id,
        image_url: result.result_image_url || result.image_url,
        result_image_url: result.result_image_url,
        original_url: result.original_url || '',
        detections: result.detections,
        count: result.count,
        model_id: `${selectedModel.id}+classifier`,
        created_at: result.created_at,
      });

      message.success('Классификация завершена успешно');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Ошибка при классификации';
      setError(errorMessage);
      message.error(errorMessage);
    } finally {
      setIsRunning(false);
      setLoading(false);
    }
  };

  return { runClassification, isRunning };
};
