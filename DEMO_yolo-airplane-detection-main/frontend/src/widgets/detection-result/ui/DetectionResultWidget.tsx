import React, { useState } from 'react';
import { Card, Button, Flex } from 'antd';
import { RocketOutlined } from '@ant-design/icons';
import { RunDetectionFeature } from '../../../features/run-detection';
import { RunClassificationFeature } from '../../../features/run-classification';
import { ViewResultsFeature } from '../../../features/view-results';
import { UnifiedModelSelector } from '../../../features/model/ui/UnifiedModelSelector';
import { useModelStore, modelApi } from '../../../entities/model';

export const DetectionResultWidget: React.FC = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedModelName, setSelectedModelName] = useState<string>('');
  const { models, setSelectedModel, setModels } = useModelStore();

  const handleModelSelected = async (modelId: string, modelName: string, source: 'standard' | 'mlflow', modelType: 'detector' | 'classifier' = 'detector') => {
    // Обновляем список моделей чтобы MLflow модель появилась в списке
    try {
      const updatedModels = await modelApi.getModels();
      setModels(updatedModels);
    } catch (error) {
      console.error('Failed to refresh models:', error);
    }

    setSelectedModel(modelId, modelType);
    setSelectedModelName(modelName);
    setIsModalOpen(false);
  };

  return (
    <Card title="Управление детекцией">
      <Flex vertical gap="large" style={{ width: '100%' }}>
        <div>
          <Flex vertical gap="small" style={{ width: '100%' }}>
            <Button
              type="primary"
              icon={<RocketOutlined />}
              onClick={() => setIsModalOpen(true)}
              style={{ width: '100%' }}
            >
              Выбрать модель
            </Button>
            {selectedModelName && (
              <div style={{ fontSize: '12px', color: '#666' }}>
                Выбрана модель: <strong>{selectedModelName}</strong>
              </div>
            )}
          </Flex>
        </div>

        <div style={{ display: 'flex', gap: '8px', width: '100%' }}>
          <div style={{ flex: 1 }}>
            <RunDetectionFeature />
          </div>
          <div style={{ flex: 1 }}>
            <RunClassificationFeature />
          </div>
        </div>

        <ViewResultsFeature />
      </Flex>

      <UnifiedModelSelector
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onModelSelected={handleModelSelected}
        standardModels={models}
      />
    </Card>
  );
};
