import React, { useEffect, useState } from 'react';
import { Row, Col, Card, Space, Button, Image, Flex } from 'antd';
import { RocketOutlined } from '@ant-design/icons';
import { ImageUploaderWidget } from '../../../widgets/image-uploader';
import { modelApi, useModelStore } from '../../../entities/model';
import { RunDetectionFeature } from '../../../features/run-detection';
import { RunClassificationFeature } from '../../../features/run-classification';
import { UnifiedModelSelector } from '../../../features/model/ui/UnifiedModelSelector';
import { useDetectionStore } from '../../../entities/detection/model';
import { useImageStore } from '../../../entities/image/model';

export const DetectionPage: React.FC = () => {
  const { setModels } = useModelStore();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedModelName, setSelectedModelName] = useState<string>('');
  const { models, setSelectedModel } = useModelStore();
  const { currentResult } = useDetectionStore();
  const { uploadedImage } = useImageStore();

  // Загружаем модели при маунте страницы
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await modelApi.getModels();
        setModels(data);
      } catch (error) {
        console.error('Failed to fetch models:', error);
      }
    };

    fetchModels();
  }, []);

  const handleModelSelected = async (modelId: string, modelName: string, source: 'standard' | 'mlflow', modelType: 'detector' | 'classifier' = 'detector') => {
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
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ margin: 0 }}>Детекция самолетов</h1>
      </div>

      {/* Верхняя строка: Управление детекцией */}
      <Row gutter={[16, 16]} style={{ marginBottom: '16px' }}>
        <Col xs={24}>
          <Card title="Управление детекцией">
            <Flex vertical gap="large" style={{ width: '100%' }}>
              <div>
                <Flex vertical gap="small" style={{ width: '100%' }}>
                  <Button
                    type="primary"
                    icon={<RocketOutlined />}
                    onClick={() => setIsModalOpen(true)}
                    style={{ width: '100%' }}
                    size="large"
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

              <Row gutter={[16, 16]}>
                <Col xs={24} md={12}>
                  <RunDetectionFeature />
                </Col>
                <Col xs={24} md={12}>
                  <RunClassificationFeature />
                </Col>
              </Row>
            </Flex>
          </Card>
        </Col>
      </Row>

      {/* Нижняя строка: Загрузка изображения + Изображение с детекцией */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12} style={{ display: 'flex' }}>
          <div style={{ width: '100%' }}>
            <ImageUploaderWidget />
          </div>
        </Col>
        <Col xs={24} lg={12} style={{ display: 'flex' }}>
          <Card title="Изображение с детекцией" style={{ width: '100%' }}>
            {currentResult && currentResult.result_image_url ? (
              <div style={{ textAlign: 'center' }}>
                <Image
                  src={currentResult.result_image_url}
                  alt="Результат детекции"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '400px',
                    objectFit: 'contain',
                    borderRadius: '8px',
                  }}
                  preview={{
                    classNames: { cover: 'custom-mask' },
                  }}
                />
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '60px 20px', color: '#999' }}>
                {uploadedImage ? 'Запустите детекцию' : 'Загрузите изображение'}
              </div>
            )}
          </Card>
        </Col>
      </Row>

      <UnifiedModelSelector
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onModelSelected={handleModelSelected}
        standardModels={models}
      />
    </div>
  );
};
