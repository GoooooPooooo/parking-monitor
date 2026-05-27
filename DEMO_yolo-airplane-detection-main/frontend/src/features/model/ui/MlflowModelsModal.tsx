import { useState, useEffect } from 'react';
import { Modal, List, Tag, Spin, Alert, Typography, Space, Button, Flex } from 'antd';
import { CheckCircleOutlined, ClockCircleOutlined, StarFilled, DownloadOutlined } from '@ant-design/icons';
import { modelApi, type MLflowModel } from '../../../entities/model/api';

const { Title, Text } = Typography;

interface MlflowModelsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onModelSelected?: (modelName: string) => void;
}

export const MlflowModelsModal = ({ isOpen, onClose, onModelSelected }: MlflowModelsModalProps) => {
  const [models, setModels] = useState<MLflowModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadModels();
    }
  }, [isOpen]);

  const loadModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await modelApi.getMlflowModels();
      setModels(data);
    } catch (err) {
      setError('Не удалось загрузить модели из MLflow');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectModel = async (model: MLflowModel) => {
    setDownloading(model.run_id);
    setError(null);
    try {
      const modelName = `mlflow_${model.model_type}_${model.run_name}`.replace(/[^a-zA-Z0-9_]/g, '_');
      await modelApi.downloadMlflowModel(model.run_id, modelName);
      
      onModelSelected?.(modelName);
      onClose();
    } catch (err) {
      setError('Не удалось загрузить модель');
      console.error(err);
      setDownloading(null);
    }
  };

  return (
    <Modal
      title={
        <div>
          <Title level={4} style={{ margin: 0 }}>Выберите модель из MLflow</Title>
          <Text type="secondary" style={{ fontSize: '14px' }}>
            Модели отсортированы по точности (mAP50 для детекторов, Accuracy для классификаторов)
          </Text>
        </div>
      }
      open={isOpen}
      onCancel={onClose}
      footer={[
        <Text key="count" type="secondary" style={{ float: 'left', lineHeight: '32px' }}>
          Найдено моделей: {models.length}
        </Text>,
        <Button key="cancel" onClick={onClose}>
          Отмена
        </Button>
      ]}
      width={800}
      style={{ top: 20 }}
    >
      {loading && (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>
            <Text type="secondary">Загрузка моделей...</Text>
          </div>
        </div>
      )}

      {error && (
        <Alert
          message={error}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {!loading && models.length === 0 && (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Text type="secondary">Модели не найдены в MLflow</Text>
        </div>
      )}

      {!loading && models.length > 0 && (
        <List
          dataSource={models}
          renderItem={(model) => {
            const isClassifier = model.is_classifier;
            const primaryMetric = isClassifier ? model.accuracy : model.mAP50;
            const metricLabel = isClassifier ? 'Accuracy' : 'mAP50';
            const metricColor = primaryMetric > 0.9 ? '#52c41a' : primaryMetric > 0.7 ? '#1890ff' : undefined;

            return (
            <List.Item
              key={model.run_id}
              style={{
                cursor: downloading ? 'not-allowed' : 'pointer',
                padding: '16px',
                border: '1px solid #d9d9d9',
                borderRadius: '8px',
                marginBottom: '12px',
                transition: 'all 0.3s',
                opacity: downloading && downloading !== model.run_id ? 0.5 : 1,
              }}
              onMouseEnter={(e) => {
                if (!downloading) {
                  e.currentTarget.style.borderColor = '#1890ff';
                  e.currentTarget.style.backgroundColor = '#e6f7ff';
                }
              }}
              onMouseLeave={(e) => {
                if (!downloading) {
                  e.currentTarget.style.borderColor = '#d9d9d9';
                  e.currentTarget.style.backgroundColor = 'transparent';
                }
              }}
              onClick={() => !downloading && handleSelectModel(model)}
            >
              <List.Item.Meta
                title={
                  <Space>
                    <Text strong>{model.run_name}</Text>
                    {isClassifier ? (
                      <Tag color="cyan">Classifier</Tag>
                    ) : (
                      <Tag color="blue">Detector</Tag>
                    )}
                    {model.status === 'FINISHED' ? (
                      <Tag icon={<CheckCircleOutlined />} color="success">
                        {model.status}
                      </Tag>
                    ) : (
                      <Tag icon={<ClockCircleOutlined />} color="warning">
                        {model.status}
                      </Tag>
                    )}
                    {primaryMetric > 0.9 && (
                      <Tag icon={<StarFilled />} color="purple">
                        Лучшая
                      </Tag>
                    )}
                    {downloading === model.run_id && (
                      <Tag icon={<Spin size="small" />} color="blue">
                        Загрузка...
                      </Tag>
                    )}
                  </Space>
                }
                description={
                  <div style={{ marginTop: 8 }}>
                    <Flex vertical gap={4} style={{ width: '100%' }}>
                      <Space size={16} wrap>
                        <Text type="secondary">
                          <strong>Тип:</strong> {model.model_type}
                        </Text>
                        <Text type="secondary">
                          <strong>Эпохи:</strong> {model.epochs}
                        </Text>
                        <Text type="secondary">
                          <strong>{metricLabel}:</strong>{' '}
                          <Text
                            strong
                            style={{ color: metricColor }}
                          >
                            {(primaryMetric * 100).toFixed(2)}%
                          </Text>
                        </Text>
                        {isClassifier && model.top5_accuracy > 0 && (
                          <Text type="secondary">
                            <strong>Top-5:</strong>{' '}
                            <Text strong>{(model.top5_accuracy * 100).toFixed(2)}%</Text>
                          </Text>
                        )}
                        {!isClassifier && model.top5_accuracy > 0 && (
                          <Text type="secondary">
                            <strong>mAP50-95:</strong>{' '}
                            <Text strong>{(model.top5_accuracy * 100).toFixed(2)}%</Text>
                          </Text>
                        )}
                        <Text type="secondary">
                          <strong>Формат:</strong>{' '}
                          {model.has_onnx && <Tag color="blue">ONNX</Tag>}
                          {model.has_pt && <Tag color="purple">PyTorch</Tag>}
                        </Text>
                      </Space>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        Создано: {new Date(model.created_at).toLocaleString('ru-RU')}
                      </Text>
                    </Flex>
                  </div>
                }
              />
            </List.Item>
            );
          }}
        />
      )}
    </Modal>
  );
};
