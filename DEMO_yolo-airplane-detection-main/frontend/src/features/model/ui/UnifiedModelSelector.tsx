import { useState, useEffect } from 'react';
import { Modal, Flex, Tag, Spin, Alert, Typography, Space, Divider, Tabs } from 'antd';
import { CheckCircleOutlined, ClockCircleOutlined, StarFilled, RocketOutlined, ScanOutlined, AppstoreOutlined } from '@ant-design/icons';
import { modelApi, type MLflowModel } from '../../../entities/model/api';
import type { DetectionModel } from '../../../shared/types';

const { Title, Text } = Typography;

interface UnifiedModelSelectorProps {
  isOpen: boolean;
  onClose: () => void;
  onModelSelected?: (modelId: string, modelName: string, source: 'standard' | 'mlflow', modelType?: 'detector' | 'classifier') => void;
  standardModels: DetectionModel[];
}

export const UnifiedModelSelector = ({
  isOpen,
  onClose,
  onModelSelected,
  standardModels
}: UnifiedModelSelectorProps) => {
  const [mlflowModels, setMlflowModels] = useState<MLflowModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'detectors' | 'classifiers'>('detectors');

  useEffect(() => {
    if (isOpen) {
      loadMlflowModels();
    }
  }, [isOpen]);

  const loadMlflowModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await modelApi.getMlflowModels();
      setMlflowModels(data);
    } catch (err) {
      setError('Не удалось загрузить модели из MLflow');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectStandardModel = (model: DetectionModel) => {
    onModelSelected?.(model.id, model.name, 'standard', 'detector');
    // selectedModel будет установлен через setSelectedModel из DetectionPage
    onClose();
  };

  const handleSelectMlflowModel = (model: MLflowModel) => {
    // Просто передаём run_id, бэкенд сам загрузит модель из MLflow
    const modelId = `mlflow:${model.run_id}`;
    const modelType = model.is_classifier ? 'classifier' : 'detector';
    onModelSelected?.(modelId, model.run_name, 'mlflow', modelType);
    onClose();
  };

  return (
    <Modal
      title={
        <div>
          <Title level={4} style={{ margin: 0 }}>Выберите модель</Title>
          <Text type="secondary" style={{ fontSize: '14px' }}>
            Стандартные модели или загрузите из MLflow
          </Text>
        </div>
      }
      open={isOpen}
      onCancel={onClose}
      footer={null}
      width={800}
      style={{ top: 20 }}
    >
      {/* Стандартные модели */}
      <div style={{ marginBottom: 24 }}>
        <Title level={5}>
          <RocketOutlined /> Стандартные модели
        </Title>
        <Flex vertical gap={8}>
          {standardModels.map((model) => (
            <div
              key={model.id}
              style={{
                cursor: 'pointer',
                padding: '12px 16px',
                border: '1px solid #d9d9d9',
                borderRadius: '8px',
                transition: 'all 0.3s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = '#1890ff';
                e.currentTarget.style.backgroundColor = '#e6f7ff';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#d9d9d9';
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
              onClick={() => handleSelectStandardModel(model)}
            >
              <div>
                <Text strong>{model.name}</Text>
              </div>
              <div>
                <Text type="secondary">{model.description}</Text>
              </div>
            </div>
          ))}
        </Flex>
      </div>

      <Divider />

      {/* MLflow модели */}
      <div>
        <Title level={5}>
          <StarFilled /> Модели из MLflow
        </Title>
        <Text type="secondary" style={{ fontSize: '12px', display: 'block', marginBottom: 12 }}>
          Отсортированы по точности (mAP50 для детекторов, Accuracy для классификаторов)
        </Text>

        {/* Табы для переключения между детекторами и классификаторами */}
        <Tabs
          activeKey={activeTab}
          onChange={(key) => setActiveTab(key as 'detectors' | 'classifiers')}
          items={[
            {
              key: 'detectors',
              label: (
                <span>
                  <ScanOutlined /> Детекторы
                  <Tag color="blue" style={{ marginLeft: 8, fontSize: '11px' }}>
                    {mlflowModels.filter(m => !m.is_classifier).length}
                  </Tag>
                </span>
              ),
            },
            {
              key: 'classifiers',
              label: (
                <span>
                  <AppstoreOutlined /> Классификаторы
                  <Tag color="cyan" style={{ marginLeft: 8, fontSize: '11px' }}>
                    {mlflowModels.filter(m => m.is_classifier).length}
                  </Tag>
                </span>
              ),
            },
          ]}
          style={{ marginBottom: 16 }}
        />

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

        {!loading && mlflowModels.filter(m => activeTab === 'detectors' ? !m.is_classifier : m.is_classifier).length === 0 && (
          <div style={{ textAlign: 'center', padding: '20px 0' }}>
            <Text type="secondary">
              {activeTab === 'detectors' ? 'Детекторы не найдены в MLflow' : 'Классификаторы не найдены в MLflow'}
            </Text>
          </div>
        )}

        {!loading && mlflowModels.filter(m => activeTab === 'detectors' ? !m.is_classifier : m.is_classifier).length > 0 && (
          <Flex vertical gap={8}>
            {mlflowModels.filter(m => activeTab === 'detectors' ? !m.is_classifier : m.is_classifier).map((model) => {
              const isClassifier = model.is_classifier;
              const primaryMetric = isClassifier ? model.accuracy : model.mAP50;
              const metricLabel = isClassifier ? 'Accuracy' : 'mAP50';
              const metricColor = primaryMetric > 0.9 ? '#52c41a' : primaryMetric > 0.7 ? '#1890ff' : undefined;

              return (
              <div
                key={model.run_id}
                style={{
                  cursor: 'pointer',
                  padding: '12px 16px',
                  border: '1px solid #d9d9d9',
                  borderRadius: '8px',
                  transition: 'all 0.3s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#1890ff';
                  e.currentTarget.style.backgroundColor = '#e6f7ff';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#d9d9d9';
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
                onClick={() => handleSelectMlflowModel(model)}
              >
                <div>
                  <Space>
                    <Text strong>{model.run_name}</Text>
                    {isClassifier ? (
                      <Tag color="cyan" style={{ fontSize: '11px' }}>Classifier</Tag>
                    ) : (
                      <Tag color="blue" style={{ fontSize: '11px' }}>Detector</Tag>
                    )}
                    {model.status === 'FINISHED' ? (
                      <Tag icon={<CheckCircleOutlined />} color="success" style={{ fontSize: '11px' }}>
                        {model.status}
                      </Tag>
                    ) : (
                      <Tag icon={<ClockCircleOutlined />} color="warning" style={{ fontSize: '11px' }}>
                        {model.status}
                      </Tag>
                    )}
                    {primaryMetric > 0.9 && (
                      <Tag icon={<StarFilled />} color="purple" style={{ fontSize: '11px' }}>
                        Лучшая
                      </Tag>
                    )}
                  </Space>
                </div>
                <div style={{ marginTop: 8 }}>
                  <Flex vertical gap={4} style={{ width: '100%' }}>
                    <Space size={12} wrap style={{ fontSize: '12px' }}>
                      <Text type="secondary">
                        <strong>Тип:</strong> {model.model_type}
                      </Text>
                      <Text type="secondary">
                        <strong>Эпохи:</strong> {model.epochs}
                      </Text>
                      <Text type="secondary">
                        <strong>{metricLabel}:</strong>{' '}
                        <Text strong style={{ color: metricColor }}>
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
                    </Space>
                    <Space size={8}>
                      {model.has_pt && <Tag color="purple">PyTorch</Tag>}
                      {model.has_onnx && <Tag color="blue">ONNX</Tag>}
                    </Space>
                  </Flex>
                </div>
              </div>
              );
            })}
          </Flex>
        )}
      </div>
    </Modal>
  );
};
