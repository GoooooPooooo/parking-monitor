import React from 'react';
import { Card, Row, Col, Image, Empty, Tag, Button, message, theme } from 'antd';
import { DeleteOutlined, PictureOutlined, ScanOutlined } from '@ant-design/icons';
import type { HistoryItem } from '../model';
import { useHistoryStore } from '../model/store';
import { historyApi } from '../api';

interface HistoryCollectionsProps {
  items: HistoryItem[];
}

export const HistoryCollections: React.FC<HistoryCollectionsProps> = ({ items }) => {
  const { removeItem } = useHistoryStore();
  const { token } = theme.useToken();

  const handleDelete = async (id: string) => {
    try {
      await historyApi.deleteResult(id);
      removeItem(id);
      message.success('Результат удалён');
    } catch (error) {
      message.error('Ошибка при удалении');
    }
  };

  if (items.length === 0) return null;

  return (
    <div style={{ marginTop: '16px' }}>
      <Row gutter={[16, 16]}>
        {items.map((item) => (
          <Col xs={24} key={item.id}>
            <Card
              size="small"
              title={
                <span>
                  <ScanOutlined style={{ marginRight: '8px' }} />
                  Результат #{item.id.slice(0, 8)}
                </span>
              }
              extra={
                <Button
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => handleDelete(item.id)}
                  size="small"
                >
                  Удалить
                </Button>
              }
            >
              <Row gutter={[16, 16]}>
                {/* Исходное изображение */}
                <Col xs={24} md={12}>
                  <h4 style={{ marginBottom: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <PictureOutlined style={{ marginRight: '6px' }} />
                    Исходное
                  </h4>
                  <div style={{ display: 'flex', justifyContent: 'center' }}>
                    {item.original_url ? (
                      <Image
                        src={item.original_url}
                        alt="Исходное изображение"
                        style={{
                          maxWidth: '100%',
                          maxHeight: '300px',
                          objectFit: 'contain',
                          borderRadius: '8px',
                        }}
                      />
                    ) : (
                      <Empty description="Нет изображения" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                    )}
                  </div>
                </Col>

                {/* Результат с детекцией */}
                <Col xs={24} md={12}>
                  <h4 style={{ marginBottom: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                    <ScanOutlined />
                    Результат
                    <Tag color="green">{item.count} объектов</Tag>
                  </h4>
                  <div style={{ display: 'flex', justifyContent: 'center' }}>
                    {(item.result_image_url || item.image_url) ? (
                      <Image
                        src={item.result_image_url || item.image_url}
                        alt="Результат детекции"
                        style={{
                          maxWidth: '100%',
                          maxHeight: '300px',
                          objectFit: 'contain',
                          borderRadius: '8px',
                        }}
                      />
                    ) : (
                      <Empty description="Нет результата" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                    )}
                  </div>
                  {/* Список обнаруженных объектов */}
                  {item.detections.length > 0 && (
                    <div
                      style={{
                        marginTop: '12px',
                        padding: '8px',
                        background: token.colorBgContainer,
                        borderRadius: '6px',
                        border: `1px solid ${token.colorBorder}`,
                      }}
                    >
                      {item.detections.map((det, index) => {
                        const topClassifications = (det as any).classification_top || [];
                        const hasClassification = topClassifications.length > 0;

                        return (
                          <div
                            key={index}
                            style={{
                              padding: '8px',
                              marginBottom: '6px',
                              background: token.colorBgElevated,
                              borderRadius: '6px',
                            }}
                          >
                            <div style={{ fontWeight: 500, marginBottom: '4px' }}>
                              <strong>#{index + 1}</strong> <span style={{ color: token.colorTextSecondary, fontSize: '12px' }}>(det: {det.class_name})</span>
                            </div>
                            {hasClassification ? (
                              topClassifications.map((cls: any, clsIndex: number) => (
                                <div
                                  key={clsIndex}
                                  style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    padding: '2px 0',
                                    fontSize: clsIndex === 0 ? '14px' : '12px',
                                    color: clsIndex === 0 ? token.colorPrimary : token.colorTextSecondary,
                                  }}
                                >
                                  <span>{clsIndex + 1}. {cls.class_name}</span>
                                  <span style={{ fontWeight: clsIndex === 0 ? 700 : 400 }}>
                                    {(cls.confidence * 100).toFixed(1)}%
                                  </span>
                                </div>
                              ))
                            ) : (
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>{det.class_name}</span>
                                <span style={{ color: token.colorPrimary, fontWeight: 500 }}>
                                  {(det.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </Col>
              </Row>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};
