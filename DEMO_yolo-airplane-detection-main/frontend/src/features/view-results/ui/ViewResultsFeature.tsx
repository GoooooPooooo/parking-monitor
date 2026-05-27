import React from 'react';
import { Card, Statistic, Descriptions, theme } from 'antd';
import type { BoundingBox, DetectionResult } from '../../../shared/types';
import { useDetectionStore } from '../../../entities/detection/model';

interface ViewResultsProps {
  result?: DetectionResult;
}

export const ViewResultsFeature: React.FC<ViewResultsProps> = ({ result }) => {
  const { currentResult, selectedBox, setSelectedBox } = useDetectionStore();
  const { token } = theme.useToken();
  const displayResult = result || currentResult;

  if (!displayResult) {
    return (
      <Card title="Результаты детекции">
        <p style={{ textAlign: 'center', color: token.colorTextSecondary }}>
          Результаты детекции появятся здесь
        </p>
      </Card>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <Card title="Результаты детекции">
        <Descriptions bordered column={1} size="small">
          <Descriptions.Item label="Модель">{displayResult.model_name}</Descriptions.Item>
          <Descriptions.Item label="Время">
            {new Date(displayResult.created_at).toLocaleString('ru-RU')}
          </Descriptions.Item>
          <Descriptions.Item label="Всего обнаружено">{displayResult.detections.length}</Descriptions.Item>
        </Descriptions>
      </Card>

      {/* Классификация объектов */}
      {displayResult.detections.length > 0 && (displayResult.detections[0] as any)?.classification_top?.length > 0 && (
        <Card title="Классификация объектов" size="small">
          {displayResult.detections.map((det: any, index: number) => {
            const topClassifications = det.classification_top || [];
            if (topClassifications.length === 0) return null;
            return (
              <div
                key={index}
                style={{
                  marginBottom: '12px',
                  padding: '8px 12px',
                  background: index % 2 === 0 ? token.colorBgContainer : token.colorBgElevated,
                  borderRadius: '6px',
                  border: `1px solid ${token.colorBorder}`,
                }}
              >
                <div style={{ fontWeight: 500, marginBottom: '4px' }}>
                  <strong>#{index + 1}</strong> <span style={{ color: token.colorTextSecondary }}>(det: {det.class_name})</span>
                </div>
                {topClassifications.map((cls: any, clsIndex: number) => {
                  if (!cls) return null;
                  return (
                    <div
                      key={clsIndex}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '4px 0',
                        borderBottom: clsIndex < topClassifications.length - 1 ? `1px solid ${token.colorBorder}` : 'none',
                      }}
                    >
                      <span style={{ color: clsIndex === 0 ? token.colorPrimary : token.colorText }}>
                        {clsIndex + 1}. {cls.class_name}
                      </span>
                      <span style={{ color: token.colorPrimary, fontWeight: clsIndex === 0 ? 700 : 400 }}>
                        {(cls.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </Card>
      )}

      {selectedBox && (
        <Card title="Информация об объекте" size="small">
          <Statistic title="Класс" value={selectedBox.class} />
          <Statistic
            title="Уверенность"
            value={(selectedBox.confidence * 100).toFixed(2)}
            suffix="%"
            style={{ marginTop: '8px' }}
          />
        </Card>
      )}
    </div>
  );
};
