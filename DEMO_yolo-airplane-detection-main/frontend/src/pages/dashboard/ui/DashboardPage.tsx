import React, { useEffect } from 'react';
import { Card, Row, Col, Statistic } from 'antd';
import { MetricsTable } from '../../../entities/metrics/ui';
import { useMetricsStore } from '../../../entities/metrics/model';
import { metricsApi } from '../../../entities/metrics/api';

export const DashboardPage: React.FC = () => {
  const { runs, isLoading, setRuns, setLoading } = useMetricsStore();

  useEffect(() => {
    const fetchMetrics = async () => {
      setLoading(true);
      try {
        const data = await metricsApi.getMetrics();
        setRuns(data);
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, []);

  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ marginBottom: '24px' }}>Дашборд MLflow</h1>
      
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic title="Всего запусков" value={runs.length} />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Успешных"
              value={runs.filter((r) => r.status === 'FINISHED').length}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Неудачных"
              value={runs.filter((r) => r.status === 'FAILED').length}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Моделей"
              value={new Set(runs.map((r) => r.name)).size}
            />
          </Card>
        </Col>
      </Row>

      <Card title="Метрики MLflow">
        <MetricsTable runs={runs} loading={isLoading} />
      </Card>
    </div>
  );
};
