import React from 'react';
import { Button } from 'antd';
import { ExperimentOutlined } from '@ant-design/icons';
import { useRunClassification } from '../lib/useRunClassification';

export const RunClassificationFeature: React.FC = () => {
  const { runClassification, isRunning } = useRunClassification();

  return (
    <Button
      type="primary"
      icon={<ExperimentOutlined />}
      onClick={runClassification}
      loading={isRunning}
      size="large"
      style={{ width: '100%' }}
    >
      Запустить классификацию
    </Button>
  );
};
