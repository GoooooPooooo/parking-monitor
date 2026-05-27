import React from 'react';
import { Button } from 'antd';
import { ScanOutlined } from '@ant-design/icons';
import { useRunDetection } from '../lib/useRunDetection';

export const RunDetectionFeature: React.FC = () => {
  const { runDetection, isRunning } = useRunDetection();

  return (
    <Button
      type="primary"
      icon={<ScanOutlined />}
      onClick={runDetection}
      loading={isRunning}
      size="large"
      style={{ width: '100%' }}
    >
      Запустить детекцию
    </Button>
  );
};
