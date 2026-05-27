import React from 'react';
import { Card } from 'antd';
import { FilterHistoryFeature } from '../../../features/filter-history';

export const HistoryWidget: React.FC = () => {
  return (
    <Card title="Фильтры">
      <FilterHistoryFeature />
    </Card>
  );
};
