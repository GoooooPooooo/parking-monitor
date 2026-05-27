import React from 'react';
import { Input, DatePicker, Button, Space } from 'antd';
import { SearchOutlined, ClearOutlined } from '@ant-design/icons';
import { useFilterHistory } from '../lib/useFilterHistory';

const { RangePicker } = DatePicker;

export const FilterHistoryFeature: React.FC = () => {
  const { filters, updateFilter, resetFilters } = useFilterHistory();

  return (
    <Space orientation="vertical" style={{ width: '100%' }}>
      <Input
        placeholder="Поиск..."
        prefix={<SearchOutlined />}
        value={filters.search}
        onChange={(e) => updateFilter('search', e.target.value)}
        allowClear
      />
      <RangePicker
        style={{ width: '100%' }}
        onChange={(dates) => {
          if (dates) {
            updateFilter('dateFrom', dates[0]?.format('YYYY-MM-DD') || null);
            updateFilter('dateTo', dates[1]?.format('YYYY-MM-DD') || null);
          } else {
            updateFilter('dateFrom', null);
            updateFilter('dateTo', null);
          }
        }}
      />
      <Button icon={<ClearOutlined />} onClick={resetFilters}>
        Сбросить фильтры
      </Button>
    </Space>
  );
};
