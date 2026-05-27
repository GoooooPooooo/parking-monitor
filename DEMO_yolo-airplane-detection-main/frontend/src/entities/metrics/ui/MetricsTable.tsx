import React from 'react';
import { Table, Tag } from 'antd';
import type { MlflowRun } from '../../../shared/types';
import { formatDate } from '../../../shared/lib/helpers';

interface MetricsTableProps {
  runs: MlflowRun[];
  loading: boolean;
  onRowClick?: (run: MlflowRun) => void;
}

export const MetricsTable: React.FC<MetricsTableProps> = ({ runs, loading, onRowClick }) => {
  const columns = [
    { title: 'Название', dataIndex: 'name', key: 'name' },
    {
      title: 'Статус',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const color = status === 'FINISHED' ? 'green' : status === 'FAILED' ? 'red' : 'orange';
        return <Tag color={color}>{status}</Tag>;
      },
    },
    {
      title: 'Дата начала',
      dataIndex: 'start_time',
      key: 'start_time',
      render: (time: string) => formatDate(time),
    },
    {
      title: 'Метрики',
      key: 'metrics',
      render: (_: unknown, record: MlflowRun) => (
        <>
          {record.metrics.slice(0, 3).map((metric) => (
            <Tag key={metric.id} color="blue">
              {metric.name}: {metric.value.toFixed(3)}
            </Tag>
          ))}
          {record.metrics.length > 3 && <Tag>+{record.metrics.length - 3}</Tag>}
        </>
      ),
    },
  ];

  return (
    <Table
      columns={columns}
      dataSource={runs}
      loading={loading}
      rowKey="id"
      onRow={(record) => ({
        onClick: () => onRowClick?.(record),
        style: { cursor: onRowClick ? 'pointer' : 'default' },
      })}
    />
  );
};
