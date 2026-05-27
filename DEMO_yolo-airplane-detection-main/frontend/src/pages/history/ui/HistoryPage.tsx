import React, { useEffect, useMemo, useState } from 'react';
import { Empty, Spin, Pagination } from 'antd';
import { HistoryCollections, useHistoryStore, historyApi } from '../../../entities/history';

export const HistoryPage: React.FC = () => {
  const { items, isLoading, setHistory, setLoading } = useHistoryStore();
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(3);

  useEffect(() => {
    const fetchHistory = async () => {
      setLoading(true);
      try {
        const data = await historyApi.getHistory();
        setHistory(data);
      } catch (error) {
        console.error('Failed to fetch history:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  // Сортировка по дате (новые сверху)
  const sortedItems = useMemo(() => {
    return [...items].sort((a, b) => {
      const dateA = new Date(a.created_at).getTime();
      const dateB = new Date(b.created_at).getTime();
      return dateB - dateA;
    });
  }, [items]);

  // Пагинация
  const paginatedItems = useMemo(() => {
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    return sortedItems.slice(startIndex, endIndex);
  }, [sortedItems, currentPage, pageSize]);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ margin: 0 }}>История детекций ({items.length})</h1>
      </div>

      <Spin spinning={isLoading}>
        {sortedItems.length > 0 ? (
          <>
            <HistoryCollections items={paginatedItems} />
            {sortedItems.length > 0 && (
              <div style={{ marginTop: '24px', textAlign: 'center' }}>
                <Pagination
                  current={currentPage}
                  total={sortedItems.length}
                  pageSize={pageSize}
                  onChange={handlePageChange}
                  onShowSizeChange={(current, size) => {
                    setPageSize(size);
                    setCurrentPage(1);
                  }}
                  showSizeChanger={true}
                  pageSizeOptions={[3, 5, 10, 15, 50]}
                  showTotal={(total, range) => `${range[0]}-${range[1]} из ${total}`}
                />
              </div>
            )}
          </>
        ) : (
          !isLoading && (
            <Empty
              description="История детекций пуста"
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          )
        )}
      </Spin>
    </div>
  );
};
