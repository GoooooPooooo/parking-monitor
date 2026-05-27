import { useState } from 'react';
import { useDebounce } from '../../../shared/lib/hooks';

export interface FilterHistoryState {
  search: string;
  dateFrom: string | null;
  dateTo: string | null;
  model: string | null;
}

export const useFilterHistory = () => {
  const [filters, setFilters] = useState<FilterHistoryState>({
    search: '',
    dateFrom: null,
    dateTo: null,
    model: null,
  });

  const debouncedSearch = useDebounce(filters.search, 300);

  const updateFilter = (key: keyof FilterHistoryState, value: any) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const resetFilters = () => {
    setFilters({
      search: '',
      dateFrom: null,
      dateTo: null,
      model: null,
    });
  };

  return {
    filters,
    debouncedSearch,
    updateFilter,
    resetFilters,
  };
};
