import { api } from '../../../shared/api/client';
import type { HistoryItem } from '../model';

export const historyApi = {
  getHistory: async () => {
    const response = await api.get<{ results: HistoryItem[] }>('/api/history/');
    return response.data.results;
  },
  deleteResult: async (id: string) => {
    const response = await api.delete(`/api/results/${id}/delete/`);
    return response.data;
  },
};
