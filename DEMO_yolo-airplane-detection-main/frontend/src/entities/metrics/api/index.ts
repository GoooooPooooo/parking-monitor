import { api } from '../../../shared/api/client';
import { API_ENDPOINTS } from '../../../shared/config/api';
import type { MlflowRun } from '../../../shared/types';

export const metricsApi = {
  getMetrics: async () => {
    const response = await api.get<MlflowRun[]>(API_ENDPOINTS.MLFLOW_METRICS);
    return response.data;
  },
};
