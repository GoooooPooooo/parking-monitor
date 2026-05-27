import { api } from '../../../shared/api/client';
import { API_ENDPOINTS } from '../../../shared/config/api';
import type { DetectionModel } from '../../../shared/types';

export interface MLflowModel {
  run_id: string;
  run_name: string;
  model_type: string;
  epochs: string | number;
  mAP50: number;
  accuracy: number;
  top5_accuracy: number;
  is_classifier: boolean;
  status: string;
  created_at: string;
  has_pt: boolean;
  has_onnx: boolean;
  isDownloaded?: boolean;
}

export const modelApi = {
  getModels: async () => {
    const response = await api.get<{ models: DetectionModel[] }>(API_ENDPOINTS.MODELS);
    return response.data.models;
  },

  getMlflowModels: async () => {
    const response = await api.get<{ models: MLflowModel[] }>('/api/mlflow/models/');
    return response.data.models;
  },

  downloadMlflowModel: async (runId: string, modelName: string) => {
    const response = await api.post<{ status: string; model_name: string; path: string; format: string }>(
      '/api/mlflow/download/',
      { run_id: runId, model_name: modelName }
    );
    return response.data;
  },
};

