import { api } from '../../../shared/api/client';
import { API_ENDPOINTS } from '../../../shared/config/api';
import type { DetectionResult } from '../../../shared/types';

export const detectionApi = {
  uploadImage: async (file: File) => {
    const formData = new FormData();
    formData.append('image', file);

    const response = await api.post(API_ENDPOINTS.UPLOAD, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  runDetection: async (imageId: string, modelName: string) => {
    const response = await api.post<DetectionResult>(API_ENDPOINTS.DETECT, {
      image_id: imageId,
      model_name: modelName,
    });
    return response.data;
  },

  runDetectionAndClassify: async (imageId: string, modelName: string) => {
    const response = await api.post<DetectionResult>(API_ENDPOINTS.DETECT_AND_CLASSIFY, {
      image_id: imageId,
      model_name: modelName,
    });
    return response.data;
  },

  runClassifyOnly: async (imageId: string, modelName: string) => {
    const response = await api.post<DetectionResult>(API_ENDPOINTS.CLASSIFY_ONLY, {
      image_id: imageId,
      model_name: modelName,
    });
    return response.data;
  },

  getResult: async (id: string) => {
    const response = await api.get<DetectionResult>(API_ENDPOINTS.RESULTS(id));
    return response.data;
  },
};
