import { create } from 'zustand';
import type { ModelState } from './types';
import type { DetectionModel } from '../../../shared/types';

interface ModelActions {
  setModels: (models: DetectionModel[]) => void;
  setSelectedModel: (modelId: string, modelType?: 'detector' | 'classifier') => void;
  setLoading: (loading: boolean) => void;
}

export const useModelStore = create<ModelState & ModelActions>((set, get) => ({
  models: [],
  selectedModel: null,
  isLoading: false,

  setModels: (models) => set({ models }),
  setSelectedModel: (modelId, modelType = 'detector') => {
    const models = get().models;
    const model = models.find(m => m.id === modelId);
    if (model) {
      set({ selectedModel: { ...model, _type: modelType } });
    } else if (modelId.startsWith('mlflow:')) {
      // MLflow model: create a synthetic model object
      const runName = modelId.split(':')[1];
      set({
        selectedModel: {
          id: modelId,
          name: `MLflow: ${runName}`,
          description: 'Model from MLflow',
          _type: modelType,
        }
      });
    } else {
      set({ selectedModel: null });
    }
  },
  setLoading: (loading) => set({ isLoading: loading }),
}));
