import type { DetectionModel } from '../../../shared/types';

export interface ModelState {
  models: DetectionModel[];
  selectedModel: DetectionModel | null;
  isLoading: boolean;
}
