import type { BoundingBox, DetectionResult } from '../../../shared/types';

export interface DetectionState {
  currentResult: DetectionResult | null;
  isLoading: boolean;
  error: string | null;
  selectedBox: BoundingBox | null;
}
