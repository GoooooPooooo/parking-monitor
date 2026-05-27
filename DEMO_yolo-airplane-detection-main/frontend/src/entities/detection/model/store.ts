import { create } from 'zustand';
import type { DetectionState } from './types';
import type { BoundingBox, DetectionResult } from '../../../shared/types';

interface DetectionActions {
  setResult: (result: DetectionResult) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setSelectedBox: (box: BoundingBox | null) => void;
  clearResult: () => void;
}

export const useDetectionStore = create<DetectionState & DetectionActions>((set) => ({
  currentResult: null,
  isLoading: false,
  error: null,
  selectedBox: null,

  setResult: (result) => set({ currentResult: result, isLoading: false, error: null }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error, isLoading: false }),
  setSelectedBox: (box) => set({ selectedBox: box }),
  clearResult: () => set({ currentResult: null, selectedBox: null, error: null }),
}));
