import { create } from 'zustand';
import type { MetricsState } from './types';
import type { MlflowRun } from '../../../shared/types';

interface MetricsActions {
  setRuns: (runs: MlflowRun[]) => void;
  setSelectedRun: (run: MlflowRun) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useMetricsStore = create<MetricsState & MetricsActions>((set) => ({
  runs: [],
  selectedRun: null,
  isLoading: false,
  error: null,

  setRuns: (runs) => set({ runs }),
  setSelectedRun: (run) => set({ selectedRun: run }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error, isLoading: false }),
}));
