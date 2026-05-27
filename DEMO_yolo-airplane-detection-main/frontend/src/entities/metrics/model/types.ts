import type { MlflowRun } from '../../../shared/types';

export interface MetricsState {
  runs: MlflowRun[];
  selectedRun: MlflowRun | null;
  isLoading: boolean;
  error: string | null;
}
