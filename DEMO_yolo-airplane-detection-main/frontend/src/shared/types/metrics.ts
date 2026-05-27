export interface MlflowMetric {
  id: string;
  name: string;
  value: number;
  step: number;
  timestamp: string;
  model_name: string;
}

export interface MlflowRun {
  id: string;
  name: string;
  status: string;
  metrics: MlflowMetric[];
  start_time: string;
  end_time: string;
}
