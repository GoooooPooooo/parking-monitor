export const API_ENDPOINTS = {
  UPLOAD: '/api/upload/',
  DETECT: '/api/detect/',
  DETECT_AND_CLASSIFY: '/api/detect-and-classify/',
  CLASSIFY_ONLY: '/api/classify-only/',
  RESULTS: (id: string) => `/api/results/${id}/`,
  MODELS: '/api/models/',
  HISTORY: '/api/history/',
  MLFLOW_METRICS: '/api/mlflow/metrics/',
} as const;
