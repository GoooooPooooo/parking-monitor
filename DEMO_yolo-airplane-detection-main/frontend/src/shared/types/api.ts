export interface ApiError {
  message: string;
  status: number;
  data?: unknown;
}

export interface ApiResponse<T = unknown> {
  data: T;
  status: number;
}
