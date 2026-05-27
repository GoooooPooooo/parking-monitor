export interface HistoryItem {
  id: string;
  image_url: string;
  result_image_url?: string;
  original_url: string;
  detections: Array<{
    bbox: number[];
    confidence: number;
    class_id: number;
    class_name: string;
    classification?: {
      class_id: number;
      class_name: string;
      confidence: number;
    };
  }>;
  count: number;
  model_id: string;
  created_at: string;
}

export interface HistoryState {
  items: HistoryItem[];
  isLoading: boolean;
  error: string | null;
}
