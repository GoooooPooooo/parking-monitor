export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  class: string;
}

export interface DetectionResult {
  id: string;
  image_url: string;
  result_image_url: string;
  detections: BoundingBox[];
  created_at: string;
  model_name: string;
}

export interface DetectionModel {
  id: string;
  name: string;
  description: string;
  path: string;
  _type?: 'detector' | 'classifier';
}

export interface ClassificationResult {
  aircraft_type: string;
  confidence: number;
}
