export type SeverityLevel = "low" | "moderate" | "high" | "critical";

export interface Treatment {
  summary: string;
  steps: string[];
  preventiveMeasures: string[];
}

export interface Disease {
  id: string;
  name: string;
  plant: "potato" | "tomato";
  description: string;
  severity: SeverityLevel;
  treatment: Treatment;
}

/** One entry in the top-5 candidate list returned by the model. */
export interface ClassProbability {
  classId: string;
  label: string;
  confidence: number; // 0–1
}

export interface GradCam {
  /** Base64-encoded PNG heatmap overlay */
  heatmapBase64: string;
  /** Optional raw overlay-on-original composite, if backend provides it */
  overlayBase64?: string;
}

export interface PredictionResult {
  id: string;
  predictedClass: string;
  disease: Disease;
  confidence: number; // 0–1, top-1 confidence
  topPredictions: ClassProbability[];
  gradCam: GradCam;
  inferenceTimeMs: number;
  createdAt: string; // ISO timestamp
}

/** Raw shape returned by POST /predict, before any client-side normalization */
export interface PredictionResponse {
  predicted_class: string;
  confidence: number;
  top_5: { class_id: string; label: string; confidence: number }[];
  disease: {
    id: string;
    name: string;
    plant: string;
    description: string;
    severity: SeverityLevel;
  };
  treatment: {
    summary: string;
    steps: string[];
    preventive_measures: string[];
  };
  grad_cam: {
    heatmap_base64: string;
    overlay_base64?: string;
  };
  inference_time_ms: number;
}

export interface UploadRequest {
  file: File;
}

export interface HistoryItem {
  id: string;
  thumbnailBase64: string;
  predictedClass: string;
  confidence: number;
  severity: SeverityLevel;
  createdAt: string;
}

export interface APIError {
  status: number;
  code:
    | "INVALID_IMAGE"
    | "IMAGE_TOO_LARGE"
    | "NETWORK_ERROR"
    | "SERVER_UNAVAILABLE"
    | "TIMEOUT"
    | "PREDICTION_FAILED"
    | "UNKNOWN";
  message: string;
}
