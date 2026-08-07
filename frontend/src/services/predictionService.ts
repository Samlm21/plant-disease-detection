import { apiClient } from './apiClient';
import { PredictionResponse } from '@/types/prediction';

export const predictionService = {
  /**
   * Sends an image to the backend for disease prediction.
   */
  predict: async (
    file: File, 
    modelType: 'resnet18' | 'baseline' = 'resnet18',
    enableGradCam: boolean = true
  ): Promise<PredictionResponse> => {
    
    // We must use FormData to send binary image files
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelType);
    formData.append('enable_gradcam', enableGradCam.toString());

    const response = await apiClient.post<PredictionResponse>('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  }
};