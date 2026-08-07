import axios from 'axios';

// Create a configured Axios instance
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Optional: Add interceptors here later for global error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Network Error:', error);
    return Promise.reject(error);
  }
);