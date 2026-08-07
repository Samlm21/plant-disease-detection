import { HistoryItem } from '@/types/prediction';

const HISTORY_STORAGE_KEY = 'agrivision_inference_history';

export const historyService = {
  /**
   * Retrieves all past predictions from local storage.
   */
  getAll: (): HistoryItem[] => {
    try {
      const data = localStorage.getItem(HISTORY_STORAGE_KEY);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Failed to parse history from local storage', error);
      return [];
    }
  },

  /**
   * Adds a new prediction result to the local storage history.
   */
  add: (item: HistoryItem): void => {
    try {
      const history = historyService.getAll();
      // Add the newest item to the beginning of the array
      history.unshift(item);
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
    } catch (error) {
      console.error('Failed to save history to local storage', error);
    }
  },

  /**
   * Deletes a specific prediction by ID.
   */
  delete: (id: string): void => {
    const history = historyService.getAll();
    const filtered = history.filter(item => item.id !== id);
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(filtered));
  },

  /**
   * Wipes the entire history.
   */
  clear: (): void => {
    localStorage.removeItem(HISTORY_STORAGE_KEY);
  }
};