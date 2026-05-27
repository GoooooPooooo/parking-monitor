import { create } from 'zustand';
import type { HistoryState, HistoryItem } from './types';

interface HistoryActions {
  setHistory: (items: HistoryItem[]) => void;
  addHistoryItem: (item: HistoryItem) => void;
  removeItem: (id: string) => void;
  clearHistory: () => void;
  setLoading: (isLoading: boolean) => void;
  /** Удалить последний результат детекции (где есть detections) */
  removeLastDetectionResult: () => void;
}

export const useHistoryStore = create<HistoryState & HistoryActions>((set) => ({
  items: [],
  isLoading: false,
  error: null,

  setHistory: (items) => set({ items, isLoading: false, error: null }),
  addHistoryItem: (item) => set((state) => {
    // Добавляем новый элемент и сортируем по дате (новые сверху)
    const newItems = [item, ...state.items];
    return {
      items: newItems.sort((a, b) => {
        const dateA = new Date(a.created_at).getTime();
        const dateB = new Date(b.created_at).getTime();
        return dateB - dateA;
      }),
    };
  }),
  removeItem: (id) =>
    set((state) => ({
      items: state.items.filter((item) => item.id !== id),
    })),
  clearHistory: () => set({ items: [], error: null, isLoading: false }),
  setLoading: (isLoading) => set({ isLoading }),
  removeLastDetectionResult: () => set((state) => {
    // Находим первый item с detections (результат детекции)
    const detectionIndex = state.items.findIndex(item => item.count > 0);
    if (detectionIndex === -1) return state;
    
    const newItems = [...state.items];
    newItems.splice(detectionIndex, 1);
    return { items: newItems };
  }),
}));
