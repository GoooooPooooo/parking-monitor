import { create } from 'zustand';
import type { ImageState } from './types';

interface ImageActions {
  setUploadedImage: (image: HTMLImageElement) => void;
  setImageUrl: (url: string) => void;
  setFileName: (name: string) => void;
  setUploadedId: (id: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearImage: () => void;
}

export const useImageStore = create<ImageState & ImageActions>((set) => ({
  uploadedImage: null,
  imageUrl: null,
  fileName: null,
  uploadedId: null,
  isLoading: false,
  error: null,

  setUploadedImage: (image) => set({ uploadedImage: image, isLoading: false, error: null }),
  setImageUrl: (url) => set({ imageUrl: url }),
  setFileName: (name) => set({ fileName: name }),
  setUploadedId: (id) => set({ uploadedId: id }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error, isLoading: false }),
  clearImage: () => set({ uploadedImage: null, imageUrl: null, fileName: null, uploadedId: null, error: null }),
}));
