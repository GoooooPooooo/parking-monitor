export interface ImageState {
  uploadedImage: HTMLImageElement | null;
  imageUrl: string | null;
  fileName: string | null;
  uploadedId: string | null;
  isLoading: boolean;
  error: string | null;
}
