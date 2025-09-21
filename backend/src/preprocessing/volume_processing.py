# ==== Standard Imports ====
import logging

# ==== Third Party Imports ====
import numpy as np


class VolumeProcessor:
    """
    A utility class for preprocessing 3D medical image volumes.
    The class provides depth dimension manipulation through
    cropping and padding.
    """

    def __init__(self, input_array: np.ndarray):
        """
        Initializes VolumeProcessor object.

        Args:
        :param input_array: The array shape to be preprocessed.
        """
        self.input_array = input_array
        self.logger = logging.getLogger('PreprocessingLogger')


    def trim_volume_by_threshold(self, intensity_threshold: float = 0.1, min_slices_to_keep: int = 20, max_mode: bool = True) -> np.ndarray:
        """
        Trim the volume from beginning and end to remove slices with low anatomical content
        (like legs and head in PET/CT scans) based on intensity threshold.
        
        Args:
            intensity_threshold: Threshold for mean intensity to determine if slice contains relevant anatomy
            min_slices_to_keep: Minimum number of slices to keep even if below threshold
        
        Returns:
            Trimmed volume
        """
        volume = self.input_array
        
        # 3D (N, H, W)
        depth = volume.shape[0]
        slice_intensities = np.mean(volume, axis=(1, 2))
        
        # normalize intensities to 0-1 range for consistent thresholding
        if slice_intensities.max() > slice_intensities.min():
            normalized_intensities = (slice_intensities - np.min(slice_intensities)) / (np.max(slice_intensities) - np.min(slice_intensities) + 1e-8)

        else:
            # all slices have same intensity - keep all
            normalized_intensities = np.ones_like(slice_intensities)
        
        # find first and last slice meeting threshold criteria based on max_mode
        if max_mode:
            # keep slices >= threshold - find first slice above threshold from beginning
            start_idx = 0
            for i in range(depth):
                if normalized_intensities[i] >= intensity_threshold:
                    start_idx = i
                    break
            
            # find last slice above threshold from end
            end_idx = depth - 1
            for i in range(depth - 1, -1, -1):
                if normalized_intensities[i] >= intensity_threshold:
                    end_idx = i
                    break
        else:
            # keep slices < threshold - find first slice below threshold from beginning
            start_idx = 0
            for i in range(depth):
                if normalized_intensities[i] < intensity_threshold:
                    start_idx = i
                    break
            
            # find last slice below threshold from end
            end_idx = depth - 1
            for i in range(depth - 1, -1, -1):
                if normalized_intensities[i] < intensity_threshold:
                    end_idx = i
                    break
        
        # ensure we keep minimum number of slices
        total_slices = end_idx - start_idx + 1
        if total_slices < min_slices_to_keep:

            # expand range to keep minimum slices, centered if possible
            center = (start_idx + end_idx) // 2
            half_min = min_slices_to_keep // 2
            start_idx = max(0, center - half_min)
            end_idx = min(depth - 1, start_idx + min_slices_to_keep - 1)
            self.logger.info(f'Expanded range to meet minimum slice requirement')
        
        # trim the volume
        trimmed_volume = volume[start_idx:end_idx + 1]
        return trimmed_volume