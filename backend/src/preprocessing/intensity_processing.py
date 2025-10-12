# ==== Standard Imports ====
import logging

# ==== Third Party Imports ====
import numpy as np


class IntensityProcessor:
    """
    A class to process image intensities for PET and CT modalities.
    This class provides functions for converting intensity values to 
    Hounsfield Units (HU) and Standard Uptake Value (SUV). 
    A normalization function is also included to normalize the intensity
    values.
    """

    def __init__(self, slices: list, normalize: bool = True):
        """
        Initializes IntensityProcessor object.

        Args:
        :param slices: A list of slices and their corresponding metadata.
        :param normalize: Flag indicating if normalization should be applied.
        """
        self.slices = slices
        self.normalize = normalize
        self.logger = logging.getLogger('PreprocessingLogger')


    def convert(self) -> list:
        """
        Converts the intensity of each slice depending on its modality.
        If normalization is true, images are normalized.

        :return: A list containing the processed array images.
        """
        converted_slices = []
        for image, metadata in self.slices:
            if metadata.Modality == 'CT':
                converted_img = self._convert_to_hu(image, metadata)
            else:
                converted_img = self._convert_to_suv(image, metadata)

            if self.normalize:
                converted_img = self._normalize(converted_img)

            converted_slices.append(converted_img)
        return converted_slices


    def _convert_to_hu(self, image: np.ndarray, metadata) -> np.ndarray:
        """
        Converts the image intensity to Hounsfield Units (HU) based
        on the DICOM metadata and clips values to [-1000, 400] range.

        Args:
        :param image: The image array to be converted.
        :param metadata: A DICOM metadata associated with the image.
        :return: An image converted into Hounsfield Units and clipped.
        """
        try:
            # check if RescaleSlope and RescaleIntercept exist
            if hasattr(metadata, 'RescaleSlope') and hasattr(metadata, 'RescaleIntercept'):
                rescale_slope = metadata.RescaleSlope
                rescale_intercept = metadata.RescaleIntercept
                hu = image * rescale_slope + rescale_intercept
            else:
                # if rescale parameters don't exist, assume image is already in HU
                self.logger.warning('RescaleSlope/RescaleIntercept not found, assuming image is already in HU')
                hu = image.astype(np.float32)
            
            # clip CT values to typical soft tissue range [-1000, 400] HU
            hu_clipped = np.clip(hu, -1000, 400)
            return hu_clipped
        except Exception as e:
            self.logger.error(f'Could not convert to HU: {e}')

            # return the original image if conversion fails
            return image.astype(np.float32)


    def _convert_to_suv(self, image: np.ndarray, metadata) -> np.ndarray:
        """
        Convert the image intensity to Standardized Uptake Value (SUV)
        based on the DICOM metadata.

        Args:
        :param image: The image array to be converted.
        :param metadata: A DICOM metadata associated with the image.
        :return: An image converted into Standardized Uptake Value.
        """
        try:
            # check if required metadata exists and is not None
            if not hasattr(metadata, 'PatientWeight') or metadata.PatientWeight is None:
                self.logger.warning('Skipping SUV conversion: PatientWeight is missing or None')
                return image.astype(np.float32)
            
            if not hasattr(metadata, 'RadiopharmaceuticalInformationSequence') or \
               not metadata.RadiopharmaceuticalInformationSequence:
                self.logger.warning('Skipping SUV conversion: RadiopharmaceuticalInformationSequence is missing')
                return image.astype(np.float32)
            
            weight = float(metadata.PatientWeight)
            rph_info = metadata.RadiopharmaceuticalInformationSequence[0]
            
            if not hasattr(rph_info, 'RadionuclideTotalDose') or rph_info.RadionuclideTotalDose is None:
                self.logger.warning('Skipping SUV conversion: RadionuclideTotalDose is missing or None')
                return image.astype(np.float32)
            
            dose = float(rph_info.RadionuclideTotalDose) / 1e6
            
            if dose == 0:
                self.logger.warning('Skipping SUV conversion: RadionuclideTotalDose is zero')
                return image.astype(np.float32)
            
            suv = (image.astype(np.float32) * weight) / dose
            return suv
        
        except Exception as e:
            self.logger.error(f'Could not convert to SUV: {e}')
            return image.astype(np.float32)


    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalizes the image array to a range from 0 to 1.

        Args:
        :param image: The image to be normalized
        :return: A Normalized image
        """
        try:
            return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        except Exception as e:
            logging.getLogger('PreprocessingLogger').error(f'Normalization failed: {e}')
            return image