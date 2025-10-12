# ==== Standard Imports ====
import logging
from pathlib import Path

# ==== Third Party Imports ====
import numpy as np
from pydicom import dcmread
from skimage.transform import resize


def save_pet_volume(output_path: Path, patient_id: str, volume: np.ndarray, label: str):
    """
    Saves a single trimmed PET volume for a patient as numpy array.
    
    Args:
    :param output_path: Path to save the volume.
    :param patient_id: Patient identifier.
    :param volume: Single 3D volume (N, H, W).
    :param label: Disease code.
    """
    logger = logging.getLogger('PreprocessingLogger')
    
    # create disease-specific output directory
    disease_output_path = output_path / label
    disease_output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # save as numpy array
        filename = f'{patient_id}.npy'
        file_path = disease_output_path / filename
        
        np.save(file_path, volume)
        logger.info(f'PET volume saved: {filename} (shape: {volume.shape})')
    except Exception as e:
        logger.error(f'Error saving PET volume for patient {patient_id}: {e}')


def save_ct_volume(output_path: Path, patient_id: str, volume: np.ndarray, label: str):
    """
    Saves a single trimmed CT volume for a patient as numpy array.
    
    Args:
    :param output_path: Path to save the volume.
    :param patient_id: Patient identifier.
    :param volume: Single 3D volume (N, H, W).
    :param label: Disease code.
    """
    logger = logging.getLogger('PreprocessingLogger')
    
    # create disease-specific output directory
    disease_output_path = output_path / label
    disease_output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # save as numpy array
        filename = f'{patient_id}.npy'
        file_path = disease_output_path / filename
        
        np.save(file_path, volume)
        logger.info(f'CT volume saved: {filename} (shape: {volume.shape})')
    except Exception as e:
        logger.error(f'Error saving CT volume for patient {patient_id}: {e}')


class DicomConverter:
    """
    A class to handle the conversion of DICOM files into 2D or
    3D image arrays. This class supports loading DICOM files
    from a specified directory, sorting them, and converting
    them into 2D arrays or stacking them into a 3D volume.
    """
    
    def __init__(self):
        """
        Initializes DicomConverter object.
        """
        self.logger = logging.getLogger('PreprocessingLogger')

    def to_2d_array(self, dicom_path: Path) -> list:
        """
        Loads DICOM files from a folder and returns a sorted list of (pixel_array, metadata) tuples.
        Sorts slices by z-position using ImagePositionPatient or SliceLocation.

        Args:
        :param dicom_path: The path containing the .dcm files
        :return: A List of tuples (pixel_array, metadata)
        """
        slices = []
        files = list(dicom_path.glob('*.dcm'))
        self.logger.info(f'Found {len(files)} DICOM files in {dicom_path}')

        for file in files:
            try:
                ds = dcmread(file)

                # get z-position from ImagePositionPatient or SliceLocation
                z_pos = self._get_z_position(ds)
                slices.append((z_pos, ds.pixel_array, ds))
            except Exception as e:
                self.logger.warning(f'Skipped {file.name}: {e}')

        # sort by z-position instead of InstanceNumber
        slices.sort(key=lambda x: x[0])
        self.logger.info(f'Successfully loaded and sorted {len(slices)} slices by z-position.')
        return [(pixel_array, metadata) for _, pixel_array, metadata in slices]
    

    def _get_z_position(self, dicom_dataset):
        """
        Extract z-position from DICOM dataset.
        Tries ImagePositionPatient first, then SliceLocation, then InstanceNumber as fallback.
        
        Args:
        :param dicom_dataset: DICOM dataset
        :return: Z-position value
        """
        try:
            # try ImagePositionPatient
            if hasattr(dicom_dataset, 'ImagePositionPatient') and dicom_dataset.ImagePositionPatient:
                return float(dicom_dataset.ImagePositionPatient[2])
            
        except (AttributeError, IndexError, TypeError):
            pass
        
        try:
            # try SliceLocation as fallback
            if hasattr(dicom_dataset, 'SliceLocation') and dicom_dataset.SliceLocation is not None:
                return float(dicom_dataset.SliceLocation)
            
        except (AttributeError, TypeError):
            pass
        
        # final fallback to InstanceNumber
        try:
            if hasattr(dicom_dataset, 'InstanceNumber'):
                return float(dicom_dataset.InstanceNumber)
            
        except (AttributeError, TypeError):
            pass
        
        # if all else fails, return 0
        self.logger.warning("Could not determine z-position, using 0 as fallback")
        return 0.0


    @staticmethod
    def to_3d_array(slices: list, target_size: int = 256) -> np.ndarray:
        """
        Converts a list of 2D image arrays into a 3D shape.
        Resizes all slices to the target size (default 256x256) to ensure consistency.

        Args:
        :param slices: The list of images to be converted.
        :param target_size: Target size for each slice (default 256).
        :return: A 3D image array with shape (N, target_size, target_size).
        """
        if not slices:
            raise ValueError("No slices provided")
            
        logger = logging.getLogger('PreprocessingLogger')
        resized_slices = []
        
        for i, slice_img in enumerate(slices):
            if slice_img.shape == (target_size, target_size):
                # slice already has correct size
                resized_slices.append(slice_img)
            else:
                # resize slice to target size
                logger.info(f'Resizing slice {i} from {slice_img.shape} to ({target_size}, {target_size})')
                resized_slice = resize(slice_img, (target_size, target_size), 
                                     preserve_range=True, anti_aliasing=True)
                resized_slices.append(resized_slice)
        
        logger.info(f'Stacking {len(resized_slices)} slices into 3D volume with shape ({len(resized_slices)}, {target_size}, {target_size})')
        return np.stack(resized_slices, axis=0)