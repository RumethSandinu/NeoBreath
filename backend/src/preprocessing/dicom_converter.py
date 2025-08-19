# ==== Standard Imports ====
import logging
from pathlib import Path

# ==== Third Party Imports ====
import numpy as np
import torch
from pydicom import dcmread


def save_pet_sequences(output_path: Path, patient_id: str, sequences: list, label: str):
    """
    Saves multiple fixed-length PET sequences for a single patient as PyTorch tensors.
    
    Args:
    :param output_path: Path to save the sequences.
    :param patient_id: Patient identifier.
    :param sequences: List of 3D sequences (N, 3, 256, 256).
    :param label: Disease category name.
    """
    logger = logging.getLogger('PreprocessingLogger')
    
    # Create disease-specific output directory
    disease_output_path = output_path / label
    disease_output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        for seq_idx, sequence in enumerate(sequences):

            # Use zero-padded sequence numbering: seq_000
            filename = f'{patient_id}_pet_seq_{seq_idx:03d}.pt'
            file_path = disease_output_path / filename
            
            # Convert to PyTorch tensor and save
            tensor_data = {'label': label, 'PET': torch.from_numpy(sequence).float()}
            torch.save(tensor_data, file_path)
            
            logger.info(f'PET sequence {seq_idx+1}/{len(sequences)} saved: {filename} (shape: {sequence.shape})')

    except Exception as e:
        logger.error(f'Error saving PET sequences for patient {patient_id}: {e}')


def save_ct_sequences(output_path: Path, patient_id: str, sequences: list, label: str):
    """
    Saves multiple fixed-length CT sequences for a single patient as PyTorch tensors.
    
    Args:
    :param output_path: Path to save the sequences.
    :param patient_id: Patient identifier.
    :param sequences: List of 3D sequences (N, 3, 256, 256).
    :param label: Disease category name.
    """
    logger = logging.getLogger('PreprocessingLogger')
    
    # Create disease-specific output directory
    disease_output_path = output_path / label
    disease_output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        for seq_idx, sequence in enumerate(sequences):

            # Use zero-padded sequence numbering: seq_000
            filename = f'{patient_id}_ct_seq_{seq_idx:03d}.pt'
            file_path = disease_output_path / filename
            
            # Convert to PyTorch tensor and save
            tensor_data = {'label': label, 'CT': torch.from_numpy(sequence).float()}
            torch.save(tensor_data, file_path)
            
            logger.info(f'CT sequence {seq_idx+1}/{len(sequences)} saved: {filename} (shape: {sequence.shape})')

    except Exception as e:
        logger.error(f'Error saving CT sequences for patient {patient_id}: {e}')


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

                # Get z-position from ImagePositionPatient or SliceLocation
                z_pos = self._get_z_position(ds)
                slices.append((z_pos, ds.pixel_array, ds))

            except Exception as e:
                self.logger.warning(f'Skipped {file.name}: {e}')

        # Sort by z-position instead of InstanceNumber
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
            # Try ImagePositionPatient
            if hasattr(dicom_dataset, 'ImagePositionPatient') and dicom_dataset.ImagePositionPatient:
                return float(dicom_dataset.ImagePositionPatient[2])
            
        except (AttributeError, IndexError, TypeError):
            pass
        
        try:
            # Try SliceLocation as fallback
            if hasattr(dicom_dataset, 'SliceLocation') and dicom_dataset.SliceLocation is not None:
                return float(dicom_dataset.SliceLocation)
            
        except (AttributeError, TypeError):
            pass
        
        # Final fallback to InstanceNumber
        try:
            if hasattr(dicom_dataset, 'InstanceNumber'):
                return float(dicom_dataset.InstanceNumber)
            
        except (AttributeError, TypeError):
            pass
        
        # If all else fails, return 0
        self.logger.warning("Could not determine z-position, using 0 as fallback")
        return 0.0

    @staticmethod
    def to_3d_array(slices: list) -> np.ndarray:
        """
        Converts a list of 2D image arrays into a 3D shape.

        Args:
        :param slices: The list of images to be converted.
        :return: A 3D image array.
        """
        return np.stack(slices, axis=0)