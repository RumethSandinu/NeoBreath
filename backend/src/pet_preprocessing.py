# ==== Standard Imports ====
from pathlib import Path
import logging
import numpy as np

# ==== Local Project Imports ====
from preprocessing.dicom_converter import DicomConverter, save_pet_sequences
from preprocessing.intensity_processing import IntensityProcessor
from preprocessing.volume_processing import VolumeProcessor
from utils.logger import setup_logger
from skimage.transform import resize


def preprocess_pet_patient_data(output_path: Path, dataset_dir: Path, disease_name: str, logger: logging.Logger):
    """
    Process a single PET patient's DICOM files and save as fixed-length sequences with PyTorch tensors.
    Implements complete preprocessing pipeline:
    1. Read DICOM → list of 2D slices
    2. Sort slices by z-position
    3. Stack into 3D volume
    4. Convert to SUV
    5. Normalize to [0,1]
    6. Resize every slice to 256x256
    7. Convert every slice to shape (3,256,256) by repeating channel
    8. Break into fixed-length sequences with padding
    9. Final shape per sequence: (N, 3, 256, 256)
    10. Save as PyTorch tensors
    
    Args:
    :param output_path: Path to save the processed sequences.
    :param dataset_dir: Path to the patient's DICOM directory.
    :param disease_name: Name of the disease category.
    :param logger: Logger instance for tracking processing.
    """
    patient_id = dataset_dir.name
    
    logger.info(f'=====< PRE-PROCESSING PET PATIENT {patient_id} FROM {disease_name} >=====')

    try:
        # Check for DICOM files
        dicom_files = list(dataset_dir.glob('*.dcm'))
        
        if not dicom_files:
            logger.info(f'Skipping PET patient {patient_id} due to missing DICOM files.')
            return

        # Convert DICOM slices to 2D NumPy arrays and sort by z-position
        slices = DicomConverter().to_2d_array(dataset_dir)

        # Convert to SUV and apply normalization for PET scans
        slices = IntensityProcessor(slices, True).convert() 

        # Stack the 2D NumPy arrays to a 3D shape
        volume = DicomConverter.to_3d_array(slices)
        
        logger.info(f'Original volume shape: {volume.shape}')

        # Resize every slice to 256x256
        resized_volume = np.zeros((volume.shape[0], 256, 256), dtype=np.float32)
        for i in range(volume.shape[0]):
            resized_volume[i] = resize(volume[i], (256, 256), preserve_range=True, anti_aliasing=True)
        
        logger.info(f'Resized volume shape: {resized_volume.shape}')
        
        # Convert every slice to 3-channels
        volume_3_channels = VolumeProcessor.convert_slices_to_3_channels(resized_volume)
        logger.info(f'3-channel volume shape: {volume_3_channels.shape}')
        
        # Break volumes into fixed-length sequences
        volume_processor = VolumeProcessor(volume_3_channels)
        sequences_3_channels = volume_processor.break_into_sequences(sequence_length=16)
        
        logger.info(f'Created {len(sequences_3_channels)} sequences of shape {sequences_3_channels[0].shape if sequences_3_channels else "N/A"}')
        
        # Save all sequences as PyTorch tensors
        save_pet_sequences(output_path, patient_id, sequences_3_channels, disease_name)
        
        logger.info(f'---------- Successfully processed PET patient {patient_id} from {disease_name} ----------')

    except Exception as e:
        logger.error(f'Error processing PET patient {patient_id} from {disease_name}: {e}')
        raise


def main():
    """Main function to run PET preprocessing."""

    # Setup logger
    logger = setup_logger(Path('backend/src/logs'), 'pet_preprocessing.log', 'PreprocessingLogger')
    
    # Define paths for PET processing
    pet_dicom_path = Path('data/raw/PET')
    pet_output_path = Path('data/preprocessed/PET')
    
    logger.info(f'=====< STARTING PET DATA PREPROCESSING >=====')
    
    # Process PET images
    for disease_dir in pet_dicom_path.iterdir():
        if disease_dir.is_dir() and not disease_dir.name.startswith('.'):
            disease_name = disease_dir.name
            logger.info(f'Processing PET disease: {disease_name}')
            
            # Iterate through patient directories within each disease
            for patient_dir in disease_dir.iterdir():
                if patient_dir.is_dir() and not patient_dir.name.startswith('.'):
                    preprocess_pet_patient_data(pet_output_path, patient_dir, disease_name, logger)
    
    logger.info(f'=====< PET DATA PREPROCESSING COMPLETED >=====')


# ========== Runnable ==========
if __name__ == '__main__':
    main()
