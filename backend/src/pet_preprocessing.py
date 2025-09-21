# ==== Standard Imports ====
from pathlib import Path
import logging
from skimage.transform import resize

# ==== Local Project Imports ====
from preprocessing.dicom_converter import DicomConverter, save_pet_volume
from preprocessing.intensity_processing import IntensityProcessor
from preprocessing.volume_processing import VolumeProcessor
from utils.logger import setup_logger


def preprocess_pet_patient_data(output_path: Path, dataset_dir: Path, disease_code: str, logger: logging.Logger, threshold: float, max: bool=True):
    """
    Process PET scan and save as volumes with PyTorch tensors.

    Implements complete preprocessing pipeline:
    1. Read DICOM -> list of 2D slices
    2. Sort slices by z-position
    3. Stack into 3D volume
    4. Convert to SUV
    5. Normalize to [0,1]
    6. Trim sequences using intensity threshold value to avoid legs/head
    7. Final shape: (N, 256, 256) where N is slice count
    8. Save as a PyTorch tensor
    
    Args:
    :param output_path: Path to save the processed volume.
    :param dataset_dir: Path to the patient's DICOM directory.
    :param logger: Logger instance to track processing.
    :param disease_code: Disease code letter (A, B, E, G).
    :param threshold: Intensity threshold for volume trimming.
    :param max: True, save the slices >= intensity threshold.
    """
    
    patient_id = dataset_dir.name
    logger.info(f'=====< PRE-PROCESSING PET PATIENT {patient_id} FROM {disease_code} >=====')

    try:
        # check for DICOM files
        dicom_files = list(dataset_dir.glob('*.dcm'))
        
        if not dicom_files:
            logger.info(f'Skipping PET patient {patient_id} due to missing DICOM files.')
            return

        # convert DICOM slices to 2D NumPy arrays and sort by z-position
        slices = DicomConverter().to_2d_array(dataset_dir)

        # convert to SUV and apply normalization for PET scans
        slices = IntensityProcessor(slices, True).convert() 

        # stack the 2D NumPy arrays to a 3D shape
        volume = DicomConverter.to_3d_array(slices)
        
        logger.info(f'Original volume shape: {volume.shape}')
        
        # trim volume using intensity threshold
        volume_processor = VolumeProcessor(volume)
        trimmed_volume = volume_processor.trim_volume_by_threshold(
            intensity_threshold=threshold,
            min_slices_to_keep=20,      # Keep minimum 20 slices for analysis
            max_mode=max
        )
        logger.info(f'Trimmed volume shape with threshold {threshold}: {trimmed_volume.shape}')
        
        # Save the trimmed volume as single PyTorch tensor using disease code
        save_pet_volume(output_path, patient_id, trimmed_volume, disease_code)
        logger.info(f'---------- Successfully processed PET patient {patient_id} from {disease_code} ----------')
    except Exception as e:
        logger.error(f'Error processing PET patient {patient_id} from {disease_code}: {e}')
        raise


def main():
    """Main function to run PET preprocessing with multiple threshold values."""

    # setup logger
    logger = setup_logger(Path('backend/src/logs'), 'pet_preprocessing.log', 'PreprocessingLogger')
    
    # define paths for PET processing
    pet_dicom_path = Path('data/raw/PET')
    pet_base_output_path = Path('data/preprocessed/PET')
    
    # threshold values to test
    threshold_values = [0.5, 0.6, 0.7, 0.8]
    logger.info(f'=====< STARTING PET DATA PREPROCESSING WITH INTENSITY THRESHOLDS {threshold_values} >=====')
    
    # process PET images with different thresholds
    for threshold in threshold_values:
        logger.info(f'===== PROCESSING WITH THRESHOLD {threshold} =====')
        
        # create threshold-specific output directory
        pet_output_path = pet_base_output_path / f'threshold_{threshold}'
        pet_output_path.mkdir(parents=True, exist_ok=True)
        
        # process PET images
        for disease_dir in pet_dicom_path.iterdir():
            if disease_dir.is_dir() and not disease_dir.name.startswith('.'):
                disease_code = disease_dir.name
                logger.info(f'Processing PET disease: {disease_code} with threshold {threshold}')
                
                # iterate through patient directories within each disease
                for patient_dir in disease_dir.iterdir():
                    if patient_dir.is_dir() and not patient_dir.name.startswith('.'):
                        preprocess_pet_patient_data(pet_output_path, patient_dir, disease_code, logger, threshold, max=True)
    
    logger.info(f'=====< PET DATA PREPROCESSING COMPLETED FOR ALL THRESHOLDS >=====')
    
    # log summary of results
    logger.info('===== PREPROCESSING SUMMARY =====')
    for threshold in threshold_values:
        threshold_path = pet_base_output_path / f'threshold_{threshold}'

        if threshold_path.exists():
            total_files = len(list(threshold_path.rglob('*.pt')))
            logger.info(f'Threshold {threshold}: {total_files} processed volumes')
        else:
            logger.info(f'Threshold {threshold}: No output directory found')


# ========== Runnable ==========
if __name__ == '__main__':
    main()
