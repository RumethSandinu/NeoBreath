# imports
from pathlib import Path
import logging
from preprocessing.dicom_converter import DicomConverter, save_pet_volume
from preprocessing.intensity_processing import IntensityProcessor
from preprocessing.volume_processing import VolumeProcessor
from utils.logger import setup_logger


def preprocess_pet_patient_data(output_path: Path, dataset_dir: Path, disease_code: str, logger: logging.Logger, threshold: float, max_mode: bool=True):
    """
    Process PET scan and save volumes as NumPy arrays.

    Implements complete preprocessing pipeline:
    1. DICOM to list of 2D slices.
    2. Sort slices by z-position.
    3. Stack into 3D volume.
    4. Convert to SUV.
    5. Normalize to [0,1].
    6. Trim sequences using intensity threshold value to avoid legs/head.
    7. Final shape: (N, 128, 128) where N is slice count.
    8. Save as NumPy arrays.
    
    Args:
    :param output_path: Path to save the processed volume.
    :param dataset_dir: Path to the patient's DICOM directory.
    :param logger: Logger instance to track processing.
    :param disease_code: Disease code letter (A, B, G).
    :param threshold: Intensity threshold for volume trimming.
    :param max_mode: True, keep slices >= intensity threshold (trim legs/head); False keeps < threshold.
    """
    
    patient_id = dataset_dir.name
    logger.info(f'=====< PRE-PROCESSING PET PATIENT {patient_id} FROM {disease_code} >=====')

    try:
        # check for any DICOM files
        if not any(dataset_dir.rglob('*.dcm')):
            logger.info(f'Skipping PET patient {patient_id} due to missing DICOM files.')
            return

        # convert DICOM slices to 2D NumPy arrays and sort by z-position
        slices = DicomConverter().to_2d_array(dataset_dir)

        # convert to SUV, defer normalization until after stacking for consistent thresholds
        slices = IntensityProcessor(slices, normalize=False).convert()

        # stack the 2D NumPy arrays to a 3D shape and resize to 128x128
        volume = DicomConverter.to_3d_array(slices, target_size=128)
        
        # normalize entire volume to [0,1] and cast to float32 for consistency and storage efficiency
        volume = IntensityProcessor._normalize(volume).astype('float32')
        
        logger.info(f'Volume shape after stacking and resizing: {volume.shape}')
        
        # trim volume using intensity threshold
        volume_processor = VolumeProcessor(volume)
        trimmed_volume = volume_processor.trim_volume_by_threshold(
            intensity_threshold=threshold,
            # keep minimum 8 slices for analysis
            min_slices_to_keep=8,
            max_mode=max_mode
        )
        logger.info(f'Trimmed volume shape with threshold {threshold}: {trimmed_volume.shape}')
        
        # save the trimmed volume as sequence using disease code
        save_pet_volume(output_path, patient_id, trimmed_volume.astype('float32'), disease_code)
        logger.info(f'---------- Successfully processed PET patient {patient_id} from {disease_code} ----------')

    except Exception as e:
        logger.error(f'Error processing PET patient {patient_id} from {disease_code}: {e}')
        raise


def main():
    """Main function to run PET preprocessing with multiple threshold values."""

    # set to True for up_threshold, False for down_threshold
    max_mode = False

    # setup logger
    logger = setup_logger(Path('backend/src/logs'), 'pet_preprocessing.log', 'PreprocessingLogger')
    
    # define paths for PET processing
    pet_dicom_path = Path('data/raw/PET')
    pet_base_output_path = Path('data/preprocessed/PET')
    
    # threshold values to test
    threshold_values = [0.5, 0.6, 0.7, 0.8]
    
    logger.info(f'=====< STARTING PET DATA PREPROCESSING WITH INTENSITY THRESHOLDS {threshold_values} >=====')
    logger.info(f'Mode: {"PROCESSING IMAGES WITH HIGHER INTENSITY VALUES" if max_mode else "PROCESSING IMAGES WITH LOWER INTENSITY VALUES"} threshold')
    
    # process PET images with different thresholds
    for threshold in threshold_values:
        logger.info(f'===== PROCESSING WITH THRESHOLD {threshold} =====')
        
        # create threshold-specific output directory based on max_mode
        prefix = 'up' if max_mode else 'down'
        pet_output_path = pet_base_output_path / f'{prefix}_threshold_{threshold}'
        pet_output_path.mkdir(parents=True, exist_ok=True)
        
        # process PET images
        for disease_dir in pet_dicom_path.iterdir():
            if disease_dir.is_dir() and not disease_dir.name.startswith('.'):
                disease_code = disease_dir.name
                logger.info(f'Processing PET disease: {disease_code} with threshold {threshold}')
                
                # iterate through patient directories within each disease
                for patient_dir in disease_dir.iterdir():
                    if patient_dir.is_dir() and not patient_dir.name.startswith('.'):
                        preprocess_pet_patient_data(pet_output_path, patient_dir, disease_code, logger, threshold, max_mode=max_mode)
    
    logger.info(f'=====< PET DATA PREPROCESSING COMPLETED FOR ALL THRESHOLDS >=====')
    
    # log summary of results
    logger.info('===== PREPROCESSING SUMMARY =====')
    for threshold in threshold_values:
        prefix = 'up' if max_mode else 'down'
        threshold_path = pet_base_output_path / f'{prefix}_threshold_{threshold}'

        if threshold_path.exists():
            total_files = len(list(threshold_path.rglob('*.npy')))
            logger.info(f'{prefix.capitalize()} Threshold {threshold}: {total_files} processed volumes')
        else:
            logger.info(f'{prefix.capitalize()} Threshold {threshold}: No output directory found')


# runnable
if __name__ == '__main__':
    main()
