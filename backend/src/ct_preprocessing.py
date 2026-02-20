# ==== Standard Imports ====
from pathlib import Path
import logging

# ==== Local Project Imports ====
from preprocessing.dicom_converter import DicomConverter, save_ct_volume
from preprocessing.intensity_processing import IntensityProcessor
from preprocessing.volume_processing import VolumeProcessor
from utils.logger import setup_logger


def preprocess_ct_patient_data(output_path: Path, dataset_dir: Path, logger: logging.Logger, disease_code: str, threshold: float, max: bool=True):
    """
    Process a single CT patient's DICOM files and save as trimmed volume with PyTorch tensors.
    Implements complete preprocessing pipeline:
    1. Read DICOM → list of 2D slices
    2. Sort slices by z-position  
    3. Stack into 3D volume
    4. Clip values (CT: HU in [-1000,400])
    5. Normalize to [0,1]
    6. Resize every slice to 512x512
    7. Trim sequences using intensity threshold value to avoid legs/head
    8. Final shape: (N, 512, 512) where N is slice count
    9. SSave as numpy array
    
    Args:
    :param output_path: Path to save the processed volume.
    :param dataset_dir: Path to the patient's DICOM directory.
    :param logger: Logger instance for tracking processing.
    :param disease_code: Disease code letter (A, B, E, G).
    :param threshold: Intensity threshold for volume trimming.
    :param max: True, save the slices >= intensity threshold.
    """

    patient_id = dataset_dir.name
    logger.info(f'=====< PRE-PROCESSING CT PATIENT {patient_id} FROM {disease_code} >=====')

    try:
        # check for DICOM files
        dicom_files = list(dataset_dir.glob('*.dcm'))
        
        if not dicom_files:
            logger.info(f'Skipping CT patient {patient_id} due to missing DICOM files.')
            return

        # convert DICOM slices to 2D NumPy arrays and sort by z-position
        slices = DicomConverter().to_2d_array(dataset_dir)

        # apply pixel intensity conversion (HU conversion with clipping) for CT scans
        intensity_processor = IntensityProcessor(slices, False)
        processed_slices = intensity_processor.convert()

        # stack the 2D NumPy arrays to a 3D shape and resize to 512x512
        volume = DicomConverter.to_3d_array(processed_slices, target_size=256)
        logger.info(f'Volume shape after stacking and resizing: {volume.shape}')
        
        # trim volume using intensity threshold
        volume_processor = VolumeProcessor(volume)
        trimmed_volume = volume_processor.trim_volume_by_threshold(
            intensity_threshold=threshold,
            min_slices_to_keep=8,      # keep minimum 20 slices for analysis
            max_mode=max
        )
        logger.info(f'Trimmed volume shape with threshold {threshold}: {trimmed_volume.shape}')
        
        # save the trimmed volume as sequence using disease code
        save_ct_volume(output_path, patient_id, trimmed_volume, disease_code)
        logger.info(f'---------- Successfully processed CT patient {patient_id} from {disease_code} ----------')
    except Exception as e:
        logger.error(f'Error processing CT patient {patient_id} from {disease_code}: {e}')
        raise


def main():
    """Main function to run CT preprocessing with multiple threshold values."""

    max_mode = True  # set to True for up_threshold, False for down_threshold
    
    # setup logger
    logger = setup_logger(Path('backend/src/logs'), 'ct_preprocessing.log', 'PreprocessingLogger')
    
    # define paths for CT processing
    ct_dicom_path = Path('data/raw/CT')
    ct_base_output_path = Path('data/preprocessed/CT')
    
    # threshold values to test
    threshold_values = [0.5]
    
    logger.info(f'=====< STARTING CT DATA PREPROCESSING WITH INTENSITY THRESHOLDS {threshold_values} >=====')
    logger.info(f'Mode: {"PROCESSING IMAGES WITH HIGHER INTENSITY VALUES" if max_mode else "PROCESSING IMAGES WITH HIGHER INTENSITY VALUES"} threshold')
    
    # process CT images with different thresholds
    for threshold in threshold_values:
        logger.info(f'===== PROCESSING WITH THRESHOLD {threshold} =====')
        
        # create threshold-specific output directory based on max_mode
        prefix = 'up' if max_mode else 'down'
        ct_output_path = ct_base_output_path / f'{prefix}_threshold_{threshold}'
        ct_output_path.mkdir(parents=True, exist_ok=True)
        
        # process CT images
        for disease_dir in ct_dicom_path.iterdir():
            if disease_dir.is_dir() and not disease_dir.name.startswith('.'):
                disease_code = disease_dir.name
                logger.info(f'Processing CT disease: {disease_code} with threshold {threshold}')
                
                # iterate through patient directories within each disease
                for patient_dir in disease_dir.iterdir():
                    if patient_dir.is_dir() and not patient_dir.name.startswith('.'):
                        preprocess_ct_patient_data(ct_output_path, patient_dir, logger, disease_code, threshold, max=max_mode)

    logger.info(f'=====< CT DATA PREPROCESSING COMPLETED FOR ALL THRESHOLDS >=====')
    
    # log summary of results
    logger.info('===== PREPROCESSING SUMMARY =====')
    for threshold in threshold_values:
        prefix = 'up' if max_mode else 'down'
        threshold_path = ct_base_output_path / f'{prefix}_threshold_{threshold}'
        
        if threshold_path.exists():
            total_files = len(list(threshold_path.rglob('*.npy')))
            logger.info(f'{prefix.capitalize()} Threshold {threshold}: {total_files} processed volumes')
        else:
            logger.info(f'{prefix.capitalize()} Threshold {threshold}: No output directory found')


# ========== Runnable ==========
if __name__ == '__main__':
    main()
