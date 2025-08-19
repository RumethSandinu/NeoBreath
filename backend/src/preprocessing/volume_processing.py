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

    def break_into_sequences(self, sequence_length: int = 16) -> list:
        """
        Break a 4D volume into multiple fixed-length sequences without overlapping.
        If the last sequence is shorter than sequence_length, it will be zero-padded.
        
        Args:
        :param sequence_length: Target length for each sequence (default: 16).
        
        Returns:
        :return: List of 4D sequences, each with fixed length.
        """
        volume = self.input_array
        current_depth = volume.shape[0]
        sequences = []
        
        self.logger.info(f'Breaking 4D volume of depth {current_depth} into non-overlapping sequences of length {sequence_length}')
        
        # Break into non-overlapping chunks
        start = 0
        while start < current_depth:
            end = min(start + sequence_length, current_depth)
            sequence = volume[start:end]
            
            # If sequence is shorter than target length, pad with zeros
            if sequence.shape[0] < sequence_length:
                padded_sequence = np.zeros((sequence_length, volume.shape[1], volume.shape[2], volume.shape[3]), dtype=volume.dtype)
                padded_sequence[:sequence.shape[0]] = sequence
                sequences.append(padded_sequence)
                self.logger.info(f'Final sequence padded from {sequence.shape[0]} to {sequence_length} slices')

            else:
                sequences.append(sequence)
            
            start += sequence_length
        
        self.logger.info(f'Created {len(sequences)} non-overlapping sequences')
        return sequences

    @staticmethod
    def convert_slices_to_3_channels(volume: np.ndarray) -> np.ndarray:
        """
        Convert each slice in a volume to 3-channel format.
        Changes shape from (N, H, W) to (N, 3, H, W).
        
        Args:
            volume: Input volume with shape (N, H, W)
        
        Returns:
            Volume with shape (N, 3, H, W)
        """
        # Add channel dimension and repeat 3 times
        volume_with_channels = np.expand_dims(volume, axis=1) 
        volume_3_channels = np.repeat(volume_with_channels, 3, axis=1)
        
        return volume_3_channels