import torch
import librosa as lib
import numpy as np
import pathlib
from glob import glob as glob
# from tqdm import tqdm
import os
from PIL import Image

CWD = pathlib.Path(os.getcwd())

def get_spectrogram(audio_file: str) -> str:
    """
    Extracts a CQT spectrogram from a given audio file.
    Args:
        audio_file (str): The path to the audio file to extract the spectrogram from.
    Returns:
        np.ndarray: A 2D numpy array representing the CQT spectrogram.
    """
    audio_file: pathlib.Path = pathlib.Path(audio_file)
    # Load audio file
    signal, sr = lib.load(str(audio_file.absolute()))
    # Compute the Constant-Q Transform (CQT) with higher resolution
    cqt = lib.cqt(signal, sr=sr, n_bins=120, bins_per_octave=24)
    # Convert the CQT to a spectrogram
    cqt_spectrogram = np.abs(cqt)
    # Convert the spectrogram to decibels
    cqt_spectrogram_db = lib.amplitude_to_db(cqt_spectrogram, ref=np.max)
    # Normalize the spectrogram to be between 0 and 255 for image representation
    cqt_spectrogram_db = 255 * (cqt_spectrogram_db - np.min(cqt_spectrogram_db)) / (np.max(cqt_spectrogram_db) - np.min(cqt_spectrogram_db))
    cqt_spectrogram_db = cqt_spectrogram_db.astype(np.uint8)

    save_dir = CWD / 'spectrograms'
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save the spectrogram as a black and white image
    image = Image.fromarray(cqt_spectrogram)
    image = image.convert('L')  # Convert to grayscale
    image_path = save_dir / f'{audio_file.stem}_spec.png'
    image.save(str(image_path))
    return image_path

def get_device():
    """Get the appropriate device for training"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device