import librosa as lib
import numpy as np
import time
import pathlib
from glob import glob as glob
from tqdm import tqdm
import os
from PIL import Image

start = time.time()

total_len_audio_total = 0
total_len_audio_extracted = 0

def get_len_audio_total(audio_files: list[str]) -> float:
    """
    Returns the total length of all audio files in seconds.
    """
    global total_len_audio_total
    total_len_audio_total = sum([lib.get_duration(y=lib.load(str(audio_file.absolute()))[0], sr=lib.load(str(audio_file.absolute()))[1]) for audio_file in audio_files])
    return total_len_audio_total

def extract_cqt_spectrogram(audio_file: pathlib.Path) -> tuple[np.ndarray, int]:
    """
    Extracts a CQT spectrogram from a given audio file.
    Args:
        audio_file (str): The path to the audio file to extract the spectrogram from.
    Returns:
        np.ndarray: A 2D numpy array representing the CQT spectrogram.
    """
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
    
    return cqt_spectrogram_db, sr

def extract_audio_for(type: str):
    data_dir = pathlib.Path(os.getcwd()) / 'data'
    audio_files = glob(str(data_dir / '**' / f'{type}*.wav'), recursive=True)   
    audio_files = [pathlib.Path(audio_file) for audio_file in audio_files]
    print(f'Found {len(audio_files)} audio files for {type} files')
    print(f'Total length of audio files: {get_len_audio_total(audio_files)} seconds')

    for audio_file in tqdm(audio_files):
        cqt_spectrogram, sr = extract_cqt_spectrogram(audio_file)

        save_dir = audio_file.parent / 'spectrograms'
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save the spectrogram as a black and white image
        image = Image.fromarray(cqt_spectrogram)
        image = image.convert('L')  # Convert to grayscale
        image.save(str(save_dir / f'{audio_file.stem}_spec.png'))

def main():
    extract_audio_for('sweep')
    extract_audio_for('tap')
    extract_audio_for('impulse')

if __name__ == '__main__':
    main()