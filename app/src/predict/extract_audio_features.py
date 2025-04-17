import pandas as pd
import librosa as lib
from librosa.feature import mfcc, zero_crossing_rate, rms, spectral_rolloff, spectral_bandwidth, spectral_centroid, spectral_flatness
from librosa.onset import onset_strength as onset
import time
import pathlib
from glob import glob as glob # glob
from tqdm import tqdm
import os
import re 
from utils import get_device
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


def extract_audio_features(audio_file: pathlib.Path) -> pd.DataFrame:
    """
    Extracts audio features from a given audio file and returns a pandas DataFrame.
    Features extracted represent a small window of the audio file.

    Args:
        audio_file (str): The path to the audio file to extract features from.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the extracted features.
    """
    global total_len_audio_extracted
    
    # Load audio file
    signal, sr = lib.load(str(audio_file.absolute()))
    dur = lib.get_duration(y=signal, sr=sr)
    frames = signal.shape[0]

    total_len_audio_extracted += dur

    # Extract features
    # Mel-Frequency Cepstral Coefficients (MFCCs) and related stats
    mfccs = mfcc(y=signal, n_mfcc=30, sr=sr, n_fft=1024).T
    # Energy Information
    zcr = zero_crossing_rate(y=signal).T
    rootms = rms(y=signal).T
    onset_strength = onset(y=signal, sr=sr).T
    low_freq = spectral_rolloff(y=signal, sr=sr,  n_fft=1024, roll_percent=0.01).T
    high_freq = spectral_rolloff(y=signal, sr=sr,  n_fft=1024, roll_percent=0.99).T
    centroid = spectral_centroid(y=signal, sr=sr, n_fft=1024).T
    bandwidth = spectral_bandwidth(y=signal, sr=sr, n_fft=1024).T
    noise = spectral_flatness(y=signal, n_fft=1024).T  

    # Extract watermelon ID from filepath
    filepath_str = str(audio_file)
    watermelon_id_match = re.search(r'session_([wW]\d+)', filepath_str)
    watermelon_id = watermelon_id_match.group(1) if watermelon_id_match else 'unknown'

    # Setting the data to be exported
    df = pd.DataFrame(mfccs, columns=[f'MFCC_{i}' for i in range(30)])
    df['ZCR'] = zcr
    df['RMS'] = rootms
    df['onset_strength'] = onset_strength
    df['low_freq'] = low_freq
    df['high_freq'] = high_freq
    df['centroid'] = centroid
    df['bandwidth'] = bandwidth
    df['noise'] = noise
    df['file_name'] = audio_file.stem
    df['watermelon_id'] = watermelon_id

    return df

def extract_audio_for(type: str):
    data_dir = pathlib.Path(os.getcwd()) / 'data'
    audio_files = glob(str(data_dir / '**' / f'{type}*.wav'), recursive=True)   
    audio_files = [pathlib.Path(audio_file) for audio_file in audio_files]
    print(f'Found {len(audio_files)} audio files for {type} files')
    print(f'Total length of audio files: {get_len_audio_total(audio_files)} seconds')

    df = pd.concat([extract_audio_features(audio_file) for audio_file in tqdm(audio_files)])
    df.to_csv(str(data_dir / f'{type}_audio_features.csv'), index=False)

def main():
    extract_audio_for('sweep')
    extract_audio_for('tap')
    extract_audio_for('impulse')

if __name__ == '__main__':
    main()