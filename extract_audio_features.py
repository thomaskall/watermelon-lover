import pandas as pd
import librosa as lib
from librosa.feature import mfcc, zero_crossing_rate, rms, spectral_rolloff, spectral_bandwidth, spectral_centroid, spectral_flatness
from librosa.onset import onset_strength as onset
import time
import pathlib
from glob import glob as glob
from tqdm import tqdm
import os
import re 
import numpy as np
import concurrent.futures
from functools import partial

start = time.time()

total_len_audio_total = 0
total_len_audio_extracted = 0

metadata_file = pathlib.Path(os.getcwd()) / 'data' / 'metadata.csv'
metadata = pd.read_csv(metadata_file)

def get_len_audio_total(audio_files: list[str]) -> float:
    """
    Returns the total length of all audio files in seconds.
    """
    global total_len_audio_total
    total_len_audio_total = sum([lib.get_duration(y=lib.load(str(audio_file.absolute()))[0], sr=lib.load(str(audio_file.absolute()))[1]) for audio_file in audio_files])
    return total_len_audio_total

def analyze_frequency_bins(signal: np.ndarray, sr: int, n_bins: int = 128) -> dict:
    """
    Analyzes frequency bins of the signal and returns statistics for each bin.
    
    Args:
        signal (np.ndarray): Audio signal
        sr (int): Sample rate
        n_bins (int): Number of frequency bins to create
        
    Returns:
        dict: Dictionary containing bin statistics
    """
    # Compute the Fourier Transform
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    magnitudes = np.abs(fft)
    
    # Create frequency bins
    max_freq = sr/2
    bin_edges = np.linspace(0, max_freq, n_bins + 1)
    
    bin_stats = {
        'avg_magnitude': [],
        'max_magnitude': [],
        'freq_at_max': []
    }
    
    for i in range(n_bins):
        # Get indices of frequencies in this bin
        mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i+1])
        bin_freqs = freqs[mask]
        bin_mags = magnitudes[mask]
        
        if len(bin_mags) > 0:
            bin_stats['avg_magnitude'].append(np.mean(bin_mags))
            max_idx = np.argmax(bin_mags)
            bin_stats['max_magnitude'].append(bin_mags[max_idx])
            bin_stats['freq_at_max'].append(bin_freqs[max_idx])
        else:
            bin_stats['avg_magnitude'].append(0)
            bin_stats['max_magnitude'].append(0)
            bin_stats['freq_at_max'].append(0)
    
    return bin_stats

def extract_audio_features(audio_file: pathlib.Path) -> pd.DataFrame:
    """
    Extracts audio features from a given audio file and returns a pandas DataFrame.
    Features are extracted from the entire audio signal.

    Args:
        audio_file (str): The path to the audio file to extract features from.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the extracted features.
    """
    global total_len_audio_extracted
    
    # Load audio file
    signal, sr = lib.load(str(audio_file.absolute()))
    dur = lib.get_duration(y=signal, sr=sr)
    total_len_audio_extracted += dur

    # Extract features for entire signal
    mfccs = mfcc(y=signal, n_mfcc=30, sr=sr, n_fft=len(signal)).mean(axis=1)
    zcr = zero_crossing_rate(y=signal).mean()
    rootms = rms(y=signal).mean()
    onset_strength = onset(y=signal, sr=sr).mean()
    low_freq = spectral_rolloff(y=signal, sr=sr, n_fft=len(signal), roll_percent=0.01).mean()
    high_freq = spectral_rolloff(y=signal, sr=sr, n_fft=len(signal), roll_percent=0.99).mean()
    centroid = spectral_centroid(y=signal, sr=sr, n_fft=len(signal)).mean()
    bandwidth = spectral_bandwidth(y=signal, sr=sr, n_fft=len(signal)).mean()
    noise = spectral_flatness(y=signal, n_fft=len(signal)).mean()

    # Extract watermelon ID from filepath
    filepath_str = str(audio_file)
    watermelon_id_match = re.search(r'session_([wW]\d+)', filepath_str)
    watermelon_id = watermelon_id_match.group(1) if watermelon_id_match else 'unknown'

    # Extract weight from metadata
    try:
        weight = metadata[metadata['watermelon_id'] == watermelon_id]['weight'].values[0]
    except:
        weight = 0
    try:
        brix_score = metadata[metadata['watermelon_id'] == watermelon_id]['brix_score'].values[0]
    except:
        brix_score = 0
    
    # Get frequency bin statistics
    bin_stats = analyze_frequency_bins(signal, sr)

    # Create DataFrame with all features
    features = {
        'watermelon_id': watermelon_id,
        'weight': weight,
        **{f'MFCC_{i}': mfccs[i] for i in range(30)},
        'ZCR': zcr,
        'RMS': rootms,
        'onset_strength': onset_strength,
        'low_freq': low_freq,
        'high_freq': high_freq,
        'centroid': centroid,
        'bandwidth': bandwidth,
        'noise': noise,
        # 'file_name': audio_file.stem,
    }

    # Add frequency bin features
    for i in range(len(bin_stats['avg_magnitude'])):
        features[f'bin_{i}_avg_magnitude'] = bin_stats['avg_magnitude'][i]
        features[f'bin_{i}_max_magnitude'] = bin_stats['max_magnitude'][i]
        features[f'bin_{i}_freq_at_max'] = bin_stats['freq_at_max'][i]

    # Add brix score 
    features['brix_score'] = brix_score

    return pd.DataFrame([features])

def extract_audio_features_for(type: str):
    data_dir = pathlib.Path(os.getcwd()) / 'data'
    audio_files = glob(str(data_dir / '**' / f'{type}*.wav'), recursive=True)   
    audio_files = [pathlib.Path(audio_file) for audio_file in audio_files]
    print(f'Found {len(audio_files)} audio files for {type} files')
    print(f'Total length of audio files: {get_len_audio_total(audio_files)} seconds')

    # Process files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Create a partial function with the extract_audio_features function
        process_file = partial(extract_audio_features)
        
        # Submit all tasks and get futures
        futures = [executor.submit(process_file, audio_file) for audio_file in audio_files]
        
        # Collect results as they complete
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {type} files"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing file: {e}")

    # Combine all results into a single DataFrame
    if results:
        df = pd.concat(results, ignore_index=True)
        df.to_csv(str(data_dir / f'{type}_audio_features.csv'), index=False)
    else:
        print(f"No results were obtained for {type} files")

def main():
    for type in ['sweep', 'tap', 'impulse']:
        extract_audio_features_for(type)

if __name__ == '__main__':
    main()