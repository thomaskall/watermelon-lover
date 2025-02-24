import sounddevice as sd
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.io.wavfile as wavfile
from scipy.io.wavfile import write
import os

print(sd.query_devices()) #use this to list audio devices if needed

# Sampling rate (Hz), choose a rate compatible with your device (usually 44100 or 16000 Hz)
file_index = 0
file_path = f"./melon_audio_data/melon_audio_"
while os.path.exists(f"{file_path}{file_index}.wav"):
    file_index += 1
OUTPUT_FILE = f"{file_path}{file_index}.wav"  # Output .wav file to save the recorded audio

SAMPLE_RATE = 44100  # Your audio device's sample rate
DURATION = 5  # Duration of the recording in seconds
WAV_FILE = 'sine_50Hz_to_400Hz.wav'  # Input .wav file for playback
CHANNELS = 1  # Mono audio input/output

input_device_index = 1  # Replace with the index of your input device
output_device_index = 1  # Replace with the index of your output device

# Setting the input and output devices for recording and playback
sd.default.device = (input_device_index, output_device_index)
sd.default.samplerate = SAMPLE_RATE

# Load the .wav file for playback
def load_wav(filename):
    with wave.open(filename, 'rb') as wf:
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        audio_data = np.frombuffer(wf.readframes(num_frames), dtype=np.int16)
    return audio_data, sample_rate

# Load the sine wave sweep for playback
audio_to_play, play_sample_rate = load_wav(WAV_FILE)

# Ensure audio length matches the duration by repeating it
num_repeats = int(np.ceil(SAMPLE_RATE * DURATION / len(audio_to_play)))
audio_to_play = np.tile(audio_to_play, num_repeats)

# Make sure the data fits the duration (adjust if needed)
audio_to_play = audio_to_play[:SAMPLE_RATE * DURATION]

# Play and record simultaneously using playrec
print("Playing and recording simultaneously...")
recorded_audio = sd.playrec(audio_to_play, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocking=True)

print(f"Min: {recorded_audio.min()}")
print(f"Max: {recorded_audio.max()}")
np.savetxt('array', recorded_audio, delimiter=',')



wavfile.write(OUTPUT_FILE,SAMPLE_RATE,recorded_audio)




# Save the recorded audio to a .wav file
# write(OUTPUT_FILE, SAMPLE_RATE, recorded_audio)
print(f"Recorded audio saved to: {OUTPUT_FILE}")
# Perform FFT on the recorded signal
# Flatten the recorded audio to 1D if necessary (it's mono in this case, so it's already 1D)
audio_data = recorded_audio.flatten()

# Compute the FFT (Fast Fourier Transform)
fft_result = fft(audio_data)
fft_freq = fftfreq(len(fft_result), d=1/SAMPLE_RATE)

# Only keep the positive frequencies (real part of FFT)
positive_freq = fft_freq[:len(fft_freq)//2]
positive_fft = np.abs(fft_result[:len(fft_result)//2])

# Find the peak frequency in the FFT result
peak_index = np.argmax(positive_fft)  # Find index of the maximum value in the FFT
peak_frequency = positive_freq[peak_index]  # Get the corresponding frequency

# Print the peak frequency
print(f"Peak Frequency: {peak_frequency} Hz")

def moving_average(data, window_size):
    if window_size % 2 == 0: #needs to be odd
        window_size += 1
    smoothed_data = np.copy(data)
    half_window = window_size // 2
    for i in range(half_window, len(data) - half_window):
        smoothed_data[i] = np.mean(data[i - half_window:i + half_window + 1])
    # For the boundaries, just use the available neighboring values
    smoothed_data[:half_window] = np.mean(data[:window_size])
    smoothed_data[-half_window:] = np.mean(data[-window_size:])
    return smoothed_data
 


smoothed_magnitudes1 = moving_average(positive_fft, window_size=7)



# Plot the FFT of the recorded signal
plt.figure(figsize=(10, 6))
dbvalues = 20 * np.log10(smoothed_magnitudes1+1e-10)
plt.plot(positive_freq, smoothed_magnitudes1)
plt.title('Smoothed FFT of Recorded Audio')
plt.xlabel('Frequency (Hz)')
plt.xlim((50, 800))
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.savefig(f"{file_path}{file_index}.png")
