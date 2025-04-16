import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd


# Parameters
duration = 5  # duration of the sweep in seconds
start_freq = 50  # starting frequency in Hz
end_freq = 400  # ending frequency in Hz
sample_rate = 44100  # samples per second (standard for audio)

# Time vector
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Frequency sweep (linear interpolation between start_freq and end_freq)
# frequencies = np.linspace(start_freq, end_freq, len(t))

# Logarithmic frequency sweep
frequencies = np.logspace(np.log10(start_freq), np.log10(end_freq), len(t))

# Generate the audio signal (sinusoidal wave)
audio_signal = np.sin(2 * np.pi * frequencies * t)

# Normalize to the range of float32 (-1 to 1)
audio_signal_normalized = np.int16(audio_signal * 32767)

# Write the signal to a .wav file
write("sine_50Hz_to_400Hz.wav", sample_rate, audio_signal_normalized)

print("Wave file generated successfully: sine_50Hz_to_400Hz.wav")


sd.play(audio_signal_normalized, samplerate=sample_rate)
sd.wait()
